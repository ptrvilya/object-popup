import logging
import pickle as pkl
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import splprep, splev
from tqdm import tqdm

from .generator import fit_obj_to_locations


class TemporalGenerator:
    def __init__(self, network, device, gen_datasets, cfg,
                 canonical_obj_meshes, canonical_obj_keypoints):
        self.gen_datasets = gen_datasets
        self.network = network.to(device)
        self.network.eval()
        self.device = device

        self.cfg = cfg
        self.network = self.load_checkpoint(self.network, cfg.checkpoint_path)

        self.objname2classid = cfg.objname2classid
        self.classid2objname = {v: k for k, v in self.objname2classid.items()}
        self.n_onehot_classes = len(self.classid2objname)

        self.canonical_obj_meshes = canonical_obj_meshes
        self.canonical_obj_keypoints = {k: torch.tensor(v["cartesian"], dtype=torch.float) for k, v in canonical_obj_keypoints.items()}

    @staticmethod
    def load_checkpoint(network, checkpoint_path):
        logging.info(f'Loaded checkpoint from: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        return network

    def get_sequence_list_from_dataset(self, data):
        # get (sbj, obj, act) triplets from the data directory
        sequence_triplets = set()
        for sample in data:
            sequence_triplets.add((sample.subject, sample.object, sample.action))

        return sorted(list(sequence_triplets))

    @staticmethod
    def load_sequence(sequence_path, obj_class):
        t_stamps = sorted(list(sequence_path.glob("t*")))
        data = {
            "sbj_point_cloud": [], "path": [], "obj_class": [],
            "t": [], "R": [], "preprocess_scale": []
        }

        for t_stamp in t_stamps:
            sbj_point_cloud = np.load(t_stamp / "subject_pointcloud.npz")["subject_pointcloud"]

            with (t_stamp / "preprocess_transform.pkl").open("rb") as fp:
                preprocess_transform = pkl.load(fp)

            data["sbj_point_cloud"].append(sbj_point_cloud.astype(np.float32))
            data["path"].append(t_stamp)
            data["obj_class"].append(obj_class)
            data["t"].append(preprocess_transform["translation"] + preprocess_transform["t"])
            data["R"].append(preprocess_transform["R"])
            data["preprocess_scale"].append(preprocess_transform["scale"])

        data["sbj_point_cloud"] = torch.tensor(np.stack(data["sbj_point_cloud"], axis=0), dtype=torch.float)
        data["t"] = torch.tensor(np.stack(data["t"], axis=0), dtype=torch.float).reshape(-1, 3)
        data["R"] = torch.tensor(np.stack(data["R"], axis=0), dtype=torch.float).reshape(-1, 3, 3)
        data["preprocess_scale"] = torch.tensor(np.stack(data["preprocess_scale"], axis=0), dtype=torch.float)
        data["obj_class"] = torch.tensor(np.stack(data["obj_class"], axis=0), dtype=torch.long)

        return data, len(t_stamps)

    def generate_offline(self, spline_k=3, spline_s=0.1):
        for dataset_name, dataset_path, data in self.gen_datasets:
            sequences = self.get_sequence_list_from_dataset(data)

            for (sbj, obj, act) in tqdm(sequences):
                sequence_path = dataset_path / sbj / f"{obj}_{act}"

                # reset sequence wide parameters
                first_pass_centers = []
                first_pass_classes = []
                first_pass_features = []

                # load sequence
                sequence, T = self.load_sequence(sequence_path, self.cfg.objname2classid[obj])

                # ============ 1 First pass [predicting centers and classes]
                for index in range(0, T, self.cfg.batch_size):
                    sbj_pc = sequence['sbj_point_cloud'][index : index + self.cfg.batch_size]

                    if self.cfg.model_name == "object_popup" and self.cfg.model_params.get("with_classifier", False):
                        with torch.no_grad():
                            enc_subject, pred_center, pred_class = \
                                self.network.encode_1st_stage(
                                    sbj_pc.to(self.device)
                                )

                            pred_class = torch.nn.functional.softmax(pred_class)
                            first_pass_centers.append(pred_center.cpu())
                            first_pass_features.append(enc_subject.cpu())
                            first_pass_classes.append(pred_class.cpu())
                    elif self.cfg.model_name == "object_popup":
                        obj_classids = sequence['obj_class'][index : index + self.cfg.batch_size]

                        with torch.no_grad():
                            enc_subject, pred_center, _ = \
                                self.network.encode_1st_stage(
                                    sbj_pc.to(self.device)
                                )

                            first_pass_centers.append(pred_center.cpu())
                            first_pass_features.append(enc_subject.cpu())
                            first_pass_class_id = obj_classids[0].cpu().item()
                    else:
                        raise ValueError(f"Unknown model name {self.cfg.model_name}")

                first_pass_centers = torch.cat(first_pass_centers, dim=0)
                first_pass_centers /= sequence['preprocess_scale'].reshape(T, 1, 1)
                first_pass_features = torch.cat(first_pass_features, dim=0)
                # UNDO preprocessing
                first_pass_centers = torch.bmm(
                    torch.transpose(sequence["R"], 2, 1),
                    first_pass_centers.reshape(T, 3, 1) - sequence["t"].unsqueeze(2),
                )
                # ===========================================

                # ============ 2 smooth class and centers before the second stage
                # spline interpolation
                if T > 2 * spline_k:
                    first_pass_centers = first_pass_centers.squeeze(2).numpy().transpose(1, 0)
                    tck, u = splprep(first_pass_centers, k=spline_k, s=spline_s)
                    first_pass_centers = np.stack(splev(u, tck), axis=1)
                    first_pass_centers = torch.tensor(first_pass_centers, dtype=torch.float).unsqueeze(2)
                # maximum over the sequence
                if self.cfg.model_name == "object_popup" and self.cfg.model_params.get("with_classifier", False):
                    first_pass_classes = torch.cat(first_pass_classes, dim=0)
                    first_pass_classes_ids = first_pass_classes.argmax(1)
                    first_pass_class_id = np.argmax(np.bincount(first_pass_classes_ids.numpy()))
                first_pass_class_probs = torch.zeros((1, len(self.classid2objname)), dtype=torch.float)
                first_pass_class_probs[0, first_pass_class_id] = 1.0
                # ===========================================

                # ============ 3 Second pass [predicting keypoints with smoothed classes and centers]
                second_pass_keypoints_in = []
                second_pass_predicted_locations = []
                for index in range(0, T, self.cfg.batch_size):
                    if self.cfg.model_name == "object_popup":
                        with torch.no_grad():
                            # REDO preprocess
                            centers = first_pass_centers[index : index + self.cfg.batch_size]
                            obj_scales = sequence['preprocess_scale'][index : index + self.cfg.batch_size]
                            R = sequence["R"][index : index + self.cfg.batch_size]
                            t = sequence["t"][index : index + self.cfg.batch_size]

                            B = centers.size(0)
                            obj_scales = obj_scales.reshape(B, 1, 1)
                            centers = obj_scales * torch.baddbmm(t.unsqueeze(2), R, centers)
                            centers = centers.reshape(B, 1, 3).to(self.device)

                            # create obj class ids tensor
                            obj_classids = torch.tensor(np.array([first_pass_class_id]), dtype=torch.int64)
                            obj_classids = obj_classids.repeat(B, 1)

                            # select keypoints
                            obj_keypoints_in = torch.clone(self.canonical_obj_keypoints[first_pass_class_id]).float()
                            obj_keypoints_in = obj_keypoints_in.to(self.device)
                            obj_keypoints_in = obj_keypoints_in.unsqueeze(0).repeat(B, 1, 1)
                            obj_keypoints_in = obj_scales.to(self.device) * obj_keypoints_in
                            obj_keypoints_in = obj_keypoints_in - obj_keypoints_in.mean(dim=1, keepdims=True) + centers

                            sbj_pc = sequence['sbj_point_cloud'][index : index + self.cfg.batch_size]
                            if self.cfg.model_params.get("with_classifier", False):
                                # run encode 2 with new keypoints
                                enc_object, enc_sbjobj, _ = self.network.encode_2nd_stage(
                                    sbj_pc.to(self.device), centers, pred_class=None, obj_center=None,
                                    obj_classids=obj_classids.to(self.device),
                                    obj_scales=None, obj_keypoints=obj_keypoints_in
                                )
                            else:
                                enc_object, enc_sbjobj, _ = self.network.encode_2nd_stage(
                                    sbj_pc.to(self.device), centers, pred_class=None, obj_center=None,
                                    obj_classids=obj_classids.to(self.device),
                                    obj_scales=None, obj_keypoints=obj_keypoints_in
                                )

                            enc_subject = first_pass_features[index : index + self.cfg.batch_size]
                            output = self.network.decode(
                                enc_subject.to(self.device), enc_object, enc_sbjobj, centers, obj_keypoints_in
                            )

                            obj_keypoints_in = obj_keypoints_in.cpu()
                            if self.cfg.model_params.get("decoder_type", "") == "offsets":
                                obj_keypoints_offsets = output["offsets"].cpu()
                                obj_keypoints_locations = (obj_keypoints_in + obj_keypoints_offsets)

                            second_pass_keypoints_in.append(obj_keypoints_in)
                            second_pass_predicted_locations.append(obj_keypoints_locations)
                    else:
                        raise ValueError(f"Unknown model name {self.cfg.model_name}")

                # UNDO preprocess transform
                second_pass_keypoints_in = torch.cat(second_pass_keypoints_in, dim=0)
                second_pass_predicted_locations = torch.cat(second_pass_predicted_locations, dim=0)
                second_pass_predicted_locations /= sequence['preprocess_scale'].reshape(T, 1, 1)
                second_pass_predicted_locations = torch.bmm(
                    torch.transpose(sequence["R"], 2, 1),
                    second_pass_predicted_locations.reshape(T, 3, -1) - sequence["t"].unsqueeze(2),
                )

                second_pass_keypoints_in = second_pass_keypoints_in.reshape(T, -1, 3)
                # ===========================================

                # ============ 4 smooth predicted keypoints
                if T > 2 * spline_k:
                    second_pass_predicted_locations = second_pass_predicted_locations.numpy().transpose(2, 1, 0)
                    n_points = second_pass_predicted_locations.shape[0]
                    smoothed_locations = np.zeros((T, 3, n_points), dtype=np.float32)
                    for i in range(n_points):
                        tck, u = splprep(second_pass_predicted_locations[i], k=spline_k, s=spline_s)
                        smoothed_locations[:, :, i] = np.stack(splev(u, tck), axis=1)
                    second_pass_predicted_locations = torch.tensor(smoothed_locations, dtype=torch.float)
                # ===========================================

                # ============ 5 save results
                vis_folder = \
                    self.cfg.exp_folder / "visualization" / f"{str(self.cfg.epoch)}_temporal" \
                    / dataset_name / sbj / f"{obj}_{act}"
                vis_folder.mkdir(parents=True, exist_ok=True)
                # REDO preprocess transform
                second_pass_keypoints_in = second_pass_keypoints_in.numpy()
                second_pass_predicted_locations = \
                    sequence['preprocess_scale'].reshape(T, 1, 1) * \
                    torch.baddbmm(sequence["t"].unsqueeze(2), sequence["R"], second_pass_predicted_locations)
                second_pass_predicted_locations = second_pass_predicted_locations.reshape(T, -1, 3).numpy()

                for index in range(T):
                    obj_keypoints_in = second_pass_keypoints_in[index]
                    obj_keypoints_locations = second_pass_predicted_locations[index]
                    obj_mesh = deepcopy(self.canonical_obj_meshes[first_pass_class_id])

                    obj_mesh.vertices *= sequence['preprocess_scale'][index].item()
                    obj_mesh_verts = np.array(obj_mesh.vertices)
                    obj_mesh.vertices = \
                        obj_mesh_verts - obj_mesh_verts.mean(axis=0) + \
                        obj_keypoints_in.mean(axis=0, keepdims=True)

                    predicted_mesh = fit_obj_to_locations(
                        obj_mesh, obj_keypoints_in, obj_keypoints_locations
                    )

                    # save results
                    if self.cfg.model_name == "object_popup":
                        if self.cfg.model_params.get("with_classifier", False):
                            predicted_class_path = vis_folder / "class_scores" / f"t{index:05d}.npy"
                            predicted_class_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(predicted_class_path, first_pass_class_probs.numpy())

                        posed_mesh_path = vis_folder / "posed_mesh" / f"t{index:05d}.obj"
                        posed_mesh_path.parent.mkdir(parents=True, exist_ok=True)

                        predicted_mesh.export(str(posed_mesh_path))
                # ===========================================
