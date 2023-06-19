import logging
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from ..utils.parallel_map import parallel_map
from ..utils.preprocess import estimate_transform


def fit_obj_to_locations(init_mesh, obj_keypoints_in, predicted_locations):
    init_vertices = np.array(obj_keypoints_in)
    pred_vertices = predicted_locations
    R, t = estimate_transform(init_vertices, pred_vertices)

    init_mesh.vertices = np.dot(init_mesh.vertices, R.T) + t
    transformed_mesh = init_mesh

    return transformed_mesh


def fit_obj_to_offsets(init_mesh, obj_keypoints_in, predicted_offsets):
    init_vertices = np.array(obj_keypoints_in)
    pred_vertices = np.copy(init_vertices) + predicted_offsets
    R, t = estimate_transform(init_vertices, pred_vertices)

    init_mesh.vertices = np.dot(init_mesh.vertices, R.T) + t
    transformed_mesh = init_mesh

    return transformed_mesh


class Generator:
    def __init__(self, device, cfg, canonical_obj_meshes, canonical_obj_keypoints):
        self.device = device

        self.cfg = cfg

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

    def get_mesh_from_predictions_wrapper(self, batch, output):
        return self.get_mesh_from_predictions(
            output, preprocess_scale=batch["preprocess_scale"], obj_class=batch["obj_class"],
            cfg=self.cfg, canonical_obj_keypoints=self.canonical_obj_keypoints,
            canonical_obj_meshes=self.canonical_obj_meshes
        )

    @staticmethod
    def get_mesh_from_predictions(
        output, preprocess_scale, obj_class, cfg,
        canonical_obj_keypoints, canonical_obj_meshes
    ):
        # legacy, equal to 1 for GRAB and BEHAVE
        obj_scales = preprocess_scale.float()
        pred_class_scores = None

        if cfg.model_name == "object_popup":
            if cfg.model_params.get("with_classifier", False):
                pred_class_scores = output["obj_class"]
                obj_classids = pred_class_scores.argmax(dim=0).item()
            else:
                obj_classids = obj_class.item()
            obj_keypoints_in = torch.clone(canonical_obj_keypoints[obj_classids])
            obj_mesh = deepcopy(canonical_obj_meshes[obj_classids])

            if obj_scales is not None:
                obj_keypoints_in = obj_keypoints_in * obj_scales.reshape(1, 1)
                obj_mesh.vertices = np.array(obj_mesh.vertices) * obj_scales.reshape(1, 1).numpy()

            # offset is predicted for object placed in the predicted center
            obj_mesh.vertices = \
                np.array(obj_mesh.vertices) - \
                obj_keypoints_in.mean(dim=0, keepdims=True).numpy() + \
                output["obj_center"].numpy()

            obj_keypoints_in = \
                obj_keypoints_in - obj_keypoints_in.mean(dim=0, keepdims=True) + output["obj_center"]

            if cfg.model_params.get("decoder_type", "") == "offsets":
                obj_keypoints_offsets = output["offsets"]

                predicted_mesh = fit_obj_to_offsets(
                    obj_mesh, obj_keypoints_in.numpy(), obj_keypoints_offsets.detach().numpy()
                )
            elif cfg.model_params.get("decoder_type", "") == "Rt":
                pred_R = output["R"].reshape(3, 3)
                pred_t = output["t"].reshape(3)

                pred_location = torch.matmul(pred_R, obj_keypoints_in.transpose(1, 0)).transpose(1, 0) + pred_t

                predicted_mesh = fit_obj_to_locations(
                    obj_mesh, obj_keypoints_in.numpy(), pred_location.detach().numpy()
                )
            else:
                raise RuntimeError(f"Unknown decoder_type {cfg.model_params.get('decoder_type', '')}")
        else:
            raise RuntimeError(f"Unknown model name {cfg.model_name}")

        return predicted_mesh, pred_class_scores

    def generate(self, network, gen_dataloaders):
        network = network.to(self.device).eval()
        for dataset_name, gen_dataloader in gen_dataloaders:
            for batch in tqdm(gen_dataloader, total=len(gen_dataloader), ncols=80):
                # GET OUTPUTS
                if self.cfg.model_name == "object_popup":
                    sbj = batch.get('sbj_point_cloud').to(self.device)
                    batch_size = sbj.size(0)

                    if self.cfg.model_params.get("with_classifier", False):
                        obj_classids = None
                    else:
                        obj_classids = batch.get('obj_class').to(self.device)

                    # legacy, equal to 1 for GRAB and BEHAVE
                    obj_scales = batch.get('preprocess_scale').float().to(self.device)

                    with torch.no_grad():
                        output = network(
                            sbj, obj_classids, obj_keypoints=None, obj_scales=obj_scales, obj_center=None
                        )
                else:
                    raise ValueError(f"Unknown model name {self.cfg.model_name}")

                # reshape output data
                output = {k: v.cpu() for k, v in output.items()}
                # dict of lists to list of lists
                output = [dict(zip(output, d)) for d in zip(*output.values())]

                # GET MESH FROM PREDICTIONS
                # run in parallel
                input_data = [{
                    "output": output[i],
                    "preprocess_scale": batch["preprocess_scale"][i].float(),
                    "obj_class": batch["obj_class"][i],
                    "cfg": self.cfg,
                    "canonical_obj_keypoints": self.canonical_obj_keypoints,
                    "canonical_obj_meshes": self.canonical_obj_meshes
                } for i in range(batch_size)]

                predictions = parallel_map(
                    input_data, self.get_mesh_from_predictions, use_kwargs=True,
                    n_jobs=self.cfg.workers, tqdm_kwargs={"leave": False}
                )

                # SAVE MESH
                for batch_i, (pred_mesh, pred_class_scores) in enumerate(predictions):
                    # create directory for visualization
                    input_path = batch.get("path")[batch_i].split("/")
                    subject, obj_action, t_stamp = input_path[-3], input_path[-2], input_path[-1]
                    vis_folder = self.cfg.exp_folder / "visualization" / str(
                        self.cfg.epoch) / dataset_name / subject / obj_action
                    vis_folder.mkdir(parents=True, exist_ok=True)

                    if self.cfg.model_name == "object_popup":
                        # save mesh
                        posed_mesh_path = vis_folder / "posed_mesh" / f"{t_stamp}.obj"
                        posed_mesh_path.parent.mkdir(parents=True, exist_ok=True)

                        pred_mesh.export(str(posed_mesh_path))

                        # save object class
                        if self.cfg.model_params.get("with_classifier", False):
                            predicted_class_path = vis_folder / "class_scores" / f"{t_stamp}.npy"
                            predicted_class_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(predicted_class_path, pred_class_scores)
                    else:
                        raise ValueError(f"Unknown model {self.cfg.model_name}")
