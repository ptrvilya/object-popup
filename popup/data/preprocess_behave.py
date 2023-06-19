"""
Code to preprocess 1fps annotations for the BEAHVE dataset.
"""
import argparse
import json
import pickle as pkl
import warnings
from collections import defaultdict
from copy import deepcopy
from multiprocessing import set_start_method
from pathlib import Path, PurePath
from types import SimpleNamespace

import igl
import numpy as np
import smplx
import tomlkit
import torch
import tqdm
from scipy.spatial.transform import Rotation

from ..utils.preprocess import tensor_to_cpu, estimate_transform, DatasetSample, preprocess_worker, \
    preprocess_worker_rawpc, get_sequences_list, th_posemap_axisang, generate_obj_keypoints, \
    trimesh_load, generate_behave_canonicalized_objects
from ..utils.parallel_map import parallel_map


def preprocess(cfg):
    set_start_method('spawn')

    # list dataset sequences
    _sequences = get_sequences_list(
        "behave", cfg.input_path / "sequences", objects=cfg.preprocess_behave["objects"], subjects=cfg.preprocess_behave["subjects"]
    )

    # filter sequences based on split
    if cfg.preprocess_behave["split"] in ["train", "test"]:
        with (cfg.preprocess_behave["split_file"]).open("r") as fp:
            split = json.load(fp)
        split_sequences = split[cfg.preprocess_behave["split"]]
        sequences = [seq for seq in _sequences if seq.name in split_sequences]
    else:
        sequences = _sequences
    print(cfg.preprocess_behave["split"], len(sequences))
    # preprocess each sequence
    contact_masks = {}
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # load sequence info
        with (sequence / "info.json").open("r") as fp:
            sequence_info = json.load(fp)  # 'cat', 'gender'

        # parse sequence name: Date<:02d>_Sub<:02d>_<object>_<optional:action>
        sequence_name = sequence.name.split("_")
        seq_date = sequence_name[0]
        seq_subject = sequence_name[1]
        seq_action = f"{sequence_name[3]}_{seq_date}" if len(sequence_name) == 4 else f"{seq_date}"
        seq_object = sequence_info["cat"]

        # Dataset structure
        # person/fit02/person_fit.pkl: ['pose', 'betas', 'trans', 'score']
        #   shapes: 156, 10, 3, 0
        # <object>/fit01/<object>_fit.pkl: ['angle', 'trans']
        #   shapes: 3, 3

        # ============ 1 extract vertices for subject
        t_stamps = sorted(list(sequence.glob("t????.000")))
        T = len(t_stamps)
        preprocess_transforms = []
        if cfg.input_type in ["smplh", "smpl"]:
            # load sbj mesh to use as a template
            sbj_mesh = trimesh_load(t_stamps[0] / "person/fit02/person_fit.ply")

            # create smplh model
            sbj_model = smplx.build_layer(
                model_path=str(cfg.SMPLX_PATH), model_type="smplh", gender=sequence_info["gender"],
                use_pca=False, num_betas=10, batch_size=T,
            )

            # load smpl(-h) parameters
            smpl_params = defaultdict(list)
            for t_stamp in t_stamps:
                with (t_stamp / "person/fit02/person_fit.pkl").open("rb") as fp:
                    model_params = pkl.load(fp)
                smpl_params["betas"].append(model_params["betas"])
                smpl_params["pose"].append(model_params["pose"])
                smpl_params["trans"].append(model_params["trans"])
            smpl_params = {k: np.array(v, dtype=np.float32) for k, v in smpl_params.items()}

            # convert parameters
            th_pose_axisangle = torch.tensor(smpl_params["pose"]).reshape(T, 52, 3)
            th_pose_rotmat = th_posemap_axisang(th_pose_axisangle.reshape(T * 52, 3)).reshape(T, 52, 9)
            body_model_params = {
                "betas": torch.tensor(smpl_params['betas']),
                "transl": torch.tensor(smpl_params["trans"]),
                "global_orient": th_pose_rotmat[:, :1].reshape(T, -1, 9),
                "body_pose": th_pose_rotmat[:, 1:22].reshape(T, -1, 9),
                "left_hand_pose": th_pose_rotmat[:, 22:37].reshape(T, -1, 9),
                "right_hand_pose": th_pose_rotmat[:, 37:].reshape(T, -1, 9),
            }
            if cfg.input_type == "smpl":
                body_model_params["left_hand_pose"] = None
                body_model_params["right_hand_pose"] = None

            # get smpl(-h) vertices
            sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
            sbj_verts = tensor_to_cpu(sbj_output.vertices)
            sbj_joints = tensor_to_cpu(sbj_output.joints)

            # align sbj meshes
            raw_pc_vertices = []
            for i in range(T):
                # custom rotation to align with grab data
                R_grab = Rotation.from_euler('x', [-90], degrees=True)

                if cfg.preprocess_behave["align_with_joints"]:
                    sbj_joints[i] = R_grab.apply(sbj_joints[i])

                    # rotation to align torso with x axis direction
                    shoulderblades = sbj_joints[i][17] - sbj_joints[i][16]
                    z = np.array([0, 0, 1])
                    dir_shoulderblades = np.cross(z, shoulderblades)
                    dir_shoulderblades[2] = 0.0  # project to xy
                    dir_shoulderblades = dir_shoulderblades / np.linalg.norm(dir_shoulderblades)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        R_align, _ = Rotation.align_vectors(np.array([[1, 0, 0]]), dir_shoulderblades[None])
                    # q followed by p is equivalent to p * q.
                    R = R_align * R_grab
                    sbj_verts[i] = R.apply(sbj_verts[i])
                    sbj_joints[i] = R_align.apply(sbj_joints[i])

                    t_reset = -1 * np.copy(sbj_joints[i][0])  # center using root joint
                    sbj_joints[i] += t_reset
                    sbj_verts[i] += t_reset
                else:
                    # only align using center
                    R_align = Rotation.from_matrix(np.eye(3, dtype=np.float32))
                    R = R_align * R_grab
                    sbj_verts[i] = R.apply(sbj_verts[i])
                    sbj_joints[i] = R.apply(sbj_joints[i])

                    t_reset = -1 * np.mean(sbj_verts[i], axis=0)
                    sbj_joints[i] += t_reset
                    sbj_verts[i] += t_reset

                preprocess_transforms.append({
                    "R": R.as_matrix(),
                    "t": np.copy(t_reset)
                })

                if cfg.preprocess_behave["use_raw_pcs"]:
                    raw_pc = np.array(trimesh_load(t_stamps[i] / "person/person.ply").vertices)
                    raw_pc = R.apply(raw_pc) + t_reset
                    raw_pc_vertices.append(raw_pc)

            sbj_verts = np.stack(sbj_verts, axis=0)
        else:
            raise ValueError(f"Unsupported input data type {cfg.input_type}")
        # ===========================================

        # ============ 2 extract vertices for object
        # mapping from class names to saved mesh names
        if seq_object in ["chairblack", "chairwood"]:
            _object_name = "chair"
        elif seq_object == "basketball":
            _object_name = "sports ball"
        elif seq_object == "yogaball":
            _object_name = "sports ball"
        else:
            _object_name = seq_object

        # load object mesh and transform using sbj transformation
        obj_verts = []
        for index, t_stamp in enumerate(t_stamps):
            obj_mesh = trimesh_load(t_stamp / f"{_object_name}/fit01/{_object_name}_fit.ply")

            obj_mesh_v = np.array(obj_mesh.vertices)
            R = Rotation.from_matrix(preprocess_transforms[index]["R"])
            t = preprocess_transforms[index]["t"]
            obj_mesh.vertices = R.apply(obj_mesh_v) + t
            obj_verts.append(np.array(obj_mesh.vertices))
        obj_verts = np.stack(obj_verts, axis=0)
        obj_mesh = trimesh_load(cfg.behave_can_objects_path / f"{seq_object}.ply")
        # ===========================================

        # ============ 3 filter based on contacts
        # contact_mask = np.zeros(T, dtype=bool)
        # for index, t_stamp in enumerate(t_stamps):
        #     fit_contact = np.load(t_stamp / f"{_object_name}/fit01/{_object_name}_fit_contact.npz")
        #     contact_labels = fit_contact["contact_label"]
        #     contact_mask[index] = contact_labels.sum() > 0
        # T = contact_mask.sum()
        #
        # # if no frame is selected continue to the next sequence
        # if T < 1:
        #     continue
        #
        # sbj_verts = sbj_verts[contact_mask]
        # obj_verts = obj_verts[contact_mask]
        contact_mask = np.zeros(T, dtype=bool)
        for t_stamp in tqdm.tqdm(range(0, T), leave=False, ncols=80):
            t_sbj_verts = sbj_verts[t_stamp]
            t_obj_verts = obj_verts[t_stamp]

            t_obj_mesh = obj_mesh
            t_obj_mesh.vertices = t_obj_verts

            t_obj_points = t_obj_mesh.sample(1000)
            t_obj2sbj_d, _, _ = igl.signed_distance(t_obj_points, t_sbj_verts, sbj_mesh.faces, return_normals=False)

            if np.any(t_obj2sbj_d < cfg.preprocess_behave["contact_threshold"]):
                contact_mask[t_stamp] = True
        T = contact_mask.sum()

        # if no frame is selected continue to the next sequence
        if T < 1:
            continue

        sbj_verts = sbj_verts[contact_mask]
        obj_verts = obj_verts[contact_mask]
        # ===========================================

        # ============ 4 calculate rotation for object ============
        obj_vtemp = np.array(obj_mesh.vertices)
        obj_rotations = []
        for t_stamp in range(0, T):
            # find transform from vertices in the canonical pose to vertices in t_stamp frame
            R_t_stamp, t_t_stamp = estimate_transform(obj_vtemp, obj_verts[t_stamp])
            obj_rotations.append(R_t_stamp)
        # ===========================================

        # ============ 5 align the ground plane ============
        for i in range(T):
            z_min = min(np.min(sbj_verts[i, :, 2]), np.min(obj_verts[i, :, 2]))
            t_align_z = np.array([0.0, 0.0, -z_min], dtype=np.float32)

            preprocess_transforms[i]["t"] += t_align_z

            sbj_verts[i] += t_align_z
            obj_verts[i] += t_align_z

            if cfg.preprocess_behave["use_raw_pcs"]:
                raw_pc_vertices[i] += t_align_z
        # ==================================================

        # ============ 6 preprocess each time stamp in parallel
        sbj_pointcloud = np.copy(sbj_verts)

        # name mapping to split sequences for the same subject from different days
        if cfg.preprocess_behave["split"] == "test":
            seq_subject = f"{seq_subject}_test"
        elif cfg.preprocess_behave["split"] == "train":
            seq_subject = f"{seq_subject}_train"

        input_data = [{
            "sample": DatasetSample(
                subject=seq_subject,
                action=seq_action,
                object=seq_object,
                t_stamp=t,
                sbj_mesh=deepcopy(sbj_mesh),
                obj_mesh=deepcopy(obj_mesh),
                sbj_pc=sbj_pointcloud[t],
                obj_verts=obj_verts[t],
                obj_rotation=obj_rotations[t],
                preprocess_transforms=preprocess_transforms[t]
            ),
            "num_points_pc_subject": cfg.num_points_pc_subject,
        } for t in range(T)]

        if cfg.preprocess_behave["use_raw_pcs"]:
            raw_pc_vertices = [v for i, v in enumerate(raw_pc_vertices) if contact_mask[i]]
            for t in range(T):
                input_data[t]["raw_pc"] = raw_pc_vertices[t]

        # Same actions for each t_stamp
        preprocess_results = parallel_map(
            input_data,
            preprocess_worker_rawpc if cfg.preprocess_behave["use_raw_pcs"] else preprocess_worker,
            use_kwargs=True, n_jobs=10, tqdm_kwargs={"leave": False}
        )
        # ===========================================

        # ============ 7 Save subject-specific data
        contact_masks[f"{seq_subject}_{seq_object}_{seq_action}"] = contact_mask
        for sample in preprocess_results:
            if sample is None:
                print("\n", seq_subject, seq_object, seq_action)
                continue
            sample.dump(cfg.output_path)
        # ===========================================

    # ============ 8 Save global info
    with (cfg.output_path / "contact_masks.pkl").open("wb") as fp:
        pkl.dump(contact_masks, fp)

    # Path is not json serializable
    with (cfg.output_path / "metadata.json").open("w") as fp:
        cfg_dict = vars(cfg)
        for k, v in cfg_dict.items():
            if isinstance(v, PurePath):
                cfg_dict[k] = str(v)
        for k, v in cfg_dict['preprocess_behave'].items():
            if isinstance(v, PurePath):
                cfg_dict['preprocess_behave'][k] = str(v)

        json.dump(cfg_dict, fp, indent=4)
    # ===========================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess BEHAVE data')

    parser.add_argument('-i', "--input-path", type=Path)
    parser.add_argument('-o', "--output-path", type=Path)
    parser.add_argument('-c', "--config", type=Path, default="./project_config.toml")
    parser.add_argument('-s', "--subjects", nargs="+", type=str, help="Preprocess only selected subjects",
                        default=["*"])
    parser.add_argument('-g', "--generate-keypoints", action="store_true", help="Generate keypoints for objects.")
    parser.add_argument("--split", type=str, help="Optionally filter sequences using official train/test split.",
                        default="")
    parser.add_argument("--split-file", type=Path, help="Supply the name of the file with train/test split.",
                        default=None)

    args = parser.parse_args()

    project_config = dict(tomlkit.parse(args.config.read_text()))
    project_config["preprocess_behave"] = dict(project_config["preprocess_behave"])
    # preset directories
    # Directory with original BEHAVE objects
    project_config["behave_orig_objects_path"] = args.input_path / "objects"
    # Directory with canonicalized BEHAVE objects
    project_config["behave_can_objects_path"] = args.input_path / "canonicalized_objects"
    # additional parameters
    project_config["input_path"] = args.input_path
    project_config["output_path"] = args.output_path
    project_config["preprocess_behave"]["subjects"] = args.subjects
    project_config["preprocess_behave"]["split"] = args.split
    project_config["preprocess_behave"]["split_file"] = \
        args.split_file if args.split_file is not None else args.input_path / "split.json"

    config = SimpleNamespace(**project_config)
    # canonicalize objects using pre-computed transforms
    generate_behave_canonicalized_objects(
        config.behave_orig_objects_path, config.behave_can_objects_path
    )
    # preprocess data
    preprocess(config)
    # optionally generate object keypoints
    if args.generate_keypoints:
        generate_obj_keypoints(
            config.preprocess_behave["objects"], Path(config.behave_can_objects_path),
            config.obj_keypoints_npoints, Path(config.output_path)
        )
