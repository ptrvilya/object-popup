"""
Code to preprocess annotations for the GRAB dataset.
"""
import argparse
import json
import pickle as pkl
import warnings
from copy import deepcopy
from multiprocessing import set_start_method
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import smplx
import tomlkit
import tqdm
import trimesh
from scipy.spatial.transform import Rotation

from ..utils.preprocess import tensor_to_cpu, estimate_transform, DatasetSample, preprocess_worker, \
    get_sequences_list, params_to_torch, generate_obj_keypoints, filter_contact_frames, \
    prepare_params, parse_npz, trimesh_load
from .grab_object_model import ObjectModel
from ..utils.parallel_map import parallel_map


def preprocess(cfg):
    set_start_method('spawn')

    # list dataset sequences
    sequences = get_sequences_list(
        "grab", cfg.input_path, objects=cfg.preprocess_grab["objects"], subjects=cfg.preprocess_grab["subjects"]
    )

    # preprocess each sequence
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # Parse original grab annotations
        grab_seq_data = parse_npz(sequence)
        seq_subject = str(grab_seq_data["sbj_id"])
        seq_action = "_".join(sequence.stem.split("_")[1:])
        seq_object = str(grab_seq_data["obj_name"])

        # ============ 1 extract vertices for subject
        preprocess_transforms = []
        if cfg.input_type in ["smplh", "smpl"]:
            # path to custom converted smplh meshes
            seq_name = f"{seq_object}_{seq_action}"
            smplh_sequence_folder = cfg.input_path / "grab_smplh" / seq_subject / seq_name

            # load sequence data for smplh
            with (smplh_sequence_folder / "sequence_data.pkl").open("rb") as fp:
                smplh_seq_data = pkl.load(fp)

            # frame mask for smplh meshes
            frame_mask = smplh_seq_data["frame_mask"]
            T_init = frame_mask.sum()

            # initial mask is at 30 fps
            # NOTE: smplh annotations are downsampled from 120 fps to 30 fps
            smplh_mask = np.ones(T_init, dtype=bool)
            if cfg.preprocess_grab["downsample"] != "None":
                downsample_mask = np.zeros_like(frame_mask)
                if cfg.preprocess_grab["downsample"] == "10fps":
                    downsample_mask[::12] = True
                else:
                    downsample_mask[::4] = True

                # downsample masks
                smplh_mask = downsample_mask[np.argwhere(frame_mask)].flatten()
                frame_mask = np.logical_and(frame_mask, downsample_mask)
            T = frame_mask.sum()

            # if no frame is selected continue to the next sequence
            if T < 1:
                continue

            # generate new meshes or load existing ones
            if cfg.preprocess_grab["load_existing_sbj_meshes"]:
                sbj_verts = []

                for t_stamp in range(T_init):
                    if not(smplh_mask[t_stamp]):
                        continue
                    sbj_mesh = trimesh_load(smplh_sequence_folder / f"{t_stamp:04d}.obj")
                    sbj_verts.append(sbj_mesh.vertices)
                sbj_verts = np.stack(sbj_verts, axis=0)
            else:
                sbj_model = smplx.build_layer(
                    model_path=str(cfg.SMPLX_PATH), model_type="smplh", gender=grab_seq_data["gender"],
                    num_betas=10, batch_size=T, use_pca=False
                )
                body_model_params = {
                    "betas": smplh_seq_data["body"]["betas"][smplh_mask],
                    "transl": smplh_seq_data["body"]["transl"][smplh_mask],
                    "global_orient": smplh_seq_data["body"]["global_orient"][smplh_mask].reshape(T, -1, 9),
                    "body_pose": smplh_seq_data["body"]["body_pose"][smplh_mask].reshape(T, -1, 9),
                    "left_hand_pose": smplh_seq_data["body"]["left_hand_pose"][smplh_mask].reshape(T, -1, 9),
                    "right_hand_pose": smplh_seq_data["body"]["right_hand_pose"][smplh_mask].reshape(T, -1, 9),
                }
                if cfg.input_type == "smpl":
                    body_model_params["left_hand_pose"] = None
                    body_model_params["right_hand_pose"] = None

                # get smpl(-h) vertices
                sbj_output = sbj_model(pose2rot=False, get_skin=True, return_full_pose=True, **body_model_params)
                sbj_verts = tensor_to_cpu(sbj_output.vertices)
                sbj_joints = tensor_to_cpu(sbj_output.joints)

                # align sbj meshes
                for i in range(T):
                    sbj_verts_center = sbj_verts[i].mean(axis=0, keepdims=True)
                    if cfg.preprocess_grab["align_with_joints"]:
                        # rotation to align torso with x axis direction
                        shoulderblades = sbj_joints[i][17] - sbj_joints[i][16]
                        z = np.array([0, 0, 1])
                        dir_shoulderblades = np.cross(z, shoulderblades)
                        dir_shoulderblades[2] = 0.0  # project to xy
                        dir_shoulderblades = dir_shoulderblades / np.linalg.norm(dir_shoulderblades)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            R_align, _ = Rotation.align_vectors(np.array([[1, 0, 0]]), dir_shoulderblades[None])

                        sbj_verts[i] = R_align.apply(sbj_verts[i])
                        sbj_joints[i] = R_align.apply(sbj_joints[i])
                        t_reset = -1 * np.copy(sbj_joints[i][0])
                        sbj_verts[i] += t_reset
                        sbj_joints[i] += t_reset
                    else:
                        # only align using center
                        R_align = Rotation.from_matrix(np.eye(3, dtype=np.float32))

                        t_reset = -1 * sbj_verts_center
                        sbj_verts[i] += t_reset

                    preprocess_transforms.append({
                        "R": R_align.as_matrix(),
                        "t": np.copy(t_reset)
                    })

                # create template mesh
                sbj_faces = sbj_model.faces
                sbj_mesh = trimesh.Trimesh(vertices=sbj_verts[0], faces=sbj_faces)
        elif cfg.input_type in ["hands"]:
            # keep only contact frames
            frame_mask = filter_contact_frames(cfg, grab_seq_data)

            # initial fps is 120, downsampled is 30 or 10
            if cfg.preprocess_grab["downsample"] != "None":
                downsample_mask = np.zeros_like(frame_mask)
                if cfg.preprocess_grab["downsample"] == "10fps":
                    downsample_mask[::12] = True
                else:
                    downsample_mask[::4] = True
                frame_mask = np.logical_and(frame_mask, downsample_mask)
            T = frame_mask.sum()

            # if no frame is selected continue to the next sequence
            if T < 1:
                continue

            # generate hand meshes
            # left hand
            lh_params = prepare_params(grab_seq_data["lhand"]["params"], frame_mask)
            lh_mesh = str(cfg.input_path / grab_seq_data["lhand"]["vtemp"])
            lh_mesh = trimesh_load(lh_mesh)
            lh_vtemp = np.array(lh_mesh.vertices)
            lh_m = smplx.create(
                model_path=str(cfg.model_path), model_type='mano', is_rhand=False, v_template=lh_vtemp,
                num_pca_comps=grab_seq_data['n_comps'], flat_hand_mean=True, batch_size=T
            )
            lh_parms = params_to_torch(lh_params)
            lh_output = lh_m(**lh_parms)
            verts_lh = tensor_to_cpu(lh_output.vertices)
            # right hand
            rh_params = prepare_params(grab_seq_data["rhand"]["params"], frame_mask)
            rh_mesh = str(cfg.input_path / grab_seq_data["rhand"]["vtemp"])
            rh_mesh = trimesh_load(rh_mesh)
            rh_vtemp = np.array(rh_mesh.vertices)
            rh_m = smplx.create(
                model_path=str(cfg.model_path), model_type='mano', is_rhand=True, v_template=rh_vtemp,
                num_pca_comps=grab_seq_data['n_comps'], flat_hand_mean=True, batch_size=T
            )
            rh_parms = params_to_torch(rh_params)
            rh_output = rh_m(**rh_parms)
            verts_rh = tensor_to_cpu(rh_output.vertices)

            # concatenate data
            sbj_verts = np.concatenate([verts_lh, verts_rh], axis=1)
            sbj_verts_t = np.concatenate([lh_mesh.vertices, rh_mesh.vertices], axis=0)
            sbj_faces_t = np.concatenate([lh_mesh.faces, rh_mesh.faces + len(lh_mesh.vertices)], axis=0)
            sbj_mesh = trimesh.Trimesh(vertices=sbj_verts_t, faces=sbj_faces_t)

            preprocess_transforms = []
            for t in range(T):
                # center hand meshes
                sbj_verts_center = sbj_verts[t].mean(axis=0, keepdims=True)
                t_reset = -1 * sbj_verts_center
                sbj_verts[t] += t_reset

                preprocess_transforms.append({
                    "R": np.eye(3),
                    "t": t_reset
                })
        else:
            raise ValueError(f"Unsupported input data type {cfg.input_type}")
        # ===========================================

        # ============ 2 extract vertices for object
        # parse obj parameters from annotations
        obj_params = prepare_params(grab_seq_data["object"]["params"], frame_mask)

        # pose all meshes in the sequence using ObjectModel
        obj_full_mesh = str(cfg.input_path / grab_seq_data["object"]["object_mesh"])
        obj_full_mesh = trimesh_load(obj_full_mesh)
        obj_full_vtemp = np.array(obj_full_mesh.vertices)
        obj_full_model = ObjectModel(v_template=obj_full_vtemp, batch_size=T)
        obj_full_parms = params_to_torch(obj_params)
        obj_full_verts = tensor_to_cpu(obj_full_model(**obj_full_parms).vertices)

        # transform using sbj transformation
        for index in range(T):
            obj_full_v = obj_full_verts[index]

            R = Rotation.from_matrix(preprocess_transforms[index]["R"])
            t = preprocess_transforms[index]["t"]
            obj_full_verts[index] = R.apply(obj_full_v) + t

        # optionally load decimated mesh
        if cfg.preprocess_grab["use_decimated_obj_meshes"]:
            obj_mesh = cfg.input_path / grab_seq_data["object"]["object_mesh"]
            obj_mesh = obj_mesh.parents[1] / "decimated_meshes" / obj_mesh.name
            obj_mesh = trimesh_load(obj_mesh)
            obj_dec_vtemp = np.array(obj_mesh.vertices)
            obj_dec_verts = np.tile(np.copy(obj_dec_vtemp), (T, 1, 1))
        else:
            obj_mesh = obj_full_mesh
        # calculate obj rotations and optionally transform decimated mesh for each t_stamp
        obj_rotations = []
        for t_stamp in range(0, T):
            # find transform from vertices in the canonical pose to vertices in t_stamp frame
            R_t_stamp, t_t_stamp = estimate_transform(obj_full_vtemp, obj_full_verts[t_stamp])
            obj_rotations.append(R_t_stamp)

            # THIS STEP IS OPTIONAL, USED TO SAVE SPACE
            if cfg.preprocess_grab["use_decimated_obj_meshes"]:
                # apply transforms to decimated mesh to obtain posed decimated mesh for all frames
                obj_dec_verts[t_stamp] = np.dot(obj_dec_verts[t_stamp], R_t_stamp.T) + t_t_stamp[None]
        if cfg.preprocess_grab["use_decimated_obj_meshes"]:
            obj_verts = obj_dec_verts
        else:
            obj_verts = obj_full_verts
        # ===========================================

        # ============ 3 align the ground plane ============
        if cfg.input_type in ["smplh", "smpl"]:
            for i in range(T):
                z_min = min(np.min(sbj_verts[i, :, 2]), np.min(obj_verts[i, :, 2]))
                t_align_z = np.array([0.0, 0.0, -z_min], dtype=np.float32)

                preprocess_transforms[i]["t"] += t_align_z

                sbj_verts[i] += t_align_z
                obj_verts[i] += t_align_z
        # ==================================================

        # ============ 4 preprocess each time stamp in parallel
        sbj_pointcloud = np.copy(sbj_verts)
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
            "num_points_pc_subject": cfg.num_points_pc_subject
        } for t in range(T)]

        # Same actions for each t_stamp
        preprocess_results = parallel_map(
            input_data, preprocess_worker, use_kwargs=True, n_jobs=10, tqdm_kwargs={"leave": False}
        )
        # ===========================================

        # ============ 5 Save subject-specific data
        for sample in preprocess_results:
            sample.dump(cfg.output_path)
        # ===========================================

    # ============ 6 Save global info
    with (cfg.output_path / "metadata.json").open("w") as fp:
        cfg_dict = vars(cfg)
        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        json.dump(cfg_dict, fp, indent=4)
    # ===========================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data')

    parser.add_argument('-i', "--input-path", type=Path)
    parser.add_argument('-o', "--output-path", type=Path)
    parser.add_argument('-c', "--config", type=Path, default="./project_config.toml")
    parser.add_argument('-d', "--downsample", type=str, choices=["None", "30fps", "10fps"],
                        default=None, help="Perform downsampling of sequences.")
    parser.add_argument('-s', "--subjects", nargs="+", type=str, default=None,
                        help="Preprocess only selected subjects")
    parser.add_argument('-g', "--generate-keypoints", action="store_true", help="Generate keypoints for objects.")

    args = parser.parse_args()

    project_config = dict(tomlkit.parse(args.config.read_text()))

    # preset directories
    if project_config["preprocess_grab"]["use_decimated_obj_meshes"]:
        project_config["grab_objects_path"] = args.input_path / "tools/object_meshes/decimated_meshes"
    else:
        project_config["grab_objects_path"] = args.input_path / "tools/object_meshes/contact_meshes"
    # additional parameters
    project_config["input_path"] = args.input_path
    project_config["output_path"] = args.output_path
    if args.downsample is not None:
        project_config["preprocess_grab"]["downsample"] = args.downsample
    if args.subjects is not None:
        project_config["preprocess_grab"]["subjects"] = args.subjects

    config = SimpleNamespace(**project_config)
    preprocess(config)

    if not isinstance(config.grab_objects_path, Path):
        print("NOT PATH")
        config.grab_objects_path = Path(config.grab_objects_path)

    if args.generate_keypoints:
        generate_obj_keypoints(
            config.preprocess_grab["objects"], Path(config.grab_objects_path),
            config.obj_keypoints_npoints, Path(config.output_path)
        )
