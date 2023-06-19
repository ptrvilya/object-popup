"""
Common utilities for data preprocessing.

Functions:
    params_to_torch, tensor_to_cpu, prepare_params, parse_npz
are from https://github.com/otaheri/GRAB/blob/master/tools/utils.py

Function:
    filter_contact_frames
is from https://github.com/otaheri/GRAB/blob/master/grab/grab_preprocessing.py

License for functions mentioned above:
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
"""
import json
import pickle as pkl
from dataclasses import dataclass, field

import numpy as np
import torch
import trimesh


def trimesh_load(path):
    return trimesh.load(str(path), process=False, validate=False, inspect=False)


def params_to_torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def tensor_to_cpu(tensor):
    return tensor.detach().cpu().numpy()


def filter_contact_frames(cfg, seq_data):
    if cfg.preprocess_grab["only_contact_frames"]:
        frame_mask = (seq_data['contact']['object'] > 0).any(axis=1)
    else:
        frame_mask = (seq_data['contact']['object'] > -1).any(axis=1)
    return frame_mask


def prepare_params(params, frame_mask, dtype = np.float32):
    return {k: v[frame_mask].astype(dtype) for k, v in params.items()}


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return npz


def estimate_transform(vertices_from, vertices_to):
    """
    Based on compute_similarity_transform from https://github.com/akanazawa/hmr/blob/master/src/benchmark/eval_util.py
    """
    R, t = None, None

    centoid_from = vertices_from.mean(0, keepdims=True)
    centoid_to = vertices_to.mean(0, keepdims=True)

    vertices_from_shifted = vertices_from - centoid_from
    vertices_to_shifted = vertices_to - centoid_to

    vertices_from_shifted = vertices_from_shifted.swapaxes(0, 1)
    vertices_to_shifted = vertices_to_shifted.swapaxes(0, 1)

    H = vertices_from_shifted @ vertices_to_shifted.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = (centoid_to.reshape(3, 1) - R @ centoid_from.reshape(3, 1)).reshape(1, 3)

    return R, t


def get_sequences_list(dataset, input_path, subjects=None, objects=None):
    all_seqs = []
    subjects = ["*"] if subjects is None else subjects
    if dataset == "grab":
        if objects is not None and len(objects) > 0:
            for sbj in subjects:
                for obj in objects:
                    all_seqs.extend(list(input_path.glob(f'grab/{sbj}/{obj}_*.npz')))
        else:
            for sbj in subjects:
                all_seqs.extend(list(input_path.glob(f'grab/{sbj}/*.npz')))
    elif dataset == "behave":
        # sequences / Date<:02d>_Sub<:02d>_<object>_<optional:action> / t<:04d>.000 /
        if objects is not None and len(objects) > 0:
            for sbj in subjects:
                for object in objects:
                    all_seqs.extend(list(input_path.glob(f'*_{sbj}_{object}*/')))
        else:
            for sbj in subjects:
                all_seqs.extend(list(input_path.glob(f'*_{sbj}_*/')))
    elif dataset == "custom":
        if objects is not None and len(objects) > 0:
            for object in objects:
                all_seqs.extend(list(input_path.glob(f'imus_smpl/*/{object}_*/')))
        else:
            all_seqs = list(input_path.glob('imus_smpl/*/*/'))

    return all_seqs


@dataclass
class DatasetSample:
    # General info
    subject: str
    object: str
    action: str
    t_stamp: int
    # Meshes and PCs
    sbj_mesh: trimesh.Trimesh
    obj_mesh: trimesh.Trimesh
    sbj_pc: np.ndarray
    obj_verts: np.ndarray
    # Object features
    obj_center: np.ndarray = field(init=False)
    obj_rotation: np.ndarray
    # Preprocessing params
    scale: float = field(init=False)
    translation: np.ndarray = field(init=False)
    preprocess_transforms: dict

    def dump(self, data_path):
        output_path = data_path / self.subject / f"{self.object}_{self.action}" / f"t{self.t_stamp:05d}"
        output_path.mkdir(exist_ok=True, parents=True)

        # meshes
        _ = self.sbj_mesh.export(output_path / "subject.ply")
        if self.obj_mesh is not None:
            _ = self.obj_mesh.export(output_path / "object.ply")

        # sbj point cloud
        np.savez(output_path / "subject_pointcloud.npz", subject_pointcloud=self.sbj_pc)

        # object features
        if self.obj_center is not None:
            np.savez(
                output_path / "object_features.npz", center=self.obj_center,
                rotation=self.obj_rotation, object_name=self.object,
            )

        # preprocessing data
        with(output_path / "preprocess_transform.pkl").open("wb") as fp:
            preprocess_transforms = self.preprocess_transforms

            preprocess_transforms.update({
                "scale": self.scale,
                "translation": self.translation,
            })
            pkl.dump(preprocess_transforms, fp)


def preprocess_worker(
    sample: DatasetSample,
    num_points_pc_subject,
):
    # ============ 1 set vertices for sbj and obj meshes
    obj_vertices = np.array(np.copy(sample.obj_verts))
    sample.sbj_mesh.vertices = np.copy(sample.sbj_pc)
    sample.obj_mesh.vertices = obj_vertices

    # ============ 2 placeholders for scale and translation
    sample.scale = 1.0
    sample.translation = np.zeros(3, dtype=np.float32)

    # ============ 3 sample point clouds over the meshes
    sample.sbj_pc = sample.sbj_mesh.sample(num_points_pc_subject)

    # ============ 4 object features
    sample.obj_center = obj_vertices.mean(axis=0)

    return sample


def preprocess_worker_rawpc(
    sample: DatasetSample,
    num_points_pc_subject,
    raw_pc
):
    # ============ 1 set vertices for sbj and obj meshes
    obj_vertices = np.array(np.copy(sample.obj_verts))
    sample.sbj_mesh.vertices = np.copy(sample.sbj_pc)
    sample.obj_mesh.vertices = obj_vertices

    # ============ 2 placeholders for scale and translation
    sample.scale = 1.0
    sample.translation = np.zeros(3, dtype=np.float32)

    # ============ 3 subsample points from the raw pc
    sbj_pc_index = np.random.choice(len(raw_pc), size=min(15 * num_points_pc_subject, len(raw_pc)), replace=False)
    sample.sbj_pc = raw_pc[sbj_pc_index]

    # ============ 4 object features
    sample.obj_center = obj_vertices.mean(axis=0)

    return sample


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = \
        norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    """
    From https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/rodrigues_layer.py
    """
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def th_posemap_axisang(pose_vectors):
    """
    Converts axis-angle to rotmat
    pose_vectors (Tensor (batch_size x 72)): pose parameters in axis-angle representation
    From: https://github.com/gulvarol/smplpytorch/blob/master/smplpytorch/pytorch/tensutils.py
    """
    rot_nb = int(pose_vectors.shape[1] / 3)
    rot_mats = []
    for joint_idx in range(rot_nb):
        axis_ang = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rot_mat = batch_rodrigues(axis_ang)
        rot_mats.append(rot_mat)

    rot_mats = torch.cat(rot_mats, 1)
    return rot_mats


def generate_obj_keypoints_from_barycentric(objects, dataset, dataset_path):
    # list of triangles indices - sampled_points_triangles_ids
    # barycentric coords - sampled_points_bary
    with open(f"./assets/{dataset}_objects_keypoints.pkl", "rb") as fp:
        barycentric_dict = pkl.load(fp)

    if len(objects) == 0:
        objects = sorted(list((dataset_path / "object_meshes").glob("*")))
        objects = [obj.stem for obj in objects]

    for obj in objects:
        object_mesh = trimesh.load(
            str(dataset_path / "object_meshes" / f"{obj}.ply"), process=False, validate=False
        )

        # load triangle_ids and barycentric coordinates
        triangles_ids = barycentric_dict[obj]["triangles_ids"]
        barycentric = barycentric_dict[obj]["barycentric"]
        triangles = object_mesh.faces[triangles_ids]
        triangles_coords = object_mesh.vertices[triangles]

        sampled_points = \
            trimesh.triangles.barycentric_to_points(triangles_coords, barycentric)

        output_folder = dataset_path / "object_keypoints"
        output_folder.mkdir(exist_ok=True, parents=True)
        np.savez(output_folder / f"{obj}.npz", cartesian=sampled_points, barycentric=barycentric,
                 triangles_ids=triangles_ids)


def generate_obj_keypoints_barycentric(objects, keypoints_path, output_path):
    if len(objects) == 0:
        objects = sorted(list(keypoints_path.glob("*")))
        objects = [obj.stem for obj in objects]

    # save only barycentric coordinates and triangles ids
    barycentric_dict = dict()
    for obj in objects:
        obj_keypoints = np.load(keypoints_path / f"{obj}.npz")
        triangles_ids = obj_keypoints["triangles_ids"]
        barycentric = obj_keypoints["barycentric"]

        barycentric_dict[obj] = {
            "triangles_ids": triangles_ids,
            "barycentric": barycentric
        }

    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "behave_objects_keypoints.pkl", "wb") as fp:
        pkl.dump(barycentric_dict, fp)


def generate_obj_keypoints(objects, objects_path, n_keypoints, output_path):
    if len(objects) == 0:
        objects = sorted(list(objects_path.glob("*")))
        objects = [obj.stem for obj in objects]

    for obj in objects:
        object_mesh = trimesh.load(str(objects_path / f"{obj}.ply"), process=False, validate=False)
        output_folder = output_path / "object_meshes"
        output_folder.mkdir(exist_ok=True, parents=True)
        _ = object_mesh.export(str(output_folder / f"{obj}.ply"))

        sampled_points = object_mesh.sample(n_keypoints)
        _, _, sampled_points_triangles_ids = trimesh.proximity.closest_point(object_mesh, sampled_points)
        sampled_points_triangles = object_mesh.faces[sampled_points_triangles_ids]

        sampled_points_bary = trimesh.triangles.points_to_barycentric(
            object_mesh.vertices[sampled_points_triangles],
            sampled_points
        )

        output_folder = output_path / "object_keypoints"
        output_folder.mkdir(exist_ok=True, parents=True)
        np.savez(output_folder / f"{obj}.npz", cartesian=sampled_points, barycentric=sampled_points_bary,
                triangles_ids=sampled_points_triangles_ids)


def generate_behave_split(behave_path, target_path):
    with (behave_path / "split.json").open("r") as fp:
        official_split = json.load(fp)

    # new split format is [[sbj_<split>, obj_act_date], ...]
    for split in ["train", "test"]:
        new_split = []
        sequences = official_split[split]
        for sequence in sequences:
            sequence_split = sequence.split("_")
            date = sequence_split[0]
            subject = sequence_split[1]
            object = sequence_split[2]
            action = "" if len(sequence_split) == 3 else sequence_split[3]

            new_subject = f"{subject}_{split}"
            new_obj_act = f"{object}_{action}_{date}" if len(action) > 0 else f"{object}_{date}"
            new_split.append([new_subject, new_obj_act])

        with (target_path / f"behave_split_{split}").open("w") as fp:
            json.dump(new_split, fp, indent=2)


def generate_behave_canonicalized_objects(behave_orig_objects_path, behave_can_objects_path):
    with open("./assets/behave_objects_canonicalization.pkl", "rb") as fp:
        transforms = pkl.load(fp)

    behave_can_objects_path.mkdir(exist_ok=True, parents=True)
    for obj, (R, t) in transforms.items():
        if obj in ["chairblack", "chairwood"]:
            suffix = "_f2500"
        elif obj == "tablesquare":
            suffix = "_f2000"
        elif obj == "monitor":
            suffix = "_closed_f1000"
        else:
            suffix = "_f1000"
        src_m = trimesh_load(str(behave_orig_objects_path / f"{obj}/{obj}{suffix}.ply"))
        tgt_v = np.array(src_m.vertices)
        tgt_v = tgt_v + t
        tgt_v = np.dot(R, tgt_v.T).T
        src_m.vertices = tgt_v
        _ = src_m.export(str(behave_can_objects_path / f"{obj}.ply"))
