"""
Implementation of nearest neighbours baseline for Object pop-up with and without class prediction.
"""
import json
from collections import defaultdict

import faiss
import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from tqdm.autonotebook import tqdm

from ..data.dataset import get_sbj_to_obj_act


def pseudo_inverse(mat):
    assert len(mat.shape) == 3
    tr = torch.bmm(mat.transpose(2, 1), mat)
    tr_inv = torch.inverse(tr)
    inv = torch.bmm(tr_inv, mat.transpose(2, 1))
    return inv


def init_object_orientation(src_axis, tgt_axis):
    pseudo = pseudo_inverse(src_axis)
    rot = torch.bmm(pseudo, tgt_axis)

    U, S, V = torch.svd(rot)
    R = torch.bmm(U, V.transpose(2, 1))

    return R


class KnnFaiss:
    def __init__(self, features, device="cpu"):
        if device == "gpu":
            resources = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0

            self.index = faiss.GpuIndexFlatL2(resources, features.shape[1], flat_config)
        else:
            self.index = faiss.IndexFlatL2(features.shape[1])
        self.index.add(features.astype(np.float32))

    def query(self, features, k=1):
        distances, indices = self.index.search(features.astype(np.float32), k=k)

        return distances, indices


def get_data_for_sequence(features, model_type, data_path, sequence_folder, objname2classid):
    sequence_path = data_path / sequence_folder
    obj_class_name = sequence_path.name.split("_")[0]
    class_id = objname2classid[obj_class_name]

    t_stamps = sorted(list((sequence_path.glob("t*"))))
    T = len(t_stamps)
    labels = []
    if features == "pose":
        human_features = np.zeros((T, 51 * 3), dtype=np.float32)
    else:
        human_features = np.zeros((T, 6890 * 3), dtype=np.float32)

    if T == 0:
        return [], -1, []

    for index, t_stamp in enumerate(t_stamps):
        if features == "pose":
            pose_mat = np.load(t_stamp / "subject_pose.npz")["sbj_pose"][1:]
            pose_rotvec = np.zeros((51, 3), dtype=np.float32)
            for i_mat in range(pose_mat.shape[0]):
                mat = Rotation.from_matrix(pose_mat[i_mat])
                pose_rotvec[i_mat] = mat.as_rotvec()
            human_features[index] = pose_rotvec.flatten()
        else:
            human_features[index] = np.array(trimesh.load(str(t_stamp / "subject.ply"), process=False).vertices).flatten()

        if model_type == "classifier":
            labels.append(class_id)
        elif model_type == "pose_general":
            # get object pose vectors
            # object pose is represented as Tx12 : rot matrix and center location
            object_features = np.load(t_stamp / "object_features.npz")
            obj_rot = object_features["rotation"]
            center = object_features["center"]
            class_label = np.array([class_id])
            object_poses = np.concatenate([class_label, obj_rot.reshape(9), center])

            labels.append(object_poses)
        elif model_type == "pose_class_specific":
            # get object pose vectors
            # object pose is represented as Tx12 : rot matrix and center location
            object_features = np.load(t_stamp / "object_features.npz")
            obj_rot = object_features["rotation"]
            center = object_features["center"]
            object_poses = np.concatenate([obj_rot.reshape(9), center])

            labels.append(object_poses)
    labels = np.stack(labels, axis=0)

    return human_features, class_id, labels


def create_nn_model(cfg, model_type: str, human_features="verts", backend="scipy"):
    assert human_features in ["verts", "pose"]
    assert model_type in ["pose_class_specific", "pose_general", "classifier"]
    # Three options:
    #   pose_class_specific - class, vertices: object_pose
    #   pose_general        - vertices       : object_class, object pose
    #   classifier          - vertices       : object_class

    # Default class mapping
    objname2classid = cfg.objname2classid

    # ===> 1. Get folders with data
    train_folders, val_folders = dict(), dict()
    data_paths = {}
    for dataset in cfg.datasets:
        path_template = "{subject}/{object}_{action}"
        if dataset == "grab":
            data_paths["grab"] = cfg.grab_path

            # get train/test sequences
            train_subjects, train_objects = cfg.grab["train_subjects"], cfg.grab["train_objects"]
            val_subjects, val_objects = cfg.grab["val_subjects"], cfg.grab["val_objects"]

            # get (sbj, obj_act) tuples for training
            train_objact = get_sbj_to_obj_act(
                path_template, data_paths["grab"], train_subjects, train_objects, actions=None
            )
            # form folder path for each pair
            grab_train_folders = []
            for subject, obj_acts in train_objact.items():
                for (obj, act) in obj_acts:
                    grab_train_folders.append(path_template.format(subject=subject, object=obj, action=act))

            # get (sbj, obj_act) tuples for testing
            val_objact = get_sbj_to_obj_act(
                path_template, data_paths["grab"], val_subjects, val_objects, actions=None
            )
            # form folder path for each pair
            grab_val_folders = []
            for subject, obj_acts in val_objact.items():
                for (obj, act) in obj_acts:
                    grab_val_folders.append(path_template.format(subject=subject, object=obj, action=act))
            # save folder paths
            train_folders["grab"] = grab_train_folders
            val_folders["grab"] = grab_val_folders
        elif dataset == "behave":
            data_paths["behave"] = cfg.behave_path

            # get train/test sequences
            train_split, train_objects = cfg.behave["train_split_file"], cfg.behave["train_objects"]
            val_split, val_objects = cfg.behave["val_split_file"], cfg.behave["val_objects"]

            # get (sbj, obj_act) tuples for training
            with open(train_split, "r") as fp:
                split = json.load(fp)
            # form folder path for each pair
            behave_train_folders = []
            for subject, obj_act in split:
                obj_act = obj_act.split("_")
                obj = obj_act[0]
                act = "_".join(obj_act[1:])
                if obj in train_objects:
                    behave_train_folders.append(path_template.format(subject=subject, object=obj, action=act))

            # get (sbj, obj_act) tuples for testing
            with open(val_split, "r") as fp:
                split = json.load(fp)
            # form folder path for each pair
            behave_val_folders = []
            for subject, obj_act in split:
                obj_act = obj_act.split("_")
                obj = obj_act[0]
                act = "_".join(obj_act[1:])
                if obj in val_objects:
                    behave_val_folders.append(path_template.format(subject=subject, object=obj, action=act))

            # save folder paths
            train_folders["behave"] = behave_train_folders
            val_folders["behave"] = behave_val_folders
    # <===

    # ===> 2. Load data and create models
    if model_type in ["classifier", "pose_general"]:
        # load training data
        train_features, train_labels = [], []
        for dataset, folders in train_folders.items():
            for folder in tqdm(folders, ncols=80):
                features, _, labels = get_data_for_sequence(
                    human_features, model_type, data_paths[dataset], folder, objname2classid
                )

                if len(features) == 0:
                    continue

                train_features.append(features)
                train_labels.append(labels)
        train_features = np.concatenate(train_features, axis=0).astype(np.float32)
        train_labels = np.concatenate(train_labels, axis=0)

        # build kdtree on training data
        if backend == "scipy":
            kdtree = KDTree(train_features, copy_data=True)
        elif backend == "faiss_cpu":
            kdtree = KnnFaiss(train_features, device="cpu")
        elif backend == "faiss_gpu":
            kdtree = KnnFaiss(train_features, device="gpu")

        # load test data
        test_queries, test_labels, test_t_stamps = {}, {}, {}
        for dataset, folders in val_folders.items():
            queries, gt_labels, gt_t_stamps = [], [], []
            for folder in folders:
                features, _, labels = get_data_for_sequence(
                    human_features, model_type, data_paths[dataset], folder, objname2classid
                )

                if len(features) == 0:
                    continue

                queries.append(features)
                gt_labels.append(labels)
                gt_t_stamps.extend([f"{folder}/t{t_stamp:05d}" for t_stamp in range(len(labels))])

            test_queries[dataset] = np.concatenate(queries, axis=0)
            test_labels[dataset] = np.concatenate(gt_labels, axis=0)
            test_t_stamps[dataset] = gt_t_stamps
    elif model_type == "pose_class_specific":
        # load training data
        train_features, _train_labels = defaultdict(list), defaultdict(list)
        for dataset, folders in train_folders.items():
            for folder in tqdm(folders, ncols=80):
                features, class_id, labels = get_data_for_sequence(
                    human_features, model_type, data_paths[dataset], folder, objname2classid
                )

                if len(features) == 0:
                    continue

                train_features[class_id].append(features)
                _train_labels[class_id].append(labels)

        # build per-class kdtrees
        kdtree, train_labels = dict(), dict()
        for class_id in train_features.keys():
            features = np.concatenate(train_features[class_id], axis=0).astype(np.float32)
            train_labels[class_id] = np.concatenate(_train_labels[class_id], axis=0)

            if backend == "scipy":
                kdtree[class_id] = KDTree(features, copy_data=True)
            elif backend == "faiss_cpu":
                kdtree[class_id] = KnnFaiss(features, device="cpu")
            elif backend == "faiss_gpu":
                kdtree[class_id] = KnnFaiss(features, device="gpu")

        # load test data
        _test_queries, _test_labels, test_t_stamps = dict(), dict(), dict()
        for dataset, folders in val_folders.items():
            _test_queries[dataset], _test_labels[dataset], test_t_stamps[dataset] = \
                defaultdict(list), defaultdict(list), defaultdict(list)

            for folder in folders:
                features, class_id, labels = get_data_for_sequence(
                    human_features, model_type, data_paths[dataset], folder, objname2classid
                )

                if len(features) == 0:
                    continue

                _test_queries[dataset][class_id].append(features)
                _test_labels[dataset][class_id].append(labels)
                test_t_stamps[dataset][class_id].extend(
                    [f"{folder}/t{t_stamp:05d}" for t_stamp in range(len(labels))]
                )

        test_queries, test_labels = dict(), dict()
        for dataset in val_folders.keys():
            for class_id in _test_queries[dataset].keys():
                test_queries[dataset][class_id] = np.concatenate(_test_queries[dataset][class_id], axis=0)
                test_labels[dataset][class_id] = np.concatenate(_test_labels[dataset][class_id], axis=0)
    # <===

    return kdtree, train_labels, test_queries, test_labels, test_t_stamps


def create_and_query_nn_model(cfg, model_type: str, n_neighbors=1, human_features="verts", backend="scipy"):
    # function that creates and quires the model simultaneously to optimize compute and storage
    assert model_type in ["pose_class_specific"]  # pose_class_specific - (class, vertices) -> object_pose

    # Default class mapping
    objname2classid = cfg.objname2classid

    # ===> 1. Get folders with data
    train_folders, val_folders = dict(), dict()
    data_paths = {}
    for dataset in cfg.datasets:
        path_template = "{subject}/{object}_{action}"
        if dataset == "grab":
            data_paths["grab"] = cfg.grab_path

            # get train/test sequences
            train_subjects, train_objects = cfg.grab["train_subjects"], cfg.grab["train_objects"]
            val_subjects, val_objects = cfg.grab["val_subjects"], cfg.grab["val_objects"]

            # get (sbj, obj_act) tuples for training
            train_objact = get_sbj_to_obj_act(
                path_template, data_paths["grab"], train_subjects, train_objects, actions=None
            )
            # form folder path for each pair
            grab_train_folders = []
            for subject, obj_acts in train_objact.items():
                for (obj, act) in obj_acts:
                    grab_train_folders.append(path_template.format(subject=subject, object=obj, action=act))

            # get (sbj, obj_act) tuples for testing
            val_objact = get_sbj_to_obj_act(
                path_template, data_paths["grab"], val_subjects, val_objects, actions=None
            )
            # form folder path for each pair
            grab_val_folders = []
            for subject, obj_acts in val_objact.items():
                for (obj, act) in obj_acts:
                    grab_val_folders.append(path_template.format(subject=subject, object=obj, action=act))
            # save folder paths
            train_folders["grab"] = grab_train_folders
            val_folders["grab"] = grab_val_folders
        elif dataset == "behave":
            data_paths["behave"] = cfg.behave_path

            # get train/test sequences
            train_split, train_objects = cfg.behave["train_split_file"], cfg.behave["train_objects"]
            val_split, val_objects = cfg.behave["val_split_file"], cfg.behave["val_objects"]

            # get (sbj, obj_act) tuples for training
            with open(train_split, "r") as fp:
                split = json.load(fp)
            # form folder path for each pair
            behave_train_folders = []
            for subject, obj_act in split:
                obj_act = obj_act.split("_")
                obj = obj_act[0]
                act = "_".join(obj_act[1:])
                if obj in train_objects:
                    behave_train_folders.append(path_template.format(subject=subject, object=obj, action=act))

            # get (sbj, obj_act) tuples for testing
            with open(val_split, "r") as fp:
                split = json.load(fp)
            # form folder path for each pair
            behave_val_folders = []
            for subject, obj_act in split:
                obj_act = obj_act.split("_")
                obj = obj_act[0]
                act = "_".join(obj_act[1:])
                if obj in val_objects:
                    behave_val_folders.append(path_template.format(subject=subject, object=obj, action=act))

            # save folder paths
            train_folders["behave"] = behave_train_folders
            val_folders["behave"] = behave_val_folders
    # <===

    # ===> 2. Load data
    # load training data
    train_features, _train_labels = defaultdict(list), defaultdict(list)
    for dataset, folders in train_folders.items():
        for folder in tqdm(folders, ncols=80):
            features, class_id, labels = get_data_for_sequence(
                human_features, model_type, data_paths[dataset], folder, objname2classid
            )

            if len(features) == 0:
                continue

            train_features[class_id].append(features)
            _train_labels[class_id].append(labels)

    # load test data
    _test_queries, _test_labels, test_t_stamps = dict(), dict(), dict()
    for dataset, folders in val_folders.items():
        _test_queries[dataset], _test_labels[dataset], test_t_stamps[dataset] = \
            defaultdict(list), defaultdict(list), defaultdict(list)

        for folder in folders:
            features, class_id, labels = get_data_for_sequence(
                human_features, model_type, data_paths[dataset], folder, objname2classid
            )

            if len(features) == 0:
                continue

            _test_queries[dataset][class_id].append(features)
            _test_labels[dataset][class_id].append(labels)
            test_t_stamps[dataset][class_id].extend(
                [f"{folder}/t{t_stamp:05d}" for t_stamp in range(len(labels))]
            )

    test_queries, test_labels = dict(), dict()
    for dataset in val_folders.keys():
        test_queries[dataset], test_labels[dataset] = dict(), dict()
        for class_id in _test_queries[dataset].keys():
            test_queries[dataset][class_id] = np.concatenate(_test_queries[dataset][class_id], axis=0)
            test_labels[dataset][class_id] = np.concatenate(_test_labels[dataset][class_id], axis=0)
    # <===

    # ===> 3. Create and query models
    # build and query per-class kdtrees
    pred_neighbors, train_labels = dict(), dict()
    for class_id in train_features.keys():
        pred_neighbors[class_id] = dict()
        features = np.concatenate(train_features[class_id], axis=0).astype(np.float32)
        train_labels[class_id] = np.concatenate(_train_labels[class_id], axis=0)

        if backend == "scipy":
            kdtree = KDTree(features, copy_data=True)
        elif backend == "faiss_cpu":
            kdtree = KnnFaiss(features, device="cpu")
        elif backend == "faiss_gpu":
            kdtree = KnnFaiss(features, device="gpu")

        for dataset in test_queries.keys():
            if class_id in test_queries[dataset]:
                test_query = test_queries[dataset][class_id]
                _, pred_neighbors[class_id][dataset] = kdtree.query(test_query, k=n_neighbors)
    # <===

    return pred_neighbors, train_labels, test_labels, test_t_stamps
