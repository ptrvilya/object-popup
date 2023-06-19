import json
import pickle as pkl
from collections import defaultdict
from typing import NamedTuple, List

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from ..utils.preprocess import estimate_transform


class DataSample(NamedTuple):
    subject: str
    object: str
    action: str
    t_stamp: str


def get_sbj_to_obj_act(path_template, grab_path, subjects, objects, actions):
    subject_to_obj_act = defaultdict(list)
    for subject in subjects:
        obj_acts = list(grab_path.glob(path_template.format(subject=subject, object="*", action="*")))
        for obj_act in obj_acts:
            obj_act = str(obj_act.name).split("_")
            obj = obj_act[0]
            act = "_".join(obj_act[1:])

            valid_action = True
            if actions is not None and len(actions) > 0:
                for exclude_action in actions:
                    if act in exclude_action:
                        valid_action = False
                        break

            if (objects is None or (len(objects) > 0 and obj in objects)) and valid_action:
                subject_to_obj_act[subject].append((obj, act))

    return subject_to_obj_act


class ObjectPopupDataset(Dataset):
    def __init__(
        self, cfg, data_path, subjects: List=None, objects: List=None, actions: List=None,
        split_file: str=None, eval_mode=False, downsample_factor=1
    ):
        super(ObjectPopupDataset, self).__init__()
        self.data_path = data_path
        self.objects = objects
        self.actions = actions
        self.eval_mode = eval_mode

        self.subjects = [] if subjects is None else subjects

        path_template = "{subject}/{object}_{action}/"
        if split_file is not None:
            with open(split_file, "r") as fp:
                split = json.load(fp)

            self.data = []
            for subject, obj_act in split:
                _oa_split = obj_act.split("_")
                obj = _oa_split[0]
                act = "_".join(_oa_split[1:])
                t_stamps = (self.data_path / path_template.format(subject=subject, object=obj, action=act)).glob("t*")
                self.data.extend([DataSample(subject, obj, act, t_stamp.name) for t_stamp in t_stamps])
        else:
            subject_to_obj_act = \
                get_sbj_to_obj_act(path_template, self.data_path, self.subjects, self.objects, self.actions)

            self.data = []
            # Triplets of subject, action, t_stamp
            for subject, obj_acts in subject_to_obj_act.items():
                for (obj, act) in obj_acts:
                    t_stamps = (self.data_path / path_template.format(subject=subject, object=obj, action=act)).glob("t*")
                    t_stamps = sorted(t_stamps)
                    self.data.extend([DataSample(subject, obj, act, t_stamp.name) for t_stamp in t_stamps])
        self.path_template = path_template + "{t_stamp}"

        if downsample_factor > 1:
            self.data = self.data[::downsample_factor]

        self.cfg = cfg
        self.classid2objname = {v: k for k, v in self.cfg.objname2classid.items()}
        dataset_objects = list((self.data_path / "object_keypoints").glob("*.npz"))
        self.dataset_objects = [object_name.stem for object_name in dataset_objects]

        self.canonical_obj_meshes = dict()
        self.canonical_obj_keypoints = dict()

        for class_id, object_name in self.classid2objname.items():
            if object_name in self.dataset_objects:
                self.canonical_obj_keypoints[class_id] = dict(np.load(
                    self.data_path / "object_keypoints" / f"{object_name}.npz"
                ))
                self.canonical_obj_meshes[class_id] = trimesh.load(
                    str(self.data_path / "object_meshes" / f"{object_name}.ply"),
                process=False)

    def distort_keypoints(self, data_sample, obj_keypoints_can, obj_keypoints_gt):
        if self.cfg.obj_keypoints_init == "local_jitter":
            obj_to_gt_r, obj_to_gt_t = estimate_transform(
                obj_keypoints_can, obj_keypoints_gt
            )

            prob = np.random.uniform()
            if prob <= 0.6:
                # canonical rotation, jittered translation
                obj_r = np.eye(3, dtype=np.float32)
                obj_t_random_jitter = (np.random.randn(3) - 0.5) / 20  # in -0.05 .. 0.05
                obj_t_random_jitter = obj_t_random_jitter.reshape(1, 3)
                obj_t = obj_to_gt_t + obj_t_random_jitter
            elif 0.6 < prob <= 0.7:
                # random rotation, jittered translataion
                obj_r = Rotation.from_euler("xyz", 0.2 * np.pi * (np.random.randn(3) - 0.5), degrees=False)
                obj_r = obj_r.as_matrix()

                obj_t_random_jitter = (np.random.randn(3) - 0.5) / 20  # in -0.05 .. 0.05
                obj_t_random_jitter = obj_t_random_jitter.reshape(1, 3)
                obj_t = obj_to_gt_t + obj_t_random_jitter
            else:
                # random rotation, gt translation
                obj_r = Rotation.from_euler("xyz", 0.2 * np.pi * (np.random.randn(3) - 0.5), degrees=False)
                obj_r = obj_r.as_matrix()

                obj_t = obj_to_gt_t

            obj_c = obj_keypoints_can.mean(axis=0, keepdims=True)
            obj_keypoints_in = np.dot(obj_keypoints_can - obj_c, obj_r.T) + obj_c + obj_t
            obj_to_gt_r, obj_to_gt_t = estimate_transform(
                obj_keypoints_in, obj_keypoints_gt
            )
            obj_r = obj_to_gt_r
            obj_t = obj_to_gt_t
            obj_c = np.zeros(3)
        else:
            raise ValueError(f"Unknown obj_keypoints_init {self.cfg.obj_keypoints_init}")

        data_sample["obj_keypoints_in"] = obj_keypoints_in.astype(np.float32)
        data_sample["obj_keypoints_offsets"] = (obj_keypoints_gt - obj_keypoints_in).astype(np.float32)
        data_sample["obj_keypoints_locations"] = (obj_keypoints_gt).astype(np.float32)
        data_sample["obj_R"] = obj_r.astype(np.float32).flatten()
        data_sample["obj_t"] = obj_t.astype(np.float32)
        data_sample["obj_c"] = obj_c.astype(np.float32)

        return data_sample

    def get_data_sample(self, index):
        sample = self.data[index]
        path = self.data_path / self.path_template.format(
            subject=sample.subject, object=sample.object, action=sample.action, t_stamp=sample.t_stamp
        )

        with (path / "preprocess_transform.pkl").open("rb") as fp:
            preprocess_transform = pkl.load(fp)
            preprocess_translation = np.array(preprocess_transform["translation"], dtype=np.float32)
            preprocess_scale = preprocess_transform["scale"]

        # point cloud input
        sbj_point_cloud = np.load(path / "subject_pointcloud.npz")["subject_pointcloud"]

        data_sample = {
            "sbj_point_cloud": sbj_point_cloud.astype(np.float32),
            "path": path,
            "object": sample.object,
            "obj_class": self.cfg.objname2classid[sample.object],
            "preprocess_scale": np.array(preprocess_scale, dtype=float),
            "preprocess_translation": np.array(preprocess_translation, dtype=float),
        }

        return data_sample

    def __getitem__(self, index):
        data_sample = self.get_data_sample(index)

        if not self.eval_mode:
            # point cloud input
            obj_mesh = trimesh.load(str(data_sample["path"] / "object.ply"), process=False)

            # calculate transformation for object keypoints
            obj_class_id = self.cfg.objname2classid[data_sample['object']]
            obj_keypoints_can = np.copy(self.canonical_obj_keypoints[obj_class_id]["cartesian"])
            obj_keypoints_can = obj_keypoints_can * data_sample["preprocess_scale"]
            obj_keypoints_triangles = obj_mesh.faces[self.canonical_obj_keypoints[obj_class_id]["triangles_ids"]]
            obj_keypoints_gt = np.array(trimesh.triangles.barycentric_to_points(
                obj_mesh.vertices[obj_keypoints_triangles],
                self.canonical_obj_keypoints[obj_class_id]["barycentric"]
            ), dtype=np.float32)

            data_sample = self.distort_keypoints(data_sample, obj_keypoints_can, obj_keypoints_gt)
            data_sample.update({
                "obj_center": obj_keypoints_gt.mean(axis=0).astype(np.float32),
            })

        data_sample["path"] = str(data_sample["path"])
        return data_sample

    def __len__(self):
        return len(self.data)
