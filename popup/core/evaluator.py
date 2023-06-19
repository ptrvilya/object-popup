import json
import logging
import pickle as pkl
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from tqdm.autonotebook import tqdm

from ..data.dataset import get_sbj_to_obj_act
from ..utils.exp import ExperimentConfig
from ..utils.parallel_map import parallel_map
from ..utils.preprocess import trimesh_load


class Evaluator:
    def __init__(self, device, cfg: ExperimentConfig):
        self.device = device

        self.cfg = cfg
        self.objname2classid = cfg.objname2classid

    def get_evaluation_sequences(self, path_template, dataset_name):
        if dataset_name == "grab":
            subject_to_obj_act = get_sbj_to_obj_act(
                path_template, self.cfg.grab_path, self.cfg.grab["gen_subjects"], self.cfg.grab["gen_objects"],
                self.cfg.grab["gen_actions"]
            )
            dataset_folder = self.cfg.grab_path
        elif dataset_name == "behave":
            dataset_folder = self.cfg.behave_path
            with open(self.cfg.behave["gen_split_file"], "r") as fp:
                split = json.load(fp)

            subject_to_obj_act = defaultdict(list)
            for subject, obj_act in split:
                _obj_act = obj_act.split("_")
                obj, act = _obj_act[0], "_".join(_obj_act[1:])
                subject_to_obj_act[subject].append((obj, act))

        return dataset_folder, subject_to_obj_act

    @staticmethod
    def eval_worker(t_stamp, gt_sequence_path, generated_folder, gt_class, undo_preprocessing_eval=False):
        metrics, counters = dict(), dict()

        # center distance, chamfer distance and V2V distance
        if (generated_folder / "posed_mesh").is_dir():
            # Load gt mesh
            gt_mesh = trimesh_load(gt_sequence_path / t_stamp / "object.ply")

            # Load predicted mesh
            pred_mesh_name = generated_folder / "posed_mesh" / f"{t_stamp}.obj"
            if not pred_mesh_name.is_file():
                raise RuntimeError(f"No mesh found for {generated_folder} - {t_stamp}")
            pred_mesh = trimesh_load(pred_mesh_name)

            # undo preprocessing
            if undo_preprocessing_eval:
                with (gt_sequence_path / t_stamp / "preprocess_transform.pkl").open("rb") as fp:
                    preprocess_transform = pkl.load(fp)
                # undo scaling and translation
                scale, translation = preprocess_transform["scale"], preprocess_transform["translation"]
                gt_mesh.vertices = gt_mesh.vertices / scale - translation
                pred_mesh.vertices = pred_mesh.vertices / scale - translation
                # undo rotation and translation
                R, t = preprocess_transform["R"], preprocess_transform["t"]
                R = Rotation.from_matrix(R).inv()
                gt_mesh.vertices = R.apply(gt_mesh.vertices - t)
                pred_mesh.vertices = R.apply(pred_mesh.vertices - t)

            # center distance
            metrics["center_dist"] = np.linalg.norm(gt_mesh.vertices.mean() - pred_mesh.vertices.mean())
            counters["center_dist"] = 1

            # v2v
            if len(gt_mesh.vertices) == len(pred_mesh.vertices):
                metrics["vertex_distance"] = np.linalg.norm(gt_mesh.vertices - pred_mesh.vertices, axis=1).mean()
                counters["vertex_distance"] = 1

            # bi-directional chamfer distance
            # gt nn
            gt_mesh_points = gt_mesh.sample(10000)
            gt_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric="l2", n_jobs=-1
            ).fit(gt_mesh_points)
            # pred nn
            pred_mesh_points = pred_mesh.sample(10000)
            posed_nn = NearestNeighbors(
                n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric="l2", n_jobs=-1
            ).fit(pred_mesh_points)
            # distances
            pred_to_gt = np.mean(gt_nn.kneighbors(pred_mesh_points)[0])
            gt_to_pred = np.mean(posed_nn.kneighbors(gt_mesh_points)[0])
            metrics["chamfer_dist"] = pred_to_gt + gt_to_pred
            counters["chamfer_dist"] = 1

        # class accuracy
        if (generated_folder / "class_scores").is_dir():
            class_scores = np.load(generated_folder / "class_scores" / f"{t_stamp}.npy")
            predicted_class = np.argmax(class_scores)

            metrics["class_accuracy"] = int(predicted_class == gt_class)
            counters["class_accuracy"] = 1

        return metrics, counters

    def evaluate(self, datasets=None):
        dataset_to_metrics = dict()
        if self.cfg.eval_temporal:
            output_string = f"{self.cfg.exp_name} - temporal\n"
        else:
            output_string = f"{self.cfg.exp_name}\n"

        eval_datasets = self.cfg.datasets if datasets is None else datasets

        for dataset_name in eval_datasets:
            if self.cfg.eval_temporal:
                viz_folder = self.cfg.exp_folder / "visualization" / f"{str(self.cfg.epoch)}_temporal" / dataset_name
            else:
                viz_folder = self.cfg.exp_folder / "visualization" / str(self.cfg.epoch) / dataset_name

            path_template = "{subject}/{object}_{action}/"
            dataset_folder, subject_to_obj_act = self.get_evaluation_sequences(path_template, dataset_name)

            subject_to_metrics = dict()
            subject_to_counters = dict()
            METRICS = [
               "center_dist", "chamfer_dist", "class_accuracy", "vertex_distance",
            ]
            for subject, obj_acts in tqdm(subject_to_obj_act.items(), total=len(subject_to_obj_act), ncols=80):
                subject_to_metrics[subject] = defaultdict(list)
                subject_to_counters[subject] = defaultdict(list)

                for (obj, act) in tqdm(obj_acts, total=len(obj_acts), ncols=80, leave=False):
                    tmp_metrics = {metric: 0.0 for metric in METRICS}
                    tmp_counters = {metric: 0 for metric in METRICS}

                    gt_sequence_path = dataset_folder / path_template.format(subject=subject, object=obj, action=act)
                    generated_folder = viz_folder / path_template.format(subject=subject, object=obj, action=act)
                    t_stamps = sorted([t_stamp.name for t_stamp in gt_sequence_path.glob("t*")])

                    # Same actions for each t_stamp
                    input_data = [{
                        "t_stamp": t_stamp,
                        "generated_folder": generated_folder,
                        "gt_sequence_path": gt_sequence_path,
                        "gt_class": self.objname2classid[obj],
                        "undo_preprocessing_eval": self.cfg.undo_preprocessing_eval
                    } for t_stamp in t_stamps]
                    sequence_results = parallel_map(
                        input_data, self.eval_worker, use_kwargs=True, n_jobs=10, tqdm_kwargs={"leave": False}
                    )

                    # aggregate results
                    for t_metrics, t_counters in sequence_results:
                        for metric in METRICS:
                            tmp_metrics[metric] += t_metrics.get(metric, 0.0)
                            tmp_counters[metric] += t_counters.get(metric, 0)

                    # add results to main dict
                    for metric in METRICS:
                        subject_to_metrics[subject][metric].append(tmp_metrics[metric])
                        subject_to_counters[subject][metric].append(tmp_counters[metric])
                    subject_to_metrics[subject]["sequences"].append((obj, act))
                    subject_to_counters[subject]["sequences"].append((obj, act))

            # calculate per subject and total values
            total_metrics = defaultdict(list)
            total_counters = defaultdict(list)
            for subject in subject_to_metrics.keys():
                for metric in METRICS:
                    sum_metric = np.sum(subject_to_metrics[subject][metric], axis=0)
                    total_metrics[metric].append(sum_metric)

                    sum_counters = np.sum(subject_to_counters[subject][metric], axis=0)
                    total_counters[metric].append(sum_counters)

            output_string += f"==> Epoch {self.cfg.epoch} Dataset {dataset_name} Total:\n"
            for metric in METRICS:
                total_counters_sum = np.sum(total_counters[metric])
                if total_counters_sum > 0:
                    output_string += f"\t{metric:<25}{np.sum(total_metrics[metric]) / total_counters_sum:.10f}\n"
            dataset_to_metrics[dataset_name] = (subject_to_metrics, subject_to_counters)
        # save results
        logging.info(output_string)
        with (self.cfg.exp_folder / f"metrics_epoch{self.cfg.epoch:04d}.txt").open("a") as fp:
            fp.write(output_string)

        with (self.cfg.exp_folder / f"metrics_epoch{self.cfg.epoch:04d}.pkl").open("wb") as fp:
            pkl.dump(dataset_to_metrics, fp)

        print(output_string)
