import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import tqdm
import trimesh
import wandb
from torch.nn import functional as F


class Trainer(object):
    def __init__(
        self, model, device, train_loader, val_loader, cfg, generator, optimizer='Adam'
    ):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        else:
            raise NotImplementedError(f"No implementation for {optimizer} optimizer.")

        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.exp_folder = cfg.exp_folder
        self.checkpoints_folder = cfg.checkpoints_folder
        self.checkpoints_folder.mkdir(exist_ok=True)

        self.loss_weights = cfg.loss_weights

        if self.cfg.lr_scheduler == "multistep":
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **self.cfg.lr_scheduler_params)
        elif self.cfg.lr_scheduler == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, **self.cfg.lr_scheduler_params)
        else:
            self.lr_scheduler = None

        # generator instance for intermediate evaluation
        self.generator = generator

    def train_step(self, batch, epoch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.get_outputs(batch, epoch)
        loss, wandb_losses = self.compute_loss(batch, output)
        loss.backward()
        self.optimizer.step()

        return wandb_losses

    def get_outputs(self, batch, epoch):
        device = self.device

        sbj = batch.get('sbj_point_cloud').to(device)
        # common
        obj_keypoints_in = batch.get('obj_keypoints_in').to(device)
        obj_scales = batch.get('preprocess_scale').to(device)  # legacy, equal to 1 for GRAB and BEHAVE
        if epoch > self.cfg.training_schedule.get("start_pred_center", -1):
            object_center = None
        else:
            object_center = batch.get('obj_center').to(device)
        # model-specific
        if self.cfg.model_params.get("with_classifier", False) and \
            epoch > self.cfg.training_schedule.get("start_pred_class", -1):
            obj_classids = None
        else:
            obj_classids = batch.get('obj_class').to(device)

        output = self.model(
            sbj, obj_classids=obj_classids, obj_keypoints=obj_keypoints_in,
            obj_scales=obj_scales, obj_center=object_center
        )

        return output

    def compute_loss(self, batch, output):
        device = self.device

        # calculate losses
        loss = 0
        wandb_losses = {}
        batch_size = len(batch["path"])

        for key, weight in self.loss_weights.items():
            if key == "obj_class":
                gt_class = batch.get(key).to(device).unsqueeze(-1)
                gt_class = gt_class.squeeze(1)
                loss_i = F.cross_entropy(output[key], gt_class, reduction='none')
            elif key == "obj_keypoints_offsets":
                pred_offsets = output["offsets"]
                gt_offsets = batch.get(key).to(device)
                loss_i = F.mse_loss(pred_offsets, gt_offsets, reduction='none').mean(-1)
            elif key in ["obj_R", "obj_t"]:
                gt_val = batch.get(key).to(device)
                loss_i = F.mse_loss(output[key[-1]], gt_val.reshape(batch_size, -1), reduction='none')
            elif key == "obj_center":
                gt_val = batch.get(key).to(device)
                loss_i = F.mse_loss(output[key], gt_val.unsqueeze(1), reduction='none')
            else:
                raise NotImplementedError(f"No implementation for {key} loss.")

            loss_i = loss_i.mean()
            loss += weight * loss_i
            wandb_losses[key] = loss_i.detach().cpu().item()

        wandb_losses["total"] = loss.detach().cpu().item()

        return loss, wandb_losses

    def train_model(self, epochs):
        start = self.load_checkpoint()

        tqdm_bar_outer = tqdm.tqdm(range(start, epochs), total=len(range(start, epochs)), ncols=80)
        for epoch in tqdm_bar_outer:
            train_log = defaultdict(float)

            tqdm_bar = tqdm.tqdm(self.train_loader, total=len(self.train_loader), ncols=80, leave=False)
            for batch in tqdm_bar:
                train_losses = self.train_step(batch, epoch)
                tqdm_bar.set_description("Loss: {:.3f}".format(train_losses["total"]))
                for k, v in train_losses.items():
                    train_log[k] += v

            train_log = {f"TRAIN_{k}": v / len(self.train_loader) for k, v in train_log.items()}
            train_log["epoch"] = epoch

            if epoch % 1 == 0:
                self.save_checkpoint(epoch)

            if epoch % 1 == 0:
                val_log = defaultdict(float)

                self.model.eval()
                tqdm_bar = tqdm.tqdm(self.val_loader, total=len(self.val_loader), ncols=80, leave=False)

                val_metrics, val_counters = defaultdict(float), defaultdict(int)
                for batch in tqdm_bar:
                    val_losses, tmp_metrics, tmp_counters = self.val_step(batch, epoch)
                    tqdm_bar.set_description("Loss: {:.3f}".format(val_losses["total"]))
                    for k, v in val_losses.items():
                        val_log[k] += v
                    for k in tmp_metrics.keys():
                        val_metrics[k] += tmp_metrics[k]
                        val_counters[k] += tmp_counters[k]

                val_log = {f"VAL_{k}": v / len(self.val_loader) for k, v in val_log.items()}
                val_log["epoch"] = epoch
                for k in val_metrics.keys():
                    val_log[f"VAL_{k}"] = val_metrics[k] / (val_counters[k] + 1e-4)

                train_log.update(val_log)
                self.model.train()

            if wandb.run is not None:
                wandb.log(train_log)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def save_checkpoint(self, epoch):
        path = self.checkpoints_folder / f'epoch_{epoch:04d}.tar'
        torch.save({
            'epoch':epoch,'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
        }, path)

    def load_checkpoint(self):
        if self.cfg.checkpoint_path is None:
            return 0

        logging.info(f'Loaded checkpoint from: {self.cfg.checkpoint_path}')
        checkpoint = torch.load(self.cfg.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
        if self.lr_scheduler is not None and scheduler_state_dict is not None:
            self.lr_scheduler.load_state_dict(scheduler_state_dict)
        return epoch

    def compute_val_metrics(self, batch, output):
        METRICS = ["center_dist", "class_accuracy", "vertex_distance"]
        tmp_metrics = {metric: 0.0 for metric in METRICS}
        tmp_counters = {metric: 0 for metric in METRICS}

        batch_size = len(batch["path"])
        for i in range(batch_size):
            batch_i = {k: v[i] for k, v in batch.items()}
            output_i = {k: v[i] if isinstance(v, list) else v[i].cpu() for k, v in output.items()}

            pred_mesh, pred_class_scores = self.generator.get_mesh_from_predictions_wrapper(batch_i, output_i)
            gt_mesh = trimesh.load(str(Path(batch_i["path"]) / "object.ply"), process=False)

            if self.cfg.undo_preprocessing_eval:
                translation = batch_i["preprocess_translation"].numpy()
                scale = batch_i["preprocess_scale"].numpy()
                gt_mesh.vertices = gt_mesh.vertices / scale - translation
                pred_mesh.vertices = pred_mesh.vertices / scale - translation

            # center dist
            tmp_metrics["center_dist"] += np.linalg.norm(gt_mesh.vertices.mean() - pred_mesh.vertices.mean())
            tmp_counters["center_dist"] += 1

            # vertex dist
            if len(gt_mesh.vertices) == len(pred_mesh.vertices):
                tmp_metrics["vertex_distance"] += \
                    np.linalg.norm(gt_mesh.vertices - pred_mesh.vertices, axis=1).mean()
                tmp_counters["vertex_distance"] += 1

            # class accuracy
            if pred_class_scores is not None:
                predicted_class = np.argmax(pred_class_scores)
                gt_class = batch_i["obj_class"]

                tmp_metrics["class_accuracy"] += (predicted_class == gt_class)
                tmp_counters["class_accuracy"] += 1

        return tmp_metrics, tmp_counters

    def val_step(self, batch, epoch):
        with torch.no_grad():
            output = self.get_outputs(batch, epoch)
            loss, wandb_losses = self.compute_loss(batch, output)
            val_metrics, val_counters = self.compute_val_metrics(batch, output)

        return wandb_losses, val_metrics, val_counters
