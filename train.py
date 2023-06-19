import argparse
import logging
import os
import shutil
import logging
from pathlib import Path

import torch

from popup.core.trainer import Trainer
from popup.core.generator import Generator
from popup.data.dataset import ObjectPopupDataset
from popup.models.object_popup import ObjectPopup
from popup.utils.exp import init_experiment


def main(cfg):
    # environment setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    # save this file
    script_path = Path(os.path.abspath(__file__))
    shutil.copy(str(script_path), str(cfg.exp_folder / script_path.name))

    # load datasets
    train_datasets, val_datasets = [], []
    canonical_obj_meshes, canonical_obj_keypoints = dict(), dict()
    for dataset in cfg.datasets:
        if dataset == "grab":
            train_datasets.append(ObjectPopupDataset(
                cfg, cfg.grab_path, objects=cfg.grab["train_objects"], subjects=cfg.grab["train_subjects"],
                actions=cfg.grab["train_actions"],
            ))
            val_datasets.append(ObjectPopupDataset(
                cfg, cfg.grab_path, objects=cfg.grab["val_objects"], subjects=cfg.grab["val_subjects"],
                actions=cfg.grab["val_actions"], downsample_factor=5
            ))
        elif dataset == "behave":
            train_datasets.append(ObjectPopupDataset(
                cfg, cfg.behave_path, objects=cfg.behave["train_objects"], split_file=cfg.behave["train_split_file"],
            ))
            val_datasets.append(ObjectPopupDataset(
                cfg, cfg.behave_path, objects=cfg.behave["val_objects"], split_file=cfg.behave["val_split_file"],
                downsample_factor=1
            ))
        logging.info(f"Loaded {dataset} with {len(train_datasets[-1])} / {len(val_datasets[-1])}")
        canonical_obj_keypoints.update(train_datasets[-1].canonical_obj_keypoints)
        canonical_obj_meshes.update(train_datasets[-1].canonical_obj_meshes)

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    if cfg.sampler == "weighted":
        train_dataset_length = len(train_dataset)
        train_weights = []
        for dataset_name, dataset in zip(cfg.datasets, train_datasets):
            dataset_length = len(dataset)
            weights = torch.ones(dataset_length, dtype=torch.double)
            weights = (train_dataset_length / dataset_length) * weights

            if dataset_name == "grab":
                 weights *= 1.3

            train_weights.append(weights)
        train_weights = torch.cat(train_weights, dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(train_weights, num_samples=30000)
    elif cfg.sampler == "random":
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=30000)
    else:
        sampler = None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, drop_last=True,
        sampler=sampler, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2*cfg.batch_size, num_workers=cfg.workers, shuffle=False, pin_memory=True
    )
    logging.info(f"Train data length: {len(train_dataset)}")
    logging.info(f"Val data length: {len(val_dataset)}")

    if cfg.model_name == "object_popup":
        network = ObjectPopup(
            canonical_obj_keypoints=canonical_obj_keypoints, **cfg.model_params
        )
    else:
        raise RuntimeError(f"Unknown model {cfg.model_name}")

    generator = Generator(torch.device("cuda:0"), cfg, canonical_obj_meshes, canonical_obj_keypoints)
    trainer = Trainer(
        network, torch.device("cuda:0"), train_dataloader, val_dataloader, cfg,
        generator=generator
    )
    trainer.train_model(cfg.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train the model")

    parser.add_argument("scenario", type=Path)

    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-c", "--project-config", type=Path, default="./project_config.toml")
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("-lr", type=float)
    parser.add_argument("-nowb", "--no-wandb", action="store_true",
                        help="Don't use wandb to log statistics.")

    resume = parser.add_mutually_exclusive_group(required=True)
    resume.add_argument("-ep", "--experiment-prefix", type=str,
                        help="Prefix of the experiment to continue with the desired start epoch "
                             "in format <prefix>:<epoch>. Epoch==-1 corresponds to the lates availible epoch."
                             "No epoch corresponds to starting from scratch.")
    resume.add_argument("-rc", "--resume-checkpoint", type=str,
                       help="Absolute path to the checkpoint to continue with the desired start epoch "
                            "in format <path>:<epoch>.")

    arguments = parser.parse_args()
    config = init_experiment(arguments, train=True)
    main(config)
