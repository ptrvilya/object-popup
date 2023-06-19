import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import torch
from popup.core.generator import Generator
from popup.core.evaluator import Evaluator
from popup.data.dataset import ObjectPopupDataset
from popup.models.object_popup import ObjectPopup
from popup.utils.exp import init_experiment


def main(cfg, generate, downsample, datasets=None):
    logging.info("Evaluation start.")
    # environment setup
    mp.set_start_method('spawn')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load datasets
    gen_datasets = []
    canonical_obj_meshes, canonical_obj_keypoints = dict(), dict()
    for dataset_name in cfg.datasets:
        if dataset_name == "grab":
            gen_datasets.append((dataset_name, ObjectPopupDataset(
                cfg, cfg.grab_path, eval_mode=True, objects=cfg.grab["gen_objects"], subjects=cfg.grab["gen_subjects"],
                actions=cfg.grab["gen_actions"]
            )))
        elif dataset_name == "behave":
            gen_datasets.append((dataset_name, ObjectPopupDataset(
                cfg, cfg.behave_path, objects=cfg.behave["gen_objects"], split_file=cfg.behave["gen_split_file"],
                eval_mode=True, downsample_factor=10 if downsample else 1
            )))
        canonical_obj_keypoints.update(gen_datasets[-1][1].canonical_obj_keypoints)
        canonical_obj_meshes.update(gen_datasets[-1][1].canonical_obj_meshes)
    gen_dataloaders = []
    for dataset_name, gen_dataset in gen_datasets:
        gen_dataloaders.append((dataset_name, torch.utils.data.DataLoader(
            gen_dataset, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=False
        )))

        logging.info(f"{dataset_name} dataset length: {len(gen_dataset)}")

    # create model
    if generate:
        if cfg.model_name == "object_popup":
            network = ObjectPopup(
                canonical_obj_keypoints=canonical_obj_keypoints, **cfg.model_params
            )
        else:
            raise RuntimeError(f"Unknown model {cfg.model_name}")

        # optionally generate predictions before computing metrics
        generator = Generator(torch.device("cuda:0"), cfg, canonical_obj_meshes, canonical_obj_keypoints)
        generator.load_checkpoint(network, cfg.checkpoint_path)
        if datasets is not None:
            gen_dataloaders = [(name, loader) for (name, loader) in gen_dataloaders if name in datasets]
        generator.generate(network, gen_dataloaders)

        evaluator = Evaluator(torch.device("cuda:0"), cfg)
    else:
        evaluator = Evaluator(torch.device("cuda:0"), cfg)
    evaluator.evaluate(datasets)
    logging.info("Evaluation end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate and optionally generate model predictions")

    parser.add_argument("scenario", type=Path)

    parser.add_argument("-c", "--project-config", type=Path, default="./project_config.toml")
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("--generate", "-g", action="store_true", help="Generate predictions before evaluating.")
    parser.add_argument("--downsample", action="store_true", help="Downsample datasets.")
    parser.add_argument("-d", "--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--exp-name", type=str, default="experiment",
                        help="Folder name for the experiment if -rc option is used.")

    resume = parser.add_mutually_exclusive_group(required=True)
    resume.add_argument("-ep", "--experiment-prefix", type=str,
                        help="Prefix of the experiment to continue with the desired epoch "
                             "in format <prefix>:<epoch>. Epoch==-1 corresponds to the latest available epoch.")
    resume.add_argument("-rc", "--resume-checkpoint", type=str,
                        help="Absolute path to the checkpoint to continue.")

    arguments = parser.parse_args()
    config = init_experiment(arguments, train=False)
    main(config, arguments.generate, arguments.downsample, arguments.datasets)
