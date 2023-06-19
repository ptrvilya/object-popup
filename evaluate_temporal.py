import argparse
import logging
from pathlib import Path

import torch
from popup.data.dataset import ObjectPopupDataset
from popup.core.evaluator import Evaluator
from popup.core.generator_temporal import TemporalGenerator
from popup.models.object_popup import ObjectPopup
from popup.utils.exp import init_experiment


def main(cfg, generate, downsample, datasets, sigma, kernel):
    logging.info("Temporal evaluation start.")
    # environment setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load datasets
    gen_datasets = []
    canonical_obj_meshes, canonical_obj_keypoints = dict(), dict()
    for dataset_name in cfg.datasets:
        if dataset_name == "grab":
            dataset = ObjectPopupDataset(
                cfg, cfg.grab_path, eval_mode=True, objects=cfg.grab["gen_objects"], subjects=cfg.grab["gen_subjects"],
                actions=cfg.grab["gen_actions"]
            )

            gen_datasets.append((dataset_name, cfg.grab_path, dataset.data))
        elif dataset_name == "behave":
            dataset = ObjectPopupDataset(
                cfg, cfg.behave_path, objects=cfg.behave["gen_objects"], split_file=cfg.behave["gen_split_file"],
                eval_mode=True, downsample_factor=10 if downsample else 1
            )

            gen_datasets.append((dataset_name, cfg.behave_path, dataset.data))
        canonical_obj_keypoints.update(dataset.canonical_obj_keypoints)
        canonical_obj_meshes.update(dataset.canonical_obj_meshes)
        logging.info(f"{dataset_name} dataset length: {len(dataset.data)}")

    if generate:
        if cfg.model_name == "object_popup":
            network = ObjectPopup(
                canonical_obj_keypoints=canonical_obj_keypoints, **cfg.model_params
            )
        else:
            raise RuntimeError(f"Unknown model {cfg.model_name}")

        if datasets is not None:
            gen_datasets = [(n, p, d) for (n, p, d) in gen_datasets if n in datasets]

        generator = TemporalGenerator(network, torch.device("cuda:0"), gen_datasets, cfg,
                                      canonical_obj_meshes, canonical_obj_keypoints)

        generator.generate_offline(kernel, sigma)
        evaluator = Evaluator(torch.device("cuda:0"), cfg)
    else:
        evaluator = Evaluator(torch.device("cuda:0"), cfg)
    evaluator.evaluate(datasets)
    logging.info("Temporal evaluation end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate model predictions")

    parser.add_argument("scenario", type=Path)

    parser.add_argument("-c", "--project-config", type=Path, default="./project_config.toml")
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("-w", "--workers", type=int)
    parser.add_argument("--generate", "-g", action="store_true", help="Generate predictions before evaluating.")
    parser.add_argument("--downsample", action="store_true", help="Downsample datasets.")
    parser.add_argument("-d", "--datasets", type=str, nargs="+", default=None)
    parser.add_argument("--exp-name", type=str, default="experiment",
                        help="Folder name for the experiment if -rc option is used.")

    parser.add_argument("-s", type=float, default=0.05)
    parser.add_argument("-k", type=int, default=3)

    resume = parser.add_mutually_exclusive_group(required=True)
    resume.add_argument("-ep", "--experiment-prefix", type=str,
                        help="Prefix of the experiment to continue with the desired epoch "
                             "in format <prefix>:<epoch>. Epoch==-1 corresponds to the latest available epoch.")
    resume.add_argument("-rc", "--resume-checkpoint", type=str,
                        help="Absolute path to the checkpoint to continue.")

    arguments = parser.parse_args()
    config = init_experiment(arguments, train=False)
    config.eval_temporal = True
    main(config, arguments.generate, arguments.downsample, arguments.datasets, arguments.s, arguments.k)
