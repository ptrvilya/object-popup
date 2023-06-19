import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict

import tomlkit


@dataclass
class ExperimentConfig:
    # EXPERIMENT
    exp_root: Path = field(init=True)
    exp_name: str = field(init=True)
    exp_folder: Path = field(init=False)
    checkpoints_folder: Path = field(init=False)
    exp_time: str = field(init=False)

    # DATASET
    datasets: List
    grab_path: Path = None
    behave_path: Path = None
    objname2classid: Dict = None
    workers: int = None
    obj_keypoints_npoints: int = 1500
    obj_keypoints_init: str = ""
    sampler: str = None

    # MODEL
    model_name: str = ""
    model_params: dict = field(default_factory=dict)

    # SELECTED CHECKPOINT
    # Used to pickup training or to generate predictions
    epoch: int = None
    checkpoint_path: Path = None

    # DATA SPLIT
    grab: Dict = field(default_factory=dict)
    behave: Dict = field(default_factory=dict)

    # GENERATOR PARAMS
    eval_temporal: bool = False
    undo_preprocessing_eval: bool = True

    # TRAINING
    batch_size: int = 8
    epochs: int = 100
    lr: float = 1e-5
    lr_scheduler: str = ""
    lr_scheduler_params: Dict = field(default_factory=dict)
    training_schedule: Dict = field(default_factory=dict)

    # LOSS
    loss_weights: Dict = None

    def __post_init__(self):
        # setting default mapping
        if self.objname2classid is None:
            self.objname2classid = {}
            for dataset in self.datasets:
                if dataset == "grab":
                    with (self.grab_path / "metadata.json").open("r") as fp:
                        dataset_objname2classid = json.load(fp)["obj_to_class"]
                elif dataset == "behave":
                    with (self.behave_path / "metadata.json").open("r") as fp:
                        dataset_objname2classid = json.load(fp)["obj_to_class"]
                self.objname2classid.update(dataset_objname2classid)

        self.exp_folder = self.exp_root / self.exp_name
        self.checkpoints_folder = self.exp_folder / "checkpoints"
        self.exp_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # set default values for data splits
        # GRAB
        self.grab = dict(self.grab)
        self.grab["train_subjects"] = self.grab.get("train_subjects", "train")
        self.grab["val_subjects"] = self.grab.get("val_subjects", "val")
        self.grab["objects"] = self.grab.get("objects", [])
        for k in ["train_objects", "val_objects", "gen_objects"]:
            self.grab[k] = self.grab.get(k,  self.grab["objects"])
        for k in ["train_actions", "val_actions", "gen_subjects","gen_actions"]:
            self.grab[k] = self.grab.get(k, [])

        grab_subjects_mapping = {
            "train": list(range(1, 9)),
            "val": [9],
            "test": [9, 10]
        }
        for k in ["train_subjects", "val_subjects", "gen_subjects"]:
            if isinstance(self.grab[k], str):
                assert self.grab[k] in ["train", "test", "val"], f"{k} string can be train, test or val"
                self.grab[k] = grab_subjects_mapping[self.grab[k]]
            else:
                assert len(self.grab[k]) <= 10, f"Incorrect {k}: {self.grab[k]}"
                assert set(self.grab[k]).issubset(set(range(1, 12))), f"Incorrect {k}: {self.grab[k]}"
            self.grab[k] = [f"s{s}" for s in self.grab[k]]

        # BEHAVE
        self.behave["objects"] = self.behave.get("objects", [])
        for k in ["train_objects", "val_objects", "gen_objects"]:
            self.behave[k] = self.behave.get(k,  self.behave["objects"])
        self.behave["gen_objects"] = self.behave.get("gen_objects", [])
        self.behave["gen_split_file"] = self.behave.get("gen_split_file", "./assets/behave_test.json")

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def _to_toml(self):
        toml_dict = dict()
        for key, value in self.__dict__.items():
            if value is None:
                continue

            if isinstance(value, Path):
                value = str(value)

            toml_dict[key] = value
        return toml_dict

    def dump(self, path: Union[str, Path] = None, prefix: str = ""):
        if path is None:
            path = self.exp_folder / "config.toml"
        else:
            path = Path(path)

        if len(prefix) > 0:
            path = path.parent / (prefix + path.name)

        with path.open("w") as fp:
            toml_string = tomlkit.dumps(self._to_toml())
            fp.write(toml_string)


def init_experiment(args, train=True, local=False):
    # Load directories from the project config
    project_config = tomlkit.parse(args.project_config.read_text())
    EXP_ROOT = Path(project_config.get("EXP_ROOT", "./experiments"))
    GRAB_PATH = Path(project_config.get("GRAB_PATH", ""))
    BEHAVE_PATH = Path(project_config.get("BEHAVE_PATH", ""))

    # Load training scenario
    scenario = tomlkit.parse(args.scenario.read_text())

    # Determine experiment type
    checkpoint_path, epoch = None, 0
    if args.experiment_prefix is not None:
        if ":" in args.experiment_prefix:
            # continue previous experiment
            exp_prefix, epoch = args.experiment_prefix.split(":")  # args.prefix for gen. exp
            epoch = int(epoch)
            exp_name = get_experiment_by_prefix(EXP_ROOT, exp_prefix)
        else:
            # try to resume nonetheless
            exp_name = get_experiment_by_prefix(EXP_ROOT, args.experiment_prefix)
            if exp_name is not None:
                epoch = -1
            else:
                # start new experiment from scratch
                epoch = 0
                exp_name = args.experiment_prefix
    elif args.resume_checkpoint is not None:
        # start new experiment from existing checkpoint
        if not(":" in args.resume_checkpoint):
            epoch = 0
            checkpoint_path = args.resume_checkpoint
        else:
            checkpoint_path, epoch = args.resume_checkpoint.split(":")
            epoch = int(epoch)
        prefix_number = get_next_experiment_number(EXP_ROOT)
        exp_name = f"{prefix_number:04d}_{args.exp_name}"
    else:
        raise NotImplementedError("Can't start experiment with the given config.")

    # Create config
    exp_config = ExperimentConfig(
        # EXPERIMENT
        exp_root=EXP_ROOT,
        exp_name=exp_name,
        checkpoint_path=checkpoint_path,
        # DATASET
        grab_path=GRAB_PATH,
        behave_path=BEHAVE_PATH,
        workers=args.workers if args.workers is not None else scenario.get("workers", 8),
        objname2classid=scenario.get("objname2classid", None),
        obj_keypoints_npoints=scenario.get("obj_keypoints_npoints", 1500),
        sampler=scenario.get("sampler", None),
        # DATA SAMPLING
        obj_keypoints_init=scenario.get("obj_keypoints_init", ""),
        # MODEL
        model_name=scenario.get("model_name", ""),
        model_params=scenario.get("model_params", dict()),
        # DATA SPLIT
        datasets=scenario.get("datasets", ["grab"]),
        grab=scenario.get("grab", {}),
        behave=scenario.get("behave", {}),
        # TRAINING
        epoch=epoch,
        epochs=scenario.get("epochs", 100),
        batch_size=args.batch_size if args.batch_size is not None else scenario.get("batch_size", 12),
        lr=args.lr if train and args.lr is not None else scenario.get("lr", 1e-5),
        lr_scheduler=scenario.get("lr_scheduler", ""),
        lr_scheduler_params=scenario.get("lr_scheduler_params", dict()),
        training_schedule=scenario.get("training_schedule", dict()),
        # EVAL
        eval_temporal=scenario.get("eval_temporal", False),
        undo_preprocessing_eval=scenario.get("undo_preprocessing_eval", True),
        # LOSS
        loss_weights=scenario.get("loss_weights", None),
    )

    # get checkpoint path for resume
    if args.experiment_prefix is not None and epoch != 0:
        exp_config.checkpoint_path = get_checkpoint_path(exp_config.checkpoints_folder, exp_config.epoch)

        # Folder with experiment is there, but there are no checkpoints
        if exp_config.checkpoint_path is None:
            exp_config.epoch = 0

    # determining epoch number
    if exp_config.epoch == -1:
        exp_config.epoch = int(exp_config.checkpoint_path.stem.split("_")[1])

    # set up logging
    exp_config.exp_folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(exp_config.exp_folder / "log"), level=logging.INFO, filemode='a+',
        format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    # save config in exp_folder
    prefix = "train_" if train else "gen_" + f"{exp_config.exp_time}_"
    exp_config.dump(prefix=prefix)

    # Init wandb for training
    if train and not(args.no_wandb):
        import wandb

        # init wandb, give multiple tries
        initialized = False
        if not local:
            logging.info("Trying to initialize WANDB")
            for _try, init_method in enumerate(["fork", "thread"]):
                try:
                    logging.info(f"Trying to initialize WANDB try {_try}")
                    wandb.init(
                        project="Object-synthesis", entity="ptrvilya", name=exp_config.exp_name,
                        dir=str(exp_config.exp_folder), resume=True,
                        settings=wandb.Settings(start_method=init_method)
                    )
                    logging.info("Initialized WANDB (inside)")
                    initialized = True
                    break
                except Exception as exc:
                    logging.error(f"Exception {exc}")
                    time.sleep(10)

            if initialized:
                logging.info("Initialized WANDB")
            else:
                raise RuntimeError("Unable to initialize WANDB")

    return exp_config


def get_experiment_by_prefix(exp_root: Union[Path, str], prefix: str):
    exp_root = Path(exp_root)
    exps = list(exp_root.glob(f"{prefix}*"))
    if len(exps) == 1:
        return exps[0].name
    else:
        if len(exps) == 0:
            logging.info(f"No experiments found for prefix {prefix}")
        else:
            raise RuntimeError(f"Found {len(exps)} experiments for prefix {prefix}")


def get_checkpoint_path(checkpoints_folder: Path, epoch: int):
    if epoch == -1:
        all_checkpoints = sorted(checkpoints_folder.glob("epoch_????.tar"))
        if len(all_checkpoints) > 0:
            checkpoint_path = all_checkpoints[-1]
        else:
            logging.info(f"No checkpoint for {checkpoints_folder}, epoch = {epoch}")
            checkpoint_path = None
    else:
        checkpoint_path = checkpoints_folder / f"epoch_{epoch:04d}.tar"

        if not checkpoint_path.is_file():
            logging.info(f"No checkpoint for {checkpoints_folder}, epoch = {epoch}")
            checkpoint_path = None

    return checkpoint_path


def get_next_experiment_number(exp_root: Union[Path, str]):
    # naming pattern: {number:04d}_{name:s}
    exp_root = Path(exp_root)
    exps = sorted(exp_root.glob("????_*"))
    if len(exps) == 0:
        exp_number = 0
    else:
        exp_number = int(str(exps[-1].name).split("_")[0]) + 1

    return exp_number
