"""
Script that unifies data preprocessing.
"""
import argparse
import subprocess
from pathlib import Path

from popup.utils.preprocess import generate_obj_keypoints_from_barycentric


def create_softlinks(source_dir, target_dir, subfolders):
    for folder in subfolders:
        subprocess.run(["ln", "-s", str(source_dir / folder), str(target_dir / folder)])


def main(args):
    # preprocess all data
    link_pairs = []
    for data_type in args.data_types:
        print(f"==>   {data_type}")

        if data_type == "rawpc":
            if not args.links_only:
                # for raw point clouds we only preprocess test split
                subprocess.run([
                    "python", "-m", "popup.data.preprocess_behave", "-i", str(args.behave),
                    "-o", str(args.target / f"behave_{data_type}"), "-c", f"configs/{data_type}.toml",
                    "--split", "test"
                ])

            # reusing object keypoints from smplh data via soft links
            link_pairs.append(
                (args.target / f"behave_smplh", args.target / f"behave_{data_type}")
            )
        else:
            for dataset in args.datasets:
                print(f"====> {dataset}")
                if dataset == "grab":
                    if not args.links_only:
                        # GRAB training data
                        command = [
                            "python", "-m", "popup.data.preprocess_grab", "-i", str(args.grab),
                            "-o", str(args.target / f"grab_{data_type}"), "-c", f"configs/{data_type}.toml",
                            "--downsample", "10fps", "--subjects", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",
                        ]
                        # object keypoints are generated only for smplh and then reused for consistency
                        if data_type == "smplh":
                            command += ["-g"]
                        subprocess.run(command)

                        # GRAB test data
                        subprocess.run([
                            "python", "-m", "popup.data.preprocess_grab", "-i", str(args.grab),
                            "-o", str(args.target / f"grab_{data_type}"), "-c", f"configs/{data_type}.toml",
                            "--downsample", "30fps", "--subjects", "s9", "s10"
                        ])

                    # reusing object keypoints from smplh data via soft links
                    if data_type != "smplh":
                        link_pairs.append(
                            (args.target / f"grab_smplh", args.target / f"grab_{data_type}")
                        )
                elif dataset == "behave" and data_type != "hands":
                    if not args.links_only:
                        # BEHAVE training data (from 30 fps annotations)
                        command = [
                            "python", "-m", "popup.data.preprocess_behave_30fps", "-i", str(args.behave),
                            "-o", str(args.target / f"behave_{data_type}"), "-c", f"configs/{data_type}.toml",
                            "--split", "train", "--downsample"
                        ]
                        # object keypoints are generated only for smplh and then reused for consistency
                        if data_type == "smplh":
                            command += ["-g"]
                        subprocess.run(command)

                        # BEHAVE training data (some sequences are only annotated with 1fps annotations)
                        subprocess.run([
                            "python", "-m", "popup.data.preprocess_behave", "-i", str(args.behave),
                            "-o", str(args.target / f"behave_{data_type}"), "-c", f"configs/{data_type}.toml",
                            "--split", "train", "--split-file", "./assets/behave_only_1fps.json"
                        ])

                        # BEHAVE test data (1fps annotations)
                        subprocess.run([
                            "python", "-m", "popup.data.preprocess_behave", "-i", str(args.behave),
                            "-o", str(args.target / f"behave_{data_type}"), "-c", f"configs/{data_type}.toml",
                            "--split", "test",
                        ])

                    # reuse object keypoints from smplh data
                    if data_type != "smplh":
                        link_pairs.append(
                            (args.target / f"behave_smplh",  args.target / f"behave_{data_type}")
                        )

                # reuse previously generated keypoints
                if data_type == "smplh" and args.reuse_keypoints:
                    generate_obj_keypoints_from_barycentric(
                        [], dataset, args.target / f"{dataset}_{data_type}",
                    )

    # create soft links
    for source, target in link_pairs:
        create_softlinks(source, target, ["object_meshes", "object_keypoints"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess data")

    parser.add_argument("datasets", type=str, nargs="+", choices=["grab", "behave"],
                        help="Datasets to process")
    parser.add_argument("-t", "--data-types", nargs="+", choices=["smpl", "smplh", "hands", "rawpc"],
                        help="Types of data to process (default: [\"smplh\"])", default=["smplh"])

    parser.add_argument("-G", "--grab", type=Path, help="Path to the GRAB dataset.")
    parser.add_argument("-B", "--behave", type=Path, help="Path to the BEHAVE dataset.")
    parser.add_argument("-T", "--target", type=Path, help="Path to the target directory with preprocessed data.")
    parser.add_argument("-r", "--reuse-keypoints", action="store_true",
                        help="Reuse previously generated keypoints from ./assets/<dataset name>_objects_keypoints.pkl"\
                             "that were used to train the model (for reproducibility).")

    parser.add_argument("-l", "--links-only", action="store_true",
                        help="Only generate links for merged folders.")

    arguments = parser.parse_args()
    arguments.datasets = set(arguments.datasets)
    arguments.data_types = set(arguments.data_types)
    main(arguments)
