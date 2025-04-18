import os
import argparse

from eth3d_downloader import eth3d_download_dataset
from eth3d_rescale import eth3d_rescale_dataset

ETH3D_DATASETS_TRAINING = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains"
]

# datasets for which we don't have evaluation
ETH3D_DATASETS_TEST = [
    "botanical_garden",
    "boulders",
    "bridge",
    "door",
    "exhibition_hall",
    "lecture_room",
    "living_room",
    "lounge",
    "observatory",
    "old_computer",
    "statue",
    "terrace_2"
]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", "-o", type=str,
                        help="Output directory to download the scans to.")
    parser.add_argument("--width", type=int, default=None,
                        help="Rescaling all images and cameras after downloading the original dataset. "
                             "Rescale does not modify the original dataset, rather it creates "
                             "a directory with the rescaled alternative.")
    parser.add_argument("datasets", type=str, nargs="+",
                        choices=ETH3D_DATASETS_TRAINING+ETH3D_DATASETS_TEST,
                        help="Any number of High-res dataset names from ETH3D to download.")

    parser.add_argument("--with-jpg", action="store_true",
                        help="download original jpg images")
    parser.add_argument("--with-depth", action="store_true",
                        help="download ground truth depth as well")

    return parser.parse_args()


# download all datasets
if __name__ == "__main__":
    args = parse_args()

    for i, dataset in enumerate(args.datasets):
        if not os.path.exists(os.path.join(args.outdir, dataset)):
            print(f"{i+1} / {len(args.datasets)}: downloading {dataset}")
            eth3d_download_dataset(dataset_name=dataset, datasets_dir=args.outdir,
                                   with_original_jpg=args.with_jpg,
                                   with_eval=(dataset in ETH3D_DATASETS_TRAINING),
                                   with_depth=(args.with_depth and (dataset in ETH3D_DATASETS_TRAINING)))
        if not args.width is None and not os.path.exists(os.path.join(args.outdir, dataset, f"{dataset}_dslr_undistorted_{args.width}")):
            print(f"{i+1} / {len(args.datasets)}: rescaling {dataset}")
            eth3d_rescale_dataset(os.path.join(args.outdir, dataset, f"{dataset}_dslr_undistorted"), new_width=args.width)
            