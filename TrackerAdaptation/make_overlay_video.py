import pickle

from tqdm import tqdm
from pathlib import Path
import logging
import numpy as np
from main import spark_setup_test, spark_config_parser, parse_args
from adapt.wrapper import FaceTrackerWrapper
from adapt.face_decoder import MultiFLAREDecoder
from utils.dataset import DeviceDataLoader, find_collate, load_img

from argparse import Namespace
import torch


@torch.no_grad()
def main(wrapper: FaceTrackerWrapper, args: Namespace, dataset, test_dir: str):
    assert isinstance(wrapper.decoder, MultiFLAREDecoder), "Only MultiFLARE decoder supports textures."
    device = wrapper.device
    out_dir: Path = args.out_dir / "flame" / test_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataloader = DeviceDataLoader(dataset, device=device, batch_size=1, collate_fn=find_collate(dataset), num_workers=0)

    data = {"shape": dataset.shape_params.cpu().numpy(), "expression": [], "pose": []}
    for views in tqdm(dataloader):
        run_dict = wrapper(views, training=False, visdict=True)
        extra = run_dict["values"]["extra"]

        data["canonical_verts"] = extra["canonical_verts"]
        data["expression"].append(extra["expression"].cpu().numpy())
        data["pose"].append(extra["pose"].cpu().numpy())

    data["expression"] = np.concatenate(data["expression"], axis=0)
    data["pose"] = np.concatenate(data["pose"], axis=0)
    with open(str(out_dir / "flame_data.pkl"), "wb") as f:
        pickle.dump(data, f)

    logging.info(f"{test_dir} Done")


if __name__ == "__main__":
    parser = spark_config_parser()
    parser.add_argument("--out", type=str, default="", help="Output directory, relative to the experiment dir or absolute")
    parser.add_argument("--visu_interval", type=int, default=1, help="Interval at which to sample frames for visualization")
    parser.add_argument("--n_frames", type=int, default=-1, help="Number of frames to process (-1 for whole video)")
    parser.add_argument("--framerate", type=int, default=30, help="Framerate for generating the edited video")
    parser.add_argument("--smooth_crops", action="store_true", help="Smooth the crops to reduce jittering")
    parser.add_argument("--texture", type=str, help="Path to a texture for rendering")
    parser.add_argument("--opacity", type=float, default=1.0, help="Opacity of the texture overlay")
    parser.add_argument("--export_meshes", action="store_true", help="Export a .obj mesh per frame")
    args = parse_args(parser)

    if not args.out:
        args.out = f"video_overlay_{args.tracker_resume}"

    for test_dir in tqdm(args.test_dirs):
        wrapper, dataset_test = spark_setup_test(args, test_dir, render_mode="full")
        main(wrapper, args, dataset_test, test_dir)
