import argparse
import torch
import logging
import os
from typing import Dict, Any
from hover_net.misc.utils import log_info

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

def create_parser():
    parser = argparse.ArgumentParser(description="HoVer-Net Pytorch Inference")
    parser.add_argument("--gpu", default="0", help="GPU list.")
    parser.add_argument("--nr_types", type=int, default=0, help="Number of nuclei types to predict.")
    parser.add_argument("--type_info_path", default="", help="Path to type info json with info about prediction classes.")
    parser.add_argument("--model_path", required=True, help="Path to saved checkpoint.")
    parser.add_argument("--model_mode", default="fast", choices=["original", "fast"], help="HoVer-Net mode.")
    parser.add_argument("--nr_inference_workers", type=int, default=8, help="Number of inference workers.")
    parser.add_argument("--nr_post_proc_workers", type=int, default=16, help="Number of post-processing workers.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ["tile", "wsi"]:
        sub = subparsers.add_parser(cmd, help=f"Run inference on {cmd}")
        sub.add_argument("--input_dir", required=True, help="Input directory path.")
        sub.add_argument("--output_dir", required=True, help="Output directory path.")
        if cmd == "tile":
            sub.add_argument("--mem_usage", type=float, default=0.2, help="Memory usage for caching.")
            sub.add_argument("--draw_dot", action="store_true", help="Draw nuclei centroid.")
            sub.add_argument("--save_qupath", action="store_true", help="Save in QuPath format.")
            sub.add_argument("--save_raw_map", action="store_true", help="Save raw prediction.")
        elif cmd == "wsi":
            sub.add_argument("--cache_path", default="cache", help="Cache path.")
            sub.add_argument("--input_mask_dir", default="", help="Input mask directory.")
            sub.add_argument("--proc_mag", type=int, default=40, help="Processing magnification.")
            sub.add_argument("--ambiguous_size", type=int, default=128, help="Ambiguous region size.")
            sub.add_argument("--chunk_shape", type=int, default=10000, help="Chunk shape for processing.")
            sub.add_argument("--tile_shape", type=int, default=2048, help="Tile shape for processing.")
            sub.add_argument("--save_thumb", action="store_true", help="Save thumbnail.")
            sub.add_argument("--save_mask", action="store_true", help="Save mask.")
        else:
            raise ValueError(f"Invalid command: {cmd}")

    return parser

def main():
    setup_logging()
    args = create_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    nr_gpus = torch.cuda.device_count()
    log_info(f"Detect #GPUS: {nr_gpus}")

    method_args = {
        "method": {
            "model_args": {
                "nr_types": args.nr_types if args.nr_types > 0 else None,
                "mode": args.model_mode,
            },
            "model_path": args.model_path,
        },
        "type_info_path": args.type_info_path or None,
    }

    run_args = {
        "batch_size": max(1, args.batch_size * nr_gpus),
        "nr_inference_workers": args.nr_inference_workers,
        "nr_post_proc_workers": args.nr_post_proc_workers,
        "patch_input_shape": 256 if args.model_mode == "fast" else 270,
        "patch_output_shape": 164 if args.model_mode == "fast" else 80,
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
    }

    if args.command == "tile":
        run_args.update({k: getattr(args, k) for k in ["mem_usage", "draw_dot", "save_qupath", "save_raw_map"]})
        from hover_net.infer.tile import InferManager
    elif args.command == "wsi":
        run_args.update({k: getattr(args, k) for k in ["cache_path", "input_mask_dir", "proc_mag", "ambiguous_size", "chunk_shape", "tile_shape", "save_thumb", "save_mask"]})
        from hover_net.infer.wsi import InferManager
    else:
        raise ValueError(f"Invalid command: {args.command}")

    InferManager(**method_args).process_file_list(run_args)

if __name__ == "__main__":
    main()