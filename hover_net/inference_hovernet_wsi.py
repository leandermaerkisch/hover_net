from infer.wsi import InferManager
from typing import Dict
import modal
import logging
import os
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = modal.App("hovernet-inference-internal")
volume = modal.Volume.from_name("hovernet", create_if_missing=True)

# alternative: image = modal.Image.from_dockerfile("./Dockerfile", add_python="3.9")
image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install([
        "libgl1-mesa-glx",
        "libopenslide0",
        "gcc",
        "python3-dev",
    ])
    .pip_install(
        "hover-net==0.0.10"
    )
)

@app.function(image=image, gpu=modal.gpu.A100(size="40GB"), timeout=3600, volumes={"/models": volume}, mounts=[modal.Mount.from_local_dir("./dataset", remote_path="/root/dataset")])
def run_hovernet_inference_wsi():
    # import hover_net
    logger.info("Starting HoVerNet inference")
    # logger.info(f"hover_net version: {hover_net.__version__}")


    model_path = "/models/hovernet_fast_pannuke_type_tf2pytorch.tar"
    input_dir = "/root/dataset/wsi/input"
    output_dir = "/root/dataset/wsi/output"

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input directory contents: {os.listdir(input_dir)}")
    logger.info(f"Looking for model at: {model_path}")
    logger.info(f"Current directory contents: {os.listdir('/')}")
    logger.info(f"Models directory contents: {os.listdir('/models')}")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    method_args: Dict[str, Dict] = {
        'method': {
            'model_args': {
                'nr_types': 6,
                'mode': 'fast'
            },
            'model_path': model_path
        },
        'type_info_path': '/models/type_info.json'
    }

    logger.info(f"Method arguments: {method_args}")

    run_args = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'cache_path': 'cache',
        'input_mask_dir': None,
        'proc_mag': 40,
        'ambiguous_size': 128,
        'chunk_shape': 10000,
        'tile_shape': 2048,
        'save_thumb': True,
        'save_mask': True,
    }

    logger.info(f"Run arguments: {run_args}")

    logger.info("Initializing InferManager")
    infer = InferManager(**method_args)

    logger.debug(f"InferManager class: {infer.__class__}")
    logger.debug(f"InferManager attributes: {infer.__dict__}")
    
    logger.info("Starting WSI processing")

    logger.info(f"Processing files in directory: {run_args['input_dir']}")

    # Now call process_wsi_list
    try:
        infer.process_wsi_list(run_args)
    except Exception as e:
        logger.error(f"Error in process_wsi_list: {str(e)}", exc_info=True)
        raise

    logger.info("HoVerNet inference completed")

if __name__ == "__main__":
    logger.info("Starting Modal app")

    with app.run(show_progress=False):
        run_hovernet_inference_wsi.remote()

    logger.info("Modal app execution completed")