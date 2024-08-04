import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock, Pool

mp.set_start_method("spawn", True)  # ! must be at top for VScode debugging

import glob
import logging
import os
import pathlib
import time
import numpy as np
from typing import Tuple, List

import cv2
import numpy as np
import torch.utils.data as data
import tqdm
from hover_net.dataloader.infer_loader import SerializeArray
from hover_net.misc.utils import (
    log_info,
    rm_n_mkdir,
)
from hover_net.misc.wsi_handler import get_file_handler

from . import base

logger = logging.getLogger(__name__)

thread_lock = Lock()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _init_worker_child(lock_):
    global lock
    lock = lock_


def _remove_inst(inst_map, remove_id_list):
    """Remove instances with id in remove_id_list.

    Args:
        inst_map: map of instances
        remove_id_list: list of ids to remove from inst_map
    """
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map


def calculate_patch_coordinates(
    image_shape: Tuple[int, int],
    input_patch_size: Tuple[int, int],
    output_patch_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the top-left coordinates of input and output patches for an image.

    This function determines the positions of overlapping patches to be extracted
    from an image, considering the difference between input and output patch sizes.

    Args:
        image_shape (Tuple[int, int]): Shape of the input image (height, width).
        input_patch_size (Tuple[int, int]): Size of the input patches (height, width).
        output_patch_size (Tuple[int, int]): Size of the output patches (height, width).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays containing the top-left coordinates
        for input and output patches respectively. Each array has shape (N, 2) where N
        is the number of patches and each row represents (y, x) coordinates.

    Raises:
        ValueError: If input sizes are invalid or incompatible.
    """
    logger.debug(f"Function input - image_shape: {image_shape}")
    logger.debug(f"Function input - input_patch_size: {input_patch_size}")
    logger.debug(f"Function input - output_patch_size: {output_patch_size}")

    # Convert inputs to numpy arrays for consistent processing
    image_shape = np.array(image_shape)
    input_patch_size = np.array(input_patch_size)
    output_patch_size = np.array(output_patch_size)

    logger.debug(f"Converted to numpy arrays - image_shape: {image_shape}")
    logger.debug(f"Converted to numpy arrays - input_patch_size: {input_patch_size}")
    logger.debug(f"Converted to numpy arrays - output_patch_size: {output_patch_size}")

    # Validate inputs
    if np.any(input_patch_size <= 0) or np.any(output_patch_size <= 0):
        logger.error("Invalid patch sizes detected")
        raise ValueError("Patch sizes must be positive.")
    if np.any(input_patch_size < output_patch_size):
        logger.error(f"Input patch size {input_patch_size} is smaller than output patch size {output_patch_size}")
        raise ValueError("Input patch size must be greater than or equal to output patch size.")
    if np.any(image_shape < input_patch_size):
        logger.error(f"Image shape {image_shape} is smaller than input patch size {input_patch_size}")
        raise ValueError("Image must be larger than the input patch size.")

    # Calculate the difference between input and output patch sizes
    patch_size_difference = input_patch_size - output_patch_size
    logger.debug(f"Patch size difference: {patch_size_difference}")

    # Calculate the number of patches in each dimension
    num_patches = np.floor((image_shape - patch_size_difference) / output_patch_size).astype(int) + 1
    logger.info(f"Number of patches: {num_patches}")

    # Calculate the coordinates of the last output patch
    last_patch_coord = (patch_size_difference // 2) + (num_patches * output_patch_size)
    logger.info(f"Last patch coordinate: {last_patch_coord}")

    # Ensure these are 1D arrays with 2 elements
    patch_size_difference = patch_size_difference.ravel()[:2]
    last_patch_coord = last_patch_coord.ravel()[:2]
    output_patch_size = output_patch_size.ravel()[:2]

    logger.info(f"Patch size difference after ravel: {patch_size_difference}")
    logger.info(f"last_patch_coord after ravel: {last_patch_coord}")
    logger.info(f"last_patch_coordafter ravel: {last_patch_coord}")


    # Generate lists of y and x coordinates for the top-left corners of output patches
    y_coords = np.arange(patch_size_difference[0] // 2, last_patch_coord[0], output_patch_size[0], dtype=np.int32)
    x_coords = np.arange(patch_size_difference[1] // 2, last_patch_coord[1], output_patch_size[1], dtype=np.int32)
    logger.info(f"Y coordinates: {y_coords}")
    logger.info(f"X coordinates: {x_coords}")

    # Create a grid of all possible (y, x) combinations
    y_grid, x_grid = np.meshgrid(y_coords, x_coords)
    output_patch_coords = np.stack([y_grid.flatten(), x_grid.flatten()], axis=-1)
    logger.debug(f"Output patch coordinates shape: {output_patch_coords.shape}")

    # Calculate the input patch coordinates
    input_patch_coords = output_patch_coords - patch_size_difference[np.newaxis, :] // 2
    logger.debug(f"Input patch coordinates shape: {input_patch_coords.shape}")

    return tuple(map(tuple, input_patch_coords)), tuple(map(tuple, output_patch_coords))

def _get_tile_info(
    image_shape: Tuple[int, int],
    tile_shape: Tuple[int, int],
    ambiguous_size: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate tile information for post-processing a large image.

    This function creates three sets of tiles:
    1. Normal tiles covering the entire image
    2. Boundary tiles covering the edges between normal tiles
    3. Cross tiles covering the intersections of four normal tiles

    Args:
        image_shape: Shape of the input image (height, width)
        tile_shape: Shape of each tile (height, width)
        ambiguous_size: Size of the ambiguous region at tile boundaries

    Returns:
        A tuple containing:
        - normal_tiles: Array of normal tile coordinates (N, 2, 2)
        - boundary_tiles: Array of boundary tile coordinates (M, 2, 2)
        - cross_tiles: Array of cross tile coordinates (P, 2, 2)
        Each tile is represented by its top-left and bottom-right coordinates.
    """
    image_shape = np.array(image_shape)
    tile_shape = np.array(tile_shape)

    # Generate normal tiles
    normal_tiles = _generate_normal_tiles(image_shape, tile_shape)

    # Generate boundary tiles
    boundary_tiles = _generate_boundary_tiles(normal_tiles, tile_shape, ambiguous_size)

    # Generate cross tiles
    cross_tiles = _generate_cross_tiles(normal_tiles, ambiguous_size)

    return normal_tiles, boundary_tiles, cross_tiles

def _generate_normal_tiles(image_shape: np.ndarray, tile_shape: np.ndarray) -> np.ndarray:
    """Generate non-overlapping tiles covering the entire image."""
    top_left_coords, _ = calculate_patch_coordinates(image_shape, tile_shape, tile_shape)
    top_left_coords = np.array(top_left_coords)

    bottom_right_coords = []
    for top_left in top_left_coords:
        bottom_right = np.minimum(top_left + tile_shape, image_shape)
        bottom_right_coords.append(bottom_right)

    bottom_right_coords = np.array(bottom_right_coords)
    return np.stack([top_left_coords, bottom_right_coords], axis=1)

def _generate_boundary_tiles(normal_tiles: np.ndarray, tile_shape: np.ndarray, ambiguous_size: int) -> np.ndarray:
    """Generate tiles covering the boundaries between normal tiles."""
    tile_grid_y = np.unique(normal_tiles[:, 0, 0])
    tile_grid_x = np.unique(normal_tiles[:, 0, 1])

    def create_boundary_tiles(coords, is_vertical):
        if is_vertical:
            top_left = np.meshgrid(tile_grid_y, coords - ambiguous_size)
            bottom_right = np.meshgrid(tile_grid_y + tile_shape[0], coords + ambiguous_size)
        else:
            top_left = np.meshgrid(coords - ambiguous_size, tile_grid_x)
            bottom_right = np.meshgrid(coords + ambiguous_size, tile_grid_x + tile_shape[1])
        
        return np.stack([_stack_coords(top_left), _stack_coords(bottom_right)], axis=1)

    vertical_boundaries = create_boundary_tiles(tile_grid_x[1:], is_vertical=True)
    horizontal_boundaries = create_boundary_tiles(tile_grid_y[1:], is_vertical=False)

    return np.concatenate([vertical_boundaries, horizontal_boundaries], axis=0)

def _generate_cross_tiles(normal_tiles: np.ndarray, ambiguous_size: int) -> np.ndarray:
    """Generate tiles covering the intersections of four normal tiles."""
    tile_grid_y = np.unique(normal_tiles[:, 0, 0])
    tile_grid_x = np.unique(normal_tiles[:, 0, 1])

    top_left = np.meshgrid(tile_grid_y[1:] - 2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size)
    bottom_right = np.meshgrid(tile_grid_y[1:] + 2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size)

    return np.stack([_stack_coords(top_left), _stack_coords(bottom_right)], axis=1)

def _stack_coords(coords: List[np.ndarray]) -> np.ndarray:
    """Stack coordinates into a 2D array."""
    return np.stack([coord.flatten() for coord in coords], axis=-1)

def _get_chunk_patch_info(
    img_shape, chunk_input_shape, patch_input_shape, patch_output_shape
):
    """Get chunk patch info. Here, chunk refers to tiles used during inference.

    Args:
        img_shape: input image shape
        chunk_input_shape: shape of tiles used for post processing
        patch_input_shape: input patch shape
        patch_output_shape: output patch shape

    """
    if any(len(shape) != 2 for shape in [img_shape, chunk_input_shape, patch_input_shape, patch_output_shape]):
        raise ValueError("All input shapes must be 2D (height, width)")

    # Ensure all shapes are 1D arrays with 2 elements
    img_shape = np.array(img_shape).ravel()[:2]
    chunk_input_shape = np.array(chunk_input_shape).ravel()[:2]
    patch_input_shape = np.array(patch_input_shape).ravel()[:2]
    patch_output_shape = np.array(patch_output_shape).ravel()[:2]

    def round_to_multiple(x, y):
        return np.floor(x / y) * y

    patch_diff_shape = patch_input_shape - patch_output_shape

    chunk_output_shape = chunk_input_shape - patch_diff_shape
    chunk_output_shape = round_to_multiple(
        chunk_output_shape, patch_output_shape
    ).astype(np.int64)
    chunk_input_shape = (chunk_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = calculate_patch_coordinates(
        img_shape, patch_input_shape, patch_output_shape
    )
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape
    patch_info_list = np.stack(
        [
            np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
            np.stack([patch_output_tl_list, patch_output_br_list], axis=1),
        ],
        axis=1,
    )

    chunk_input_tl_list, _ = calculate_patch_coordinates(
        img_shape, chunk_input_shape, chunk_output_shape
    )
    chunk_input_tl_list = np.array(chunk_input_tl_list)  # Convert back to numpy array
    chunk_input_br_list = chunk_input_tl_list + chunk_input_shape
    # * correct the coord so it stay within source image
    y_sel = np.nonzero(chunk_input_br_list[:, 0] > img_shape[0])[0]
    x_sel = np.nonzero(chunk_input_br_list[:, 1] > img_shape[1])[0]
    chunk_input_br_list[y_sel, 0] = (
        img_shape[0] - patch_diff_shape[0]
    ) - chunk_input_tl_list[y_sel, 0]
    chunk_input_br_list[x_sel, 1] = (
        img_shape[1] - patch_diff_shape[1]
    ) - chunk_input_tl_list[x_sel, 1]
    chunk_input_br_list[y_sel, 0] = round_to_multiple(
        chunk_input_br_list[y_sel, 0], patch_output_shape[0]
    )
    chunk_input_br_list[x_sel, 1] = round_to_multiple(
        chunk_input_br_list[x_sel, 1], patch_output_shape[1]
    )
    chunk_input_br_list[y_sel, 0] += chunk_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    chunk_input_br_list[x_sel, 1] += chunk_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    chunk_output_tl_list = chunk_input_tl_list + patch_diff_shape // 2
    chunk_output_br_list = chunk_input_br_list - patch_diff_shape // 2  # may off pixels
    chunk_info_list = np.stack(
        [
            np.stack([chunk_input_tl_list, chunk_input_br_list], axis=1),
            np.stack([chunk_output_tl_list, chunk_output_br_list], axis=1),
        ],
        axis=1,
    )

    return chunk_info_list, patch_info_list


def _post_proc_para_wrapper(pred_map_mmap_path, tile_info, func, func_kwargs):
    """Wrapper for parallel post processing."""
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode="r")
    tile_pred_map = wsi_pred_map_ptr[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]]
    tile_pred_map = np.array(tile_pred_map)  # from mmap to ram
    return func(tile_pred_map, **func_kwargs), tile_info


def _assemble_and_flush(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    """Assemble the results. Write to newly created holder for this wsi"""
    wsi_pred_map_ptr = np.load(wsi_pred_map_mmap_path, mmap_mode="r+")
    chunk_pred_map = wsi_pred_map_ptr[
        chunk_info[1][0][0] : chunk_info[1][1][0],
        chunk_info[1][0][1] : chunk_info[1][1][1],
    ]
    if patch_output_list is None:
        # chunk_pred_map[:] = 0 # zero flush when there is no-results
        # print(chunk_info.flatten(), 'flush 0')
        return

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        chunk_pred_map[
            pcoord[0] : pcoord[0] + pdata.shape[0],
            pcoord[1] : pcoord[1] + pdata.shape[1],
        ] = pdata
    # print(chunk_info.flatten(), 'pass')
    return


class InferManager(base.InferManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set default values
        self.ambiguous_size = kwargs.get("ambiguous_size", 128)
        self.tile_shape = kwargs.get("tile_shape", (256, 256))
        self.chunk_shape = kwargs.get("chunk_shape", (1024, 1024))
        self.save_thumb = kwargs.get("save_thumb", True)
        self.save_mask = kwargs.get("save_mask", True)
        self.batch_size = max(1, kwargs.get("batch_size", 1)) 

        self.patch_input_shape = kwargs.get("patch_input_shape", (256, 256))
        self.patch_output_shape = kwargs.get("patch_output_shape", (164, 164))
        self.proc_mag = kwargs.get("proc_mag", 40)
        self.cache_path = kwargs.get("cache_path", "./cache")
        self.nr_inference_workers = kwargs.get("nr_inference_workers", 8)
        self.nr_post_proc_workers = kwargs.get("nr_post_proc_workers", 8)

    def __run_model(self, patch_top_left_list, pbar_desc):
        logging.debug("Entering __run_model")
        logging.debug(f"patch_top_left_list length: {len(patch_top_left_list)}")
        logging.debug(f"self.batch_size: {self.batch_size}")

        if len(patch_top_left_list) == 0:
            logging.warning("patch_top_left_list is empty, returning early")
            return []
        
        dataset = SerializeArray(
            f"{self.cache_path}/cache_chunk.npy",
            patch_top_left_list,
            self.patch_input_shape,
        )

        batch_size = max(1, min(len(patch_top_left_list), self.batch_size))
        logging.debug(f"Calculated batch_size: {batch_size}")

        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=batch_size,
            drop_last=False,
        )

        pbar = tqdm.tqdm(
            desc=pbar_desc,
            leave=True,
            total=len(dataloader),
            ncols=80,
            ascii=True,
        )

        # run inference on input patches
        accumulated_patch_output = []
        for batch_data in dataloader:
            sample_data_list, sample_info_list = batch_data

            try:
                sample_output_list = self.run_step(sample_data_list)
                sample_info_list = sample_info_list.numpy()
                logging.debug(f"sample_output_list shape: {sample_output_list.shape}")
                logging.debug(f"sample_output_list dtype: {sample_output_list.dtype}")

                curr_batch_size = sample_output_list.shape[0]
                sample_output_list = np.split(sample_output_list, curr_batch_size, axis=0)
                sample_info_list = np.split(sample_info_list, curr_batch_size, axis=0)
                sample_output_list = list(zip(sample_info_list, sample_output_list))
                accumulated_patch_output.extend(sample_output_list)
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                raise
            pbar.update()
        pbar.close()
        return accumulated_patch_output

    def __select_valid_patches(self, patch_info_list, has_output_info=True):
        """Select valid patches from the list of input patch information.

        Args:
            patch_info_list: patch input coordinate information
            has_output_info: whether output information is given

        """
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        selected_indices = []
        for idx in range(patch_info_list.shape[0]):
            patch_info = patch_info_list[idx]
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            if has_output_info:
                output_bbox = patch_info[1] * down_sample_ratio
            else:
                output_bbox = patch_info * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e center regions)
            output_roi = self.wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            if np.sum(output_roi) > 0:
                selected_indices.append(idx)
        sub_patch_info_list = patch_info_list[selected_indices]
        return sub_patch_info_list

    def __get_raw_prediction(self, chunk_info_list, patch_info_list):
        """Process input tiles (called chunks for inference) with HoVer-Net.

        Args:
            chunk_info_list: list of inference tile coordinate information
            patch_info_list: list of patch coordinate information

        """
        logging.debug("Entering __get_raw_prediction")
        logging.debug(f"chunk_info_list shape: {chunk_info_list.shape}")
        logging.debug(f"patch_info_list shape: {patch_info_list.shape}")

        # Ensure patch_input_shape is a 1D array with 2 elements
        self.patch_input_shape = np.array(self.patch_input_shape).ravel()[:2]

        # 1 dedicated thread just to write results back to disk
        proc_pool = Pool(processes=1)
        wsi_pred_map_mmap_path = f"{self.cache_path}/pred_map.npy"

        def masking(x, a, b):
            return (a[0] <= x[:, 0]) & (x[:, 0] <= b[0]) & (a[1] <= x[:, 1]) & (x[:, 1] <= b[1])

        for idx in range(0, chunk_info_list.shape[0]):
            chunk_info = chunk_info_list[idx]
            # select patch basing on top left coordinate of input
            start_coord = chunk_info[0, 0]
            end_coord = chunk_info[0, 1] - self.patch_input_shape
            selection = masking(
                patch_info_list[:, 0, 0], start_coord, end_coord
            )
            chunk_patch_info_list = patch_info_list[selection]

            # further select only the patches within the provided mask
            chunk_patch_info_list = self.__select_valid_patches(chunk_patch_info_list)

            logging.debug(f"chunk_patch_info_list length: {len(chunk_patch_info_list)}")

            # there no valid patches, so flush 0 and skip
            if chunk_patch_info_list.shape[0] == 0:
                logging.warning(f"No valid patches for chunk {idx}, skipping")
                proc_pool.apply_async(
                    _assemble_and_flush, args=(wsi_pred_map_mmap_path, chunk_info, None)
                )
                continue

            # shift the coordinate from wrt slide to wrt chunk
            chunk_patch_info_list -= chunk_info[:, 0]
            chunk_data = self.wsi_handler.read_region(
                chunk_info[0][0][::-1], (chunk_info[0][1] - chunk_info[0][0])[::-1]
            )
            chunk_data = np.array(chunk_data)[..., :3]
            np.save(f"{self.cache_path}/cache_chunk.npy", chunk_data)

            pbar_desc = f"Process Chunk {idx}/{chunk_info_list.shape[0]}"
            patch_output_list = self.__run_model(
                chunk_patch_info_list[:, 0, 0], pbar_desc
            )

            proc_pool.apply_async(
                _assemble_and_flush,
                args=(wsi_pred_map_mmap_path, chunk_info, patch_output_list),
            )
        proc_pool.close()
        proc_pool.join()
        return

    def __dispatch_post_processing(self, tile_info_list, callback):
        """Post processing initialisation."""
        proc_pool = None
        if self.nr_post_proc_workers > 0:
            proc_pool = ProcessPoolExecutor(self.nr_post_proc_workers)

        future_list = []
        wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]

            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                "nr_types": self.method["model_args"]["nr_types"],
                "return_centroids": True,
            }

            # TODO: standarize protocol
            if proc_pool is not None:
                proc_future = proc_pool.submit(
                    _post_proc_para_wrapper,
                    wsi_pred_map_mmap_path,
                    tile_info,
                    self.post_proc_func,
                    func_kwargs,
                )
                # ! manually poll future and call callback later as there is no guarantee
                # ! that the callback is called from main thread
                future_list.append(proc_future)
            else:
                results = _post_proc_para_wrapper(
                    wsi_pred_map_mmap_path, tile_info, self.post_proc_func, func_kwargs
                )
                callback(results)
        if proc_pool is not None:
            silent_crash = False
            # loop over all to check state a.k.a polling
            for future in as_completed(future_list):
                # ! silent crash, cancel all and raise error
                if future.exception() is not None:
                    silent_crash = True
                    # ! cancel somehow leads to cascade error later
                    # ! so just poll it then crash once all future
                    # ! acquired for now
                    # for future in future_list:
                    #     future.cancel()
                    # break
                else:
                    callback(future.result())
            assert not silent_crash
        return

    def _parse_args(self, run_args):
        """Parse command line arguments and set as instance variables."""
        for variable, value in run_args.items():
            self.__setattr__(variable, value)
        # to tuple
        self.chunk_shape = [self.chunk_shape, self.chunk_shape]
        self.tile_shape = [self.tile_shape, self.tile_shape]
        self.patch_input_shape = [self.patch_input_shape, self.patch_input_shape]
        self.patch_output_shape = [self.patch_output_shape, self.patch_output_shape]
        return

    def process_single_file(self, wsi_path, msk_path, output_dir):
        """Process a single whole-slide image and save the results.

        Args:
            wsi_path: path to input whole-slide image
            msk_path: path to input mask. If not supplied, mask will be automatically generated.
            output_dir: path where output will be saved

        """
        # TODO: customize universal file handler to sync the protocol
        print(f"Processing file: {wsi_path}")
        print(f"Output directory: {output_dir}")

        ambiguous_size = self.ambiguous_size
        tile_shape = (np.array(self.tile_shape)).astype(np.int64)
        chunk_input_shape = np.array(self.chunk_shape)
        patch_input_shape = np.array(self.patch_input_shape)
        patch_output_shape = np.array(self.patch_output_shape)

        path_obj = pathlib.Path(wsi_path)
        wsi_ext = path_obj.suffix
        wsi_name = path_obj.stem

        start = time.perf_counter()
        self.wsi_handler = get_file_handler(wsi_path, backend=wsi_ext)
        self.wsi_proc_shape = self.wsi_handler.get_dimensions(self.proc_mag)
        self.wsi_handler.prepare_reading(
            read_mag=self.proc_mag, cache_path="%s/src_wsi.npy" % self.cache_path
        )
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1])  # to Y, X

        if msk_path is not None and os.path.isfile(msk_path):
            self.wsi_mask = cv2.imread(msk_path)
            self.wsi_mask = cv2.cvtColor(self.wsi_mask, cv2.COLOR_BGR2GRAY)
            self.wsi_mask[self.wsi_mask > 0] = 1
        else:
            log_info(
                "WARNING: No mask found, generating mask via thresholding at 1.25x!"
            )

            from skimage import morphology

            # simple method to extract tissue regions using intensity thresholding and morphological operations
            def simple_get_mask():
                scaled_wsi_mag = 1.25  # ! hard coded
                wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=scaled_wsi_mag)
                gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                mask = morphology.remove_small_objects(
                    mask == 0, min_size=16 * 16, connectivity=2
                )
                mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
                mask = morphology.binary_dilation(mask, morphology.disk(16))
                return mask

            self.wsi_mask = np.array(simple_get_mask() > 0, dtype=np.uint8)
        if np.sum(self.wsi_mask) == 0:
            log_info("Skip due to empty mask!")
            return
        if self.save_mask:
            cv2.imwrite("%s/mask/%s.png" % (output_dir, wsi_name), self.wsi_mask * 255)
        if self.save_thumb:
            wsi_thumb_rgb = self.wsi_handler.get_full_img(read_mag=1.25)
            cv2.imwrite(
                "%s/thumb/%s.png" % (output_dir, wsi_name),
                cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2BGR),
            )

        # * declare holder for output
        # create a memory-mapped .npy file with the predefined dimensions and dtype
        # TODO: dynamicalize this, retrieve from model?
        out_ch = 3 if self.method["model_args"]["nr_types"] is None else 4
        self.wsi_inst_info = {}
        # TODO: option to use entire RAM if users have too much available, would be faster than mmap
        self.wsi_inst_map = np.lib.format.open_memmap(
            "%s/pred_inst.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape),
            dtype=np.int32,
        )
        # self.wsi_inst_map[:] = 0 # flush fill

        # warning, the value within this is uninitialized
        self.wsi_pred_map = np.lib.format.open_memmap(
            "%s/pred_map.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            dtype=np.float32,
        )
        # ! for debug
        # self.wsi_pred_map = np.load('%s/pred_map.npy' % self.cache_path, mmap_mode='r')
        end = time.perf_counter()
        log_info("Preparing Input Output Placement: {0}".format(end - start))

        # * raw prediction
        start = time.perf_counter()
        chunk_info_list, patch_info_list = _get_chunk_patch_info(
            self.wsi_proc_shape,
            chunk_input_shape,
            patch_input_shape,
            patch_output_shape,
        )

        logging.debug(f"chunk_info_list shape: {chunk_info_list.shape}")
        logging.debug(f"patch_info_list shape: {patch_info_list.shape}")

        # get the raw prediction of HoVer-Net, given info of inference tiles and patches
        self.__get_raw_prediction(chunk_info_list, patch_info_list)
        end = time.perf_counter()
        log_info("Inference Time: {0}".format(end - start))

        # TODO: deal with error banding
        ##### * post processing
        ##### * done in 3 stages to ensure that nuclei at the boundaries are dealt with accordingly
        start = time.perf_counter()
        tile_coord_set = _get_tile_info(self.wsi_proc_shape, tile_shape, ambiguous_size)
        # 3 sets of patches are extracted and are dealt with differently
        # tile_grid_info: central region of post processing tiles
        # tile_boundary_info: boundary region of post processing tiles
        # tile_cross_info: region at corners of post processing tiles
        tile_grid_info, tile_boundary_info, tile_cross_info = tile_coord_set
        tile_grid_info = self.__select_valid_patches(tile_grid_info, False)
        tile_boundary_info = self.__select_valid_patches(tile_boundary_info, False)
        tile_cross_info = self.__select_valid_patches(tile_cross_info, False)

        ####################### * Callback can only receive 1 arg
        def post_proc_normal_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                return  # when there is nothing to do

            top_left = pos_args[1][::-1]

            # ! WARNING:
            # ! inst ID may not be contiguous,
            # ! hence must use max as safeguard

            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())
            for inst_id, inst_info in inst_info_dict.items():
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            self.wsi_inst_map[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]] = (
                pred_inst
            )

            pbar.update()  # external
            return

        ####################### * Callback can only receive 1 arg
        def post_proc_fixing_tile_callback(args):
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()  # external
                return  # when there is nothing to do

            top_left = pos_args[1][::-1]

            # for fixing the boundary, keep all nuclei split at boundary (i.e within unambigous region)
            # of the existing prediction map, and replace all nuclei within the region with newly predicted

            # ! WARNING:
            # ! inst ID may not be contiguous,
            # ! hence must use max as safeguard

            # ! must get before the removal happened
            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())

            # * exclude ambiguous out from old prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_inst = self.wsi_inst_map[
                tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]
            ]
            roi_inst = np.copy(roi_inst)
            roi_edge = np.concatenate(
                [roi_inst[[0, -1], :].flatten(), roi_inst[:, [0, -1]].flatten()]
            )
            roi_boundary_inst_list = np.unique(roi_edge)[1:]  # exclude background
            roi_inner_inst_list = np.unique(roi_inst)[1:]
            roi_inner_inst_list = np.setdiff1d(
                roi_inner_inst_list, roi_boundary_inst_list, assume_unique=True
            )
            roi_inst = _remove_inst(roi_inst, roi_inner_inst_list)
            self.wsi_inst_map[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]] = (
                roi_inst
            )
            for inst_id in roi_inner_inst_list:
                self.wsi_inst_info.pop(inst_id, None)

            # * exclude unambiguous out from new prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_edge = pred_inst[roi_inst > 0]  # remove all overlap
            boundary_inst_list = np.unique(roi_edge)  # no background to exclude
            inner_inst_list = np.unique(pred_inst)[1:]
            inner_inst_list = np.setdiff1d(
                inner_inst_list, boundary_inst_list, assume_unique=True
            )
            pred_inst = _remove_inst(pred_inst, boundary_inst_list)

            # * proceed to overwrite
            for inst_id in inner_inst_list:
                # ! happen because we alrd skip thoses with wrong
                # ! contour (<3 points) within the postproc, so
                # ! sanity gate here
                if inst_id not in inst_info_dict:
                    log_info("Nuclei id=%d not in saved dict WRN1." % inst_id)
                    continue
                inst_info = inst_info_dict[inst_id]
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            pred_inst = roi_inst + pred_inst
            self.wsi_inst_map[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]] = (
                pred_inst
            )

            pbar.update()  # external
            return

        #######################
        def pbar_creator(x, y):
            return tqdm.tqdm(desc=y, leave=True, total=int(len(x)), ncols=80, ascii=True)
        pbar = pbar_creator(tile_grid_info, "Post Proc Phase 1")
        # * must be in sequential ordering
        self.__dispatch_post_processing(tile_grid_info, post_proc_normal_tile_callback)
        pbar.close()

        pbar = pbar_creator(tile_boundary_info, "Post Proc Phase 2")
        self.__dispatch_post_processing(
            tile_boundary_info, post_proc_fixing_tile_callback
        )
        pbar.close()

        pbar = pbar_creator(tile_cross_info, "Post Proc Phase 3")
        self.__dispatch_post_processing(tile_cross_info, post_proc_fixing_tile_callback)
        pbar.close()

        end = time.perf_counter()
        log_info("Total Post Proc Time: {0}".format(end - start))

        # ! cant possibly save the inst map at high res, too large
        start = time.perf_counter()
        if self.save_mask or self.save_thumb:
            json_path = "%s/json/%s.json" % (output_dir, wsi_name)
        else:
            json_path = "%s/%s.json" % (output_dir, wsi_name)
        self.__save_json(json_path, self.wsi_inst_info, mag=self.proc_mag)
        end = time.perf_counter()
        log_info("Save Time: {0}".format(end - start))

    def process_wsi_list(self, run_args):
        """Process a list of whole-slide images.

        Args:
            run_args: arguments as defined in run_infer.py

        """
        self._parse_args(run_args)

        if not os.path.exists(self.cache_path):
            rm_n_mkdir(self.cache_path)

        if not os.path.exists(self.output_dir + "/json/"):
            rm_n_mkdir(self.output_dir + "/json/")
        if self.save_thumb:
            if not os.path.exists(self.output_dir + "/thumb/"):
                rm_n_mkdir(self.output_dir + "/thumb/")
        if self.save_mask:
            if not os.path.exists(self.output_dir + "/mask/"):
                rm_n_mkdir(self.output_dir + "/mask/")

        wsi_path_list = glob.glob(self.input_dir + "/*")
        wsi_path_list.sort()  # ensure ordering
        for wsi_path in wsi_path_list[:]:
            if os.path.isdir(wsi_path):
                continue
            wsi_base_name = pathlib.Path(wsi_path).stem
            msk_path = "%s/%s.png" % (self.input_mask_dir, wsi_base_name)
            if self.save_thumb or self.save_mask:
                output_file = "%s/json/%s.json" % (self.output_dir, wsi_base_name)
            else:
                output_file = "%s/%s.json" % (self.output_dir, wsi_base_name)
            if os.path.exists(output_file):
                log_info("Skip: %s" % wsi_base_name)
                continue
            try:
                log_info("Process: %s" % wsi_base_name)
                self.process_single_file(wsi_path, msk_path, self.output_dir)
                log_info("Finish")
            except Exception:
                logging.exception("Crash")
        rm_n_mkdir(self.cache_path)  # clean up all cache
        return
