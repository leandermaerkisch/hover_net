import numpy as np

import torch.utils.data as data



####
class SerializeFileList(data.Dataset):
    """Read a single file as multiple patches of same shape, perform the padding beforehand."""

    def __init__(self, img_list, patch_info_list, patch_size):
        super().__init__()
        self.patch_size = patch_size

        self.img_list = img_list
        self.patch_info_list = patch_info_list

        if not self.patch_info_list:
            raise ValueError("patch_info_list is empty")

        print(f"SerializeFileList initialized with {len(self.patch_info_list)} patches")

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        global_curr_img_idx = patch_info[-1]
        curr_img = self.img_list[global_curr_img_idx]

        # Extract patch
        start_h, start_w = patch_info[0], patch_info[1]
        end_h = start_h + self.patch_size
        end_w = start_w + self.patch_size
        patch = curr_img[start_h:end_h, start_w:end_w]

        return patch, patch_info

    def __len__(self):
        return len(self.patch_info_list)

    def __iter__(self):
        print(f"Starting iteration over {len(self.patch_info_list)} patches")
        for idx in range(len(self)):
            yield self.__getitem__(idx)


####
class SerializeArray(data.Dataset):
    def __init__(self, mmap_array_path, patch_info_list, patch_size, preproc=None):
        super().__init__()
        self.patch_size = patch_size

        # use mmap as intermediate sharing, else variable will be duplicated
        # accross torch worker => OOM error, open in read only mode
        self.image = np.load(mmap_array_path, mmap_mode="r")

        self.patch_info_list = patch_info_list
        self.preproc = preproc
        return

    def __len__(self):
        return len(self.patch_info_list)

    def __getitem__(self, idx):
        patch_info = self.patch_info_list[idx]
        patch_data = self.image[
            patch_info[0] : patch_info[0] + self.patch_size[0],
            patch_info[1] : patch_info[1] + self.patch_size[1],
        ]
        if self.preproc is not None:
            patch_data = self.preproc(patch_data)
        return patch_data, patch_info
