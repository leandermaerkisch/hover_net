"""Main HoVer-Net training script."""

import argparse
import cv2
cv2.setNumThreads(0)
import glob
import inspect
import json
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from config import Config
from dataloader.train_loader import FileLoader
from misc.utils import rm_n_mkdir
from run_utils.engine import RunEngine
from run_utils.utils import (
    check_manual_seed,
    colored,
    convert_pytorch_checkpoint,
)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_seed = torch.randint(0, 2**32, (1,))[0].cpu().item() + worker_id
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return

class TrainManager(Config):
    def __init__(self):
        super().__init__()

    def view_dataset(self, mode="train"):
        self.nr_gpus = 1
        import matplotlib.pyplot as plt

        check_manual_seed(self.seed)
        phase_list = self.model_config["phase_list"][0]
        target_info = phase_list["target_info"]
        prep_func, prep_kwargs = target_info["viz"]
        dataloader = self._get_datagen(2, mode, target_info["gen"])
        for batch_data in dataloader:
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.show()
        self.nr_gpus = -1

    def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=0, fold_idx=0):
        nr_procs = nr_procs if not self.debug else 0

        file_list = []
        data_dir_list = self.train_dir_list if run_mode == "train" else self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob(f"{dir_path}/*.npy"))
        file_list.sort()

        assert len(file_list) > 0, f"No .npy found for `{run_mode}`, please check `{run_mode}_dir_list` in `config.py`"
        print(f"Dataset {run_mode}: {len(file_list)}")
        input_dataset = FileLoader(
            file_list,
            mode=run_mode,
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=target_gen,
            **self.shape_info[run_mode],
        )

        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=batch_size * self.nr_gpus,
            shuffle=run_mode == "train",
            drop_last=run_mode == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader

    def run_once(self, opt, run_engine_opt, log_dir, prev_log_dir=None, fold_idx=0):
        check_manual_seed(self.seed)

        log_info = {}
        if self.logging:
            rm_n_mkdir(log_dir)
            tfwriter = SummaryWriter(log_dir=log_dir)
            json_log_file = f"{log_dir}/stats.json"
            with open(json_log_file, "w") as json_file:
                json.dump({}, json_file)
            log_info = {
                "json_file": json_log_file,
                "tfwriter": tfwriter,
            }

        loader_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            loader_dict[runner_name] = self._get_datagen(
                opt["batch_size"][runner_name],
                runner_name,
                opt["target_info"]["gen"],
                nr_procs=runner_opt["nr_procs"],
                fold_idx=fold_idx,
            )

        def get_last_chkpt_path(prev_phase_dir, net_name):
            stat_file_path = f"{prev_phase_dir}/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            epoch_list = [int(v) for v in info.keys()]
            last_chkpts_path = f"{prev_phase_dir}/{net_name}_epoch={max(epoch_list)}.tar"
            return last_chkpts_path

        net_run_info = {}
        net_info_opt = opt["run_info"]
        for net_name, net_info in net_info_opt.items():
            assert inspect.isclass(net_info["desc"]) or inspect.isfunction(net_info["desc"]), \
                "`desc` must be a Class or Function which instantiate NEW objects !!!"
            net_desc = net_info["desc"]()

            pretrained_path = net_info["pretrained"]
            if pretrained_path is not None:
                if pretrained_path == -1:
                    pretrained_path = get_last_chkpt_path(prev_log_dir, net_name)
                    net_state_dict = torch.load(pretrained_path)["desc"]
                else:
                    chkpt_ext = os.path.basename(pretrained_path).split(".")[-1]
                    if chkpt_ext == "npz":
                        net_state_dict = dict(np.load(pretrained_path))
                        net_state_dict = {k: torch.from_numpy(v) for k, v in net_state_dict.items()}
                    elif chkpt_ext == "tar":
                        net_state_dict = torch.load(pretrained_path)["desc"]

                colored_word = colored(net_name, color="red", attrs=["bold"])
                print(f"Model `{colored_word}` pretrained path: {pretrained_path}")

                net_state_dict = convert_pytorch_checkpoint(net_state_dict)
                load_feedback = net_desc.load_state_dict(net_state_dict, strict=False)
                print("Missing Variables: \n", load_feedback[0])
                print("Detected Unknown Variables: \n", load_feedback[1])

            net_desc = DataParallel(net_desc)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net_desc = net_desc.to(device)
            print(f"Using device: {device}")

            optimizer, optimizer_args = net_info["optimizer"]
            optimizer = optimizer(net_desc.parameters(), **optimizer_args)
            scheduler = net_info["lr_scheduler"](optimizer)
            net_run_info[net_name] = {
                "desc": net_desc,
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "extra_info": net_info["extra_info"],
            }

        assert "train" in run_engine_opt, "No engine for training detected in description file"

        runner_dict = {}
        for runner_name, runner_opt in run_engine_opt.items():
            runner_dict[runner_name] = RunEngine(
                dataloader=loader_dict[runner_name],
                engine_name=runner_name,
                run_step=runner_opt["run_step"],
                run_info=net_run_info,
                log_info=log_info,
            )

        for runner_name, runner in runner_dict.items():
            callback_info = run_engine_opt[runner_name]["callbacks"]
            for event, callback_list in callback_info.items():
                for callback in callback_list:
                    if callback.engine_trigger:
                        triggered_runner_name = callback.triggered_engine_name
                        callback.triggered_engine = runner_dict[triggered_runner_name]
                    runner.add_event_handler(event, callback)

        main_runner = runner_dict["train"]
        main_runner.state.logging = self.logging
        main_runner.state.log_dir = log_dir
        main_runner.run(opt["nr_epochs"])

        print("\n")
        print("########################################################")
        print("########################################################")
        print("\n")

    def run(self):
        self.nr_gpus = torch.cuda.device_count()
        print(f"Detect #GPUS: {self.nr_gpus}")

        phase_list = self.model_config["phase_list"]
        engine_opt = self.model_config["run_engine"]

        prev_save_path = None
        for phase_idx, phase_info in enumerate(phase_list):
            save_path = self.log_dir if len(phase_list) == 1 else f"{self.log_dir}/{phase_idx:02d}/"
            self.run_once(phase_info, engine_opt, save_path, prev_log_dir=prev_save_path)
            prev_save_path = save_path

def main():
    parser = argparse.ArgumentParser(description="HoVer-Net Training Script")
    parser.add_argument("--gpu", default="0,1,2,3", help="Comma separated GPU list")
    parser.add_argument("--view", choices=["train", "valid"], help="Visualize images after augmentation")
    args = parser.parse_args()

    trainer = TrainManager()

    if args.view:
        trainer.view_dataset(args.view)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        trainer.run()

if __name__ == "__main__":
    main()