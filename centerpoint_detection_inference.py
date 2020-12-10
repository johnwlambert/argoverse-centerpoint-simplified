


import argparse
import copy
import json
import logging
import os
import pickle 
import sys
import time
from collections import OrderedDict, defaultdict
from types import SimpleNamespace


try:
    import apex
except:
    print("No APEX!")

import torch.distributed as dist

from argoverse.utils.json_utils import read_json_file, save_json_dict
from argoverse.utils.pkl_utils import load_pkl_dictionary

import numpy as np
import torch
import yaml
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.utils import all_gather


"""
python tools/dist_test.py configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset.py --work_dir work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset  --checkpoint work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/epoch_20.pth  --speed_test 

"""

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    """ """
    args_dict = {
        'config': 'configs/centerpoint/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset.py',
        'work_dir': 'work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset',
        'checkpoint': 'work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/epoch_20.pth',
        'txt_result': False,
        'gpus': 1,
        'launcher': 'none',
        'speed_test': True,
        'local_rank': 0,
        'testset': False
    }
    args = SimpleNamespace(**args_dict)

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args



def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=log_level
        )
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    return logger


def example_to_device(example, device=None, non_blocking=False) -> dict:
    """ """
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels"]:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_second_losses(losses):
    """ """
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


def batch_processor(model, data, train_mode, **kwargs):
    """ """
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    example = example_to_device(data, device, non_blocking=False)

    del data

    if train_mode:
        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs
    else:
        return model(example, return_loss=False)


def main():
    """ """
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    pdb.set_trace()
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = model.cuda()
    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in ["metadata"]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            if args.local_rank == 0:
                prog_bar.update()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(predictions, args.work_dir)
    pkl_fpath = os.path.join(args.work_dir, 'prediction.pkl')
    predictions = load_pkl_dictionary()
    
    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()


