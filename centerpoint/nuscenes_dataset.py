import sys
import pickle
import json
import random
import operator
from typing import Any, Dict

import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from centerpoint.dataset.point_cloud_dataset import PointCloudDataset
from centerpoint.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)


class Reformat(object):
    def __init__(self, **kwargs):
        double_flip = kwargs.get('double_flip', False)
        self.double_flip = double_flip 

    def __call__(self, res: Dict[str, Any], info: Dict[str, Any]):
        """
        
        Returns:
            if 'double_flip' mode, return length-4 list of bundled data for each of the mirrored variants
        """
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]

        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"]
        )

        if "anchors" in res["lidar"]["targets"]:
            anchors = res["lidar"]["targets"]["anchors"]
            data_bundle.update(dict(anchors=anchors))

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

        calib = res.get("calib", None)
        if calib:
            data_bundle["calib"] = calib

        if res["mode"] != "test":
            annos = res["lidar"]["annotations"]
            data_bundle.update(annos=annos, )

        if res["mode"] == "train":

            if "reg_targets" in res["lidar"]["targets"]: # anchor based
                labels = res["lidar"]["targets"]["labels"]
                reg_targets = res["lidar"]["targets"]["reg_targets"]
                reg_weights = res["lidar"]["targets"]["reg_weights"]

                data_bundle.update(
                    dict(labels=labels, reg_targets=reg_targets, reg_weights=reg_weights)
                )
            else: # anchor free
                data_bundle.update(res["lidar"]["targets"])

        elif self.double_flip:
            # y axis 
            yflip_points = res["lidar"]["yflip_points"]
            yflip_voxels = res["lidar"]["yflip_voxels"] 
            yflip_data_bundle = dict(
                metadata=meta,
                points=yflip_points,
                voxels=yflip_voxels["voxels"],
                shape=yflip_voxels["shape"],
                num_points=yflip_voxels["num_points"],
                num_voxels=yflip_voxels["num_voxels"],
                coordinates=yflip_voxels["coordinates"],
                annos=annos,  
            )
            if calib:
                yflip_data_bundle["calib"] = calib 

            # x axis 
            xflip_points = res["lidar"]["xflip_points"]
            xflip_voxels = res["lidar"]["xflip_voxels"] 
            xflip_data_bundle = dict(
                metadata=meta,
                points=xflip_points,
                voxels=xflip_voxels["voxels"],
                shape=xflip_voxels["shape"],
                num_points=xflip_voxels["num_points"],
                num_voxels=xflip_voxels["num_voxels"],
                coordinates=xflip_voxels["coordinates"],
                annos=annos, 
            )
            if calib:
                xflip_data_bundle["calib"] = calib

            # double axis flip 
            double_flip_points = res["lidar"]["double_flip_points"]
            double_flip_voxels = res["lidar"]["double_flip_voxels"] 
            double_flip_data_bundle = dict(
                metadata=meta,
                points=double_flip_points,
                voxels=double_flip_voxels["voxels"],
                shape=double_flip_voxels["shape"],
                num_points=double_flip_voxels["num_points"],
                num_voxels=double_flip_voxels["num_voxels"],
                coordinates=double_flip_voxels["coordinates"],
                annos=annos, 
            )
            if calib:
                double_flip_data_bundle["calib"] = calib

            return [data_bundle, yflip_data_bundle, xflip_data_bundle, double_flip_data_bundle], info

        return data_bundle, info





class NuScenesDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, ring_index

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0, # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        version="v1.0-trainval",
        **kwargs,
    ):
        super(NuScenesDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self.nsweeps = nsweeps
        # print('self.nsweeps', self.nsweeps)
        assert self.nsweeps > 0, "At least input one sweep please!"
        # assert self.nsweeps > 0, "At least input one sweep please!"
        print(self.nsweeps)

        self._info_path = info_path
        self._class_names = class_names

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        self._num_point_features = NuScenesDataset.NumPointFeatures
        self._name_mapping = general_to_detection

        self.version = version
        self.eval_version = "detection_cvpr_2019"

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._nusc_infos_all)
        self._nusc_infos = self._nusc_infos_all[: self.frac]

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _nusc_infos_all = pickle.load(f)

        if not self.test_mode:  # if training
            self.frac = int(len(_nusc_infos_all) * 0.25)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _nusc_infos_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {k: len(v) / duplicated_samples for k, v in _cls_infos.items()}

            self._nusc_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._nusc_infos += np.random.choice(
                    cls_infos, int(len(cls_infos) * ratio)
                ).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._nusc_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._nusc_infos) for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_nusc_infos_all, dict):
                self._nusc_infos = []
                for v in _nusc_infos_all.values():
                    self._nusc_infos.extend(v)
            else:
                self._nusc_infos = _nusc_infos_all

    def __len__(self):

        if not hasattr(self, "_nusc_infos"):
            self.load_infos(self._info_path)

        return len(self._nusc_infos)

    def get_sensor_data(self, idx):
        """ """
        info = self._nusc_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
                "lidar_fpath": info["sample"]["lidar_path"]
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

