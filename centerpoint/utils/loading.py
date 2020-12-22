

import os.path as osp
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pyntcloud



def load_ply_xyzir(ply_fpath: str) -> np.ndarray:
    """Load a point cloud file from a filepath.
    
    Args:
        ply_fpath: Path to a PLY file
    Returns:
        arr: Array of shape (N, 5)
    """

    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    i = np.array(data.points.intensity)[:, np.newaxis]
    ring_index = np.array(data.points.laser_number)[:, np.newaxis]

    return np.concatenate((x, y, z, i, ring_index), axis=1)



def read_file(path, tries=2, num_point_feature=4):
    """
    Per https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L249
    intensity is 4th, and laser number is 5th
    """
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            if 'ply' in path:
                points = load_ply_xyzir(path) 
            else:
                points = np.fromfile(path, dtype=np.float32)
                s = points.shape[0]
                if s % 5 != 0:
                    points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None

    return points


def remove_close(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Removes point too close within a certain radius from origin.

    Args:
        points: array of shape (4,N), e.g where N=34720
        radius: distance of ball, e.g. 1 meter, representing
            Radius below which points are removed.

    Returns:
        points: array of shape (4,M), where M < N, e.g. M=27007
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep: Dict[str,Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        sweep: dictionary with keys "lidar_path", "transform_matrix", "time_lag"
    
    Returns:
        points_sweep: (N,4) array, e.g. N=27007
        curr_times:  (N,1) array
    """
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"])).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(
        self,
        res: Dict[str,Any],
        info: Dict[str,Any]
    ) -> Tuple[ Dict[str,Any], Dict[str,Any] ]:
        """
        Args:
            res: dictionary with keys
                'lidar', 'metadata', 'calib', 'cam', 'mode', 'type'
            
                res['lidar'] is also a dictionary, with keys
                    'type', 'points', 'nsweeps', 'annotations', 'times', 'combined'

            info: dictionary with keys
                'lidar_path', 'cam_front_path', 'cam_intrinsic', 'token',
                'sweeps', 'ref_from_car', 'car_from_global', 'timestamp',
                'gt_boxes', 'gt_boxes_velocity', 'gt_names', 'gt_boxes_token'
        
                info['gt_boxes'] has a shape (N, 9), e.g. N=37
        """
        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset':
            """res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }"""
            pass # already load in the above function 
        else:
            return NotImplementedError

        return res, info



class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(
        self,
        res: Dict[str,Any],
        info: Dict[str,Any]
    ) -> Tuple[ Dict[str,Any], Dict[str,Any] ]:
        """
        Args:
            res: dictionary with keys
                'lidar', 'metadata', 'calib', 'cam', 'mode', 'type'
            info: dictionary with keys
                'lidar_path', 'cam_front_path', 'cam_intrinsic', 'token',
                'sweeps', 'ref_from_car', 'car_from_global', 'timestamp',
                'gt_boxes', 'gt_boxes_velocity', 'gt_names', 'gt_boxes_token'
        """
        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            # load (34720, 4) point cloud
            points = read_file(str(lidar_path))

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) <= len(
                info["sweeps"]
            ), "nsweeps {} should not greater than list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            # concat length-10 list, e.g. get (277783, 4) array
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        elif self.type == "WaymoDataset":
            path = info['path']
            obj = get_obj(path)

            points_xyz = obj["lidars"]["points_xyz"]
            points_feature = obj["lidars"]["points_feature"]

            # normalize intensity 
            points_feature[:, 0] = np.tanh(points_feature[:, 0])

            res["lidar"]["points"] = np.concatenate([points_xyz, points_feature], axis=-1)

            # read boxes 
            TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
            annos = obj['objects']
            num_points_in_gt = np.array([ann['num_points'] for ann in annos])
            gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 7)
            if len(gt_boxes) != 0:
                gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
            
            gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
            mask_not_zero = (num_points_in_gt > 0).reshape(-1)

            res["lidar"]["annotations"] = {
                "boxes": gt_boxes[mask_not_zero, :].astype(np.float32),
                "names": gt_names[mask_not_zero],
            }
        else:
            raise NotImplementedError

        return res, info
