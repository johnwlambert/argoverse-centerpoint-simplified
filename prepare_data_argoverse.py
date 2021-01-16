
import argparse
import copy
import os.path as osp
import pdb
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from argoverse.utils.pkl_utils import save_pkl_dictionary
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat, quat_argo2scipy
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

try:
    from nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
except:
    print("nuScenes devkit not Found!")


from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import get_calibration_config




# equivalent of `general_to_detection` dict
argoverse_name_to_nuscenes_name = {
    'VEHICLE': 'car',
    'ON_ROAD_OBSTACLE': 'ignore', # or 'traffic_cone' ?
    'TRAILER': 'trailer',
    'BUS': 'bus',
    'LARGE_VEHICLE': 'vehicle', # or 'truck'?
    'EMERGENCY_VEHICLE': 'ignore',
    'MOTORCYCLE': 'motorcycle',
    'BICYCLE': 'bicycle',
    'PEDESTRIAN': 'pedestrian',
    'BICYCLIST': 'pedestrian', # check if right?
    'MOTORCYCLIST': 'pedestrian', # check if right?
    'ANIMAL': 'ignore',
    'UNKNOWN': 'ignore',
    'OTHER_MOVER': 'ignore',
    'MOPED':  'motorcycle',
    'STROLLER': 'ignore'
}



def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """  """
    q_scipy =  Rotation.from_matrix(R).as_quat()
    return quat_scipy2argo(q_scipy)


def quat_scipy2argo(q_scipy: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    x, y, z, w = q_scipy
    q_argo = np.array([w, x, y, z])
    return q_argo


def test_rotmat2quat():
    """ """
    R = np.eye(3)
    q = rotmat2quat(R)
    R_result = quat2rotmat(q)
    assert np.allclose(R, R_result)


def construct_argoverse_boxes_lidarfr(sweep_labels: List[Dict[str,Any]], lidar_SE3_egovehicle: SE3):
    """ Move egovehicle frame boxes, to live in the LiDAR frame instead"""
    # Make list of Box objects including coord system transforms.
    box_list = []
    for label in sweep_labels:

        width = label['width']
        length = label['length']
        height = label['height']

        x = label['center']['x']
        y = label['center']['y']
        z = label['center']['z']

        qw = label['rotation']['w']
        qx = label['rotation']['x']
        qy = label['rotation']['y']
        qz = label['rotation']['z']

        argoverse_classname = label['label_class']
        centerpoint_classname = argoverse_name_to_nuscenes_name[argoverse_classname]
        box = Box(
            center = [x, y, z], # Argoverse and nuScenes use scalar-first
            size = [width, length, height], # wlh
            orientation = Quaternion([qw, qx, qy, qz]),
            #label= , #: int = np.nan, # IGNORING SCORE FOR NOW
            #score= , #: float = np.nan, # IGNORING SCORE FOR NOW
            #velocity= , #: Tuple = (np.nan, np.nan, np.nan), IGNORING VELOCITY FOR NOW
            name = centerpoint_classname, #: str = None,
            token = label['track_label_uuid'], #: str = None):
        )
        
        # transform box from the egovehicle frame into the LiDAR frame

        box.center = lidar_SE3_egovehicle.transform_point_cloud(box.center.reshape(1,3)).squeeze()
        box.orientation = Quaternion(rotmat2quat(lidar_SE3_egovehicle.rotation @ quat2rotmat(list(box.orientation))))
        box.velocity = lidar_SE3_egovehicle.rotation @ box.velocity
        box_list.append(box)

    return box_list



def _fill_trainval_infos(split: str, root_path: str, nsweeps: int = 10, filter_zero: bool = True) -> List[Any]:
    """
    Our reference channel is "UP_LIDAR", similar to nuScenes' "LIDAR_TOP"
    channel from which we track back n sweeps to aggregate the point cloud.
    reference channel of the current sample_rec that the point clouds are mapped to.
    """
    split_argoverse_infos = []

    split_subdirs_map = {
        'train': ['train1', 'train2', 'train3', 'train4', 'train5'],
        'val': ['val'],
        'test': ['test']
    }
    split_subdirs = split_subdirs_map[split]

    for split_subdir in split_subdirs:

        # whether or not is test split
        is_test = split == 'test'

        split_root_path = f'{root_path}/{split_subdir}'
        if not Path(split_root_path).exists():
            print(f'Skipping {split_subdir}: {split_root_path} does not exist')
            continue
            
        dl = SimpleArgoverseTrackingDataLoader(data_dir=split_root_path, labels_dir=split_root_path)
        valid_log_ids = dl.sdb.get_valid_logs()
        # loop through all of the logs
        for log_id in valid_log_ids:

            # for each log, loop through all of the LiDAR sweeps
            log_ply_fpaths = dl.get_ordered_log_ply_fpaths(log_id)

            log_calib_data = dl.get_log_calibration_data(log_id)
            calibration_config = get_calibration_config(log_calib_data, camera_name='ring_front_center')
            ref_cam_intrinsic = calibration_config.intrinsic[:3,:3]

            egovehicle_SE3_lidar = SE3(
                rotation=quat2rotmat(log_calib_data['vehicle_SE3_up_lidar_']['rotation']['coefficients']),
                translation=np.array(log_calib_data['vehicle_SE3_up_lidar_']['translation'])
            )
            lidart0_SE3_egot0 = egovehicle_SE3_lidar.inverse()
            
            all_lidar_timestamps = [ int(Path(ply_fpath).stem.split('_')[-1]) for ply_fpath in log_ply_fpaths]
            has_valid_pose = [dl.get_city_SE3_egovehicle(log_id, ts) is not None for ts in all_lidar_timestamps]
            valid_idxs = np.where( np.array(has_valid_pose) == True)[0]
            first_valid_idx =  np.min(valid_idxs)
            log_ply_fpaths = log_ply_fpaths[first_valid_idx:]

            num_log_sweeps = len(log_ply_fpaths)

            for sample_idx, sample_ply_fpath in enumerate(log_ply_fpaths):
                if sample_idx % 100 == 0:
                    print(f'\t{log_id}: On {sample_idx}/{num_log_sweeps}')

                sample_lidar_timestamp = int(Path(sample_ply_fpath).stem.split('_')[-1])

                city_SE3_egot0 = dl.get_city_SE3_egovehicle(log_id, sample_lidar_timestamp)
                if city_SE3_egot0 is None:
                    print(f'Missing pose for {sample_idx}/{num_log_sweeps}')
                    continue
                egot0_SE3_city = city_SE3_egot0.inverse()

                sweep_labels = dl.get_labels_at_lidar_timestamp(log_id, sample_lidar_timestamp)

                # Argoverse timestamps are in nanoseconds, convert to seconds
                ref_time_seconds = sample_lidar_timestamp * 1e-9
                ref_time = ref_time_seconds
                ref_lidar_path = sample_ply_fpath

                if not is_test:
                    ref_boxes = construct_argoverse_boxes_lidarfr(sweep_labels, lidart0_SE3_egot0)

                ref_cam_path = dl.get_closest_im_fpath(
                    log_id,
                    camera_name='ring_front_center',
                    lidar_timestamp=sample_lidar_timestamp
                )

                info = {
                    "transform_matrix": lidart0_SE3_egot0.transform_matrix,
                    "lidar_path": f'{split_subdir}/{log_id}/lidar/{Path(sample_ply_fpath).name}',
                    "cam_front_path": ref_cam_path,
                    "cam_intrinsic": ref_cam_intrinsic,
                    "token": f'{log_id}/lidar/PC_{sample_lidar_timestamp}.ply',
                    "sweeps": [],
                    "ref_from_car": lidart0_SE3_egot0.transform_matrix,
                    "car_from_global": egot0_SE3_city.transform_matrix,
                    "timestamp": ref_time,
                }

                sweeps = []
                # should be 9 sweeps for each 1 sample if nsweeps = 10

                sweep_idxs = np.arange(sample_idx - nsweeps + 1, sample_idx)

                # if there are no samples before, just pad with the same sample
                sweep_idxs = np.maximum(sweep_idxs, np.zeros(nsweeps-1, dtype=np.int32) )
                print('Sweep comprised of ', sweep_idxs, f' at sample={sample_idx}')
                
                info["sample"] = {
                    "lidar_path": f'{split_subdir}/{log_id}/lidar/{Path(sample_ply_fpath).name}',
                    "transform_matrix": lidart0_SE3_egot0.transform_matrix,
                    "time_lag" : 0
                }

                for sweep_idx in sweep_idxs:

                    sweep_ply_fpath = log_ply_fpaths[sweep_idx]
                    sweep_lidar_timestamp = int(Path(sweep_ply_fpath).stem.split('_')[-1])
                    
                    # Argoverse timestamps are in seconds, must convert!
                    curr_time_seconds = sweep_lidar_timestamp * 1e-9
                    time_lag = ref_time_seconds - curr_time_seconds
                    if sweep_idx == sample_idx:
                        assert time_lag == 0

                    city_SE3_egoti = dl.get_city_SE3_egovehicle(log_id, sweep_lidar_timestamp)
                    egoti_SE3_lidarti = egovehicle_SE3_lidar # calibration is fixed!

                    lidart0_SE3_egoti = lidart0_SE3_egot0.compose(egot0_SE3_city).compose(city_SE3_egoti)
                    
                    sweep = {
                        "lidar_path": f'{split_subdir}/{log_id}/lidar/{Path(sweep_ply_fpath).name}',
                        "sample_data_token": f'{log_id}/lidar/PC_{sweep_lidar_timestamp}.ply',
                        "transform_matrix": lidart0_SE3_egoti.transform_matrix,
                        "global_from_car": city_SE3_egoti.transform_matrix,
                        "car_from_current": egoti_SE3_lidarti.transform_matrix,
                        "time_lag": time_lag,
                    }
                    sweeps.append(sweep)

                    # sweep = {
                    #     "lidar_path": ref_lidar_path,
                    #     "sample_data_token": lidar_timestamp,
                    #     "transform_matrix": None,
                    #     "time_lag": 0
                    # }

                info["sweeps"] = sweeps

                assert (
                    len(info["sweeps"]) == nsweeps - 1
                ), f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, you should duplicate to sweep num {nsweeps-1}"

                # save the annotations if we are looking at train/val log
                if not is_test:

                    num_gt_boxes = len(ref_boxes)
                    mask = np.ones(num_gt_boxes, dtype=bool).reshape(-1) # assume all are visible

                    # form N x 3 arrays for 3d location, dim, velocity info
                    locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
                    dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)
                    velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)

                    rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
                    names = np.array([b.name for b in ref_boxes])
                    tokens = np.array([b.token for b in ref_boxes])
                    gt_boxes = np.concatenate(
                        [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
                    )

                    assert len(gt_boxes) == len(velocity)

                    if not filter_zero:
                        info["gt_boxes"] = gt_boxes
                        info["gt_boxes_velocity"] = velocity
                        info["gt_names"] = np.array(names) # already ran `general_to_detection` conversion previously
                        info["gt_boxes_token"] = tokens
                    else:
                        info["gt_boxes"] = gt_boxes[mask, :]
                        info["gt_boxes_velocity"] = velocity[mask, :]
                        info["gt_names"] = np.array(names)[mask] # already ran `general_to_detection` conversion previously
                        info["gt_boxes_token"] = tokens[mask]

                split_argoverse_infos.append(info)

    return split_argoverse_infos


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def quaternion_yaw_scipy(q: Quaternion) -> float:
    """ """
    q_argo = q
    q_scipy = quat_argo2scipy(q_argo)
    yaw, _, _ = Rotation.from_quat(q_scipy).as_euler('zyx')
    return yaw


def create_argoverse_infos(
    pkl_save_dirpath: str, 
    root_path: str,
    nsweeps: int = 10,
    filter_zero: bool = True,
):
    """ """
    for split in ['val', 'test', 'train']:
        print(f'Preparing split {split}')
        split_argoverse_infos = _fill_trainval_infos(split, root_path, nsweeps=nsweeps, filter_zero=filter_zero)
        
        if len(split_argoverse_infos) == 0:
            print(f'Nothing populated for {split}, skipping')
            continue

        if split in ['train', 'val']:
            pkl_fpath = f'{pkl_save_dirpath}/infos_{split}_{nsweeps:02d}sweeps_withvelo_filter_{filter_zero}.pkl'
        else:
            pkl_fpath = f'{pkl_save_dirpath}/infos_{split}_{nsweeps:02d}sweeps_withvelo.pkl'
        
        print(f"{split} sample: {len(split_argoverse_infos)}")
        save_pkl_dictionary(pkl_fpath, split_argoverse_infos)



if __name__ == "__main__":
    """
    Example usage:
        ARGOVERSE_DATASET_ROOT = "data/argoverse"
        ARGOVERSE_DATASET_ROOT = '/home/ubuntu/argoverse/argoverse-tracking'
        ARGOVERSE_DATASET_ROOT = '/srv/share/cliu324/argoverse-tracking-readonly'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--argoverse_dataset_root',
        type=str,
        required=True,
        help='where logs live on disk'
    )
    parser.add_argument(
        '--pkl_save_dirpath',
        type=str,
        required=True,
        help='where to save populated pkl files with sweep metadata'
    )
    args = parser.parse_args()
    create_argoverse_infos(
        pkl_save_dirpath=args.pkl_save_dirpath,
        root_path=args.argoverse_dataset_root,
        nsweeps=5 # not using 10 for Argoverse
    )
