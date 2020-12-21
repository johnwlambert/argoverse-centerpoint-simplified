

import copy
import os.path as osp
import pdb
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat, quat_argo2scipy
from tqdm import tqdm
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

try:
    from nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
except:
    print("nuScenes devkit not Found!")


from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import get_calibration_config

"""
{
'token': '2a1710d55ac747339eae4502565b956b',
'timestamp': 1542799673198229,
'prev': 'b6cf1af09fc54c79b0d1402e346b8514',
'next': '184274eb68f24cb9bdb06f90d5728413',
'scene_token': '0dae482684ce4cd69a7258f55bc98d73',
'data': {
    'RADAR_FRONT': 'ef276fd2117d49c8baae43909579bec8',
    'RADAR_FRONT_LEFT': '010081c456d543379dc2fd62fc47ced8',
    'RADAR_FRONT_RIGHT': 'bea3a1f01d63421e8be89019b66fd75d',
    'RADAR_BACK_LEFT': '302918a3fba3472dacec20691b13b090',
    'RADAR_BACK_RIGHT': '8214182bfd304d6da8372611baaa4946',
    'LIDAR_TOP': 'ccccfc743caf4039aa2693bfc364c822',
    'CAM_FRONT': '01c5b74bfdb84508ba4252a0b451a96d',
    'CAM_FRONT_RIGHT': '802acc61bbc941749cf19799c4581896',
    'CAM_BACK_RIGHT': '2878dae393fd4de4b2d593e2c34902a9',
    'CAM_BACK': '8979ae5848c84f75bce4e9410998e7d6',
    'CAM_BACK_LEFT': '508eaea1e8a8475eae94fa49bfa11f88',
    'CAM_FRONT_LEFT': '64c634b2012143beaebaa24c6922e0fb'
    }, 'anns': [
        '345d75498f214fc38c87104491fc632a',
        'dbaa05ce77ff4682aaef46a767f6e36b',
        '390b5dd89f2d43dd83dce93e554cc788',
        '57e4dc42ef6b4f9ba042f194f22d7fc0'
    ]},
"""

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
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
    'ANIMAL': 'ignore',
    'UNKNOWN': 'ignore'
}


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



def _fill_trainval_infos(root_path: str, nsweeps: int = 10, filter_zero: bool = True):
    """ """
    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    pdb.set_trace()

    for split in ['train1', 'train2', 'train3', 'train4', 'train5', 'val', 'test']:

        # whether or not is test split
        test = split == 'test'

        split_root_path = f'{root_path}/{split}'
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

            num_log_sweeps = len(log_ply_fpaths)
            for sample_idx, ply_fpath in enumerate(log_ply_fpaths):
                print(f'On {sample_idx}/{num_log_sweeps}')

                is_first_log_sweep = sample_idx == 0
                is_last_log_sweep = sample_idx == num_log_sweeps - 1

                lidar_timestamp = int(Path(ply_fpath).stem.split('_')[-1])

                city_SE3_egot0 = dl.get_city_SE3_egovehicle(log_id, lidar_timestamp)
                egot0_SE3_city = city_SE3_egot0.inverse()

                sweep_labels = dl.get_labels_at_lidar_timestamp(log_id, lidar_timestamp)

                # nuscenes timestamps are in microseconds
                ref_time_microseconds = lidar_timestamp * 1e-6
                ref_time = ref_time_microseconds
                ref_lidar_path = ply_fpath

                ref_boxes = construct_argoverse_boxes_lidarfr(sweep_labels, lidart0_SE3_egot0)
                pdb.set_trace()

                ref_cam_path = dl.get_closest_im_fpath(
                    log_id,
                    camera_name='ring_front_center',
                    lidar_timestamp=lidar_timestamp
                )

                info = {
                    "lidar_path": ref_lidar_path,
                    "cam_front_path": ref_cam_path,
                    "cam_intrinsic": ref_cam_intrinsic,
                    "token": lidar_timestamp,
                    "sweeps": [],
                    "ref_from_car": lidart0_SE3_egot0.transform_matrix,
                    "car_from_global": egot0_SE3_city.transform_matrix,
                    "timestamp": ref_time,
                }

                sweeps = []
                # should be 9 sweeps for each 1 sample

                while len(sweeps) < nsweeps - 1:
                    

                    if is_first_log_sweep or is_last_log_sweep:
                        # if there are no samples before, just pad with the same sample
                        if len(sweeps) == 0:
                            sweep = {
                                "lidar_path": ref_lidar_path,
                                "sample_data_token": lidar_timestamp,
                                "transform_matrix": None,
                                "time_lag": 0
                            }
                            sweeps.append(sweep)
                        else:
                            sweeps.append(sweeps[-1])
                    else:
                        pdb.set_trace()
                        # accumulate the 9 prior sweeps
                        curr_sd_rec = nusc.get("sample_data", curr_sd_rec["prev"])

                        # Get past pose
                        current_pose_rec = nusc.get("ego_pose", curr_sd_rec["ego_pose_token"])

                        city_SE3_egoti = SE3(
                            rotation=quat2rotmat(current_pose_rec["rotation"]),
                            translation=np.array(current_pose_rec["translation"])
                        )

                        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                        current_cs_rec = nusc.get(
                            "calibrated_sensor", curr_sd_rec["calibrated_sensor_token"]
                        )
                        egoti_SE3_lidarti = SE3(
                            rotation=quat2rotmat(current_cs_rec["rotation"]),
                            translation=np.array(current_cs_rec["translation"])
                        )

                        lidart0_SE3_lidarti = lidart0_SE3_egot0.compose(egot0_SE3_city).compose(city_SE3_egoti).compose(egoti_SE3_lidarti)
                        lidar_path = nusc.get_sample_data_path(curr_sd_rec["token"])

                        time_lag = ref_time - 1e-6 * curr_sd_rec["timestamp"]

                        if 'n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470898274' in lidar_path:
                            # trainval example "transform_matrix"
                            expected_tm = np.array(
                                [
                                    [1, 0, 0, 0],
                                    [0, 1, 0, 0.001],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]
                                ])
                            assert np.allclose(expected_tm, lidart0_SE3_lidarti.transform_matrix, atol=1e-2)


                        sweep = {
                            "lidar_path": lidar_path,
                            "sample_data_token": curr_sd_rec["token"],
                            "transform_matrix": lidart0_SE3_lidarti.transform_matrix,
                            "global_from_car": city_SE3_egoti.transform_matrix,
                            "car_from_current": egoti_SE3_lidarti.transform_matrix,
                            "time_lag": time_lag,
                        }
                        sweeps.append(sweep)

                info["sweeps"] = sweeps

                assert (
                    len(info["sweeps"]) == nsweeps - 1
                ), f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, you should duplicate to sweep num {nsweeps-1}"

                # save the annotations if we are looking at train/val log
                if not test:
                    pdb.set_trace()

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
                        info["gt_names"] = np.array([names]) # already ran `general_to_detection` conversion previously
                        info["gt_boxes_token"] = tokens
                    else:
                        info["gt_boxes"] = gt_boxes[mask, :]
                        info["gt_boxes_velocity"] = velocity[mask, :]
                        info["gt_names"] = np.array([names])[mask] # already ran `general_to_detection` conversion previously
                        info["gt_boxes_token"] = tokens[mask]

                if 'train' in split:
                    train_nusc_infos.append(info)
                else:
                    val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


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
    root_path: str,
    nsweeps: int = 10,
    filter_zero: bool = True,
):
    """ """
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(root_path, nsweeps=nsweeps, filter_zero=filter_zero
    )

    root_path = "data/argoverse"

    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        with open(
            root_path / "infos_test_{:02d}sweeps_withvelo.pkl".format(nsweeps), "wb"
        ) as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        with open(
            root_path
            / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(
                nsweeps, filter_zero
            ),
            "wb",
        ) as f:
            pickle.dump(train_nusc_infos, f)
        with open(
            root_path
            / "infos_val_{:02d}sweeps_withvelo_filter_{}.pkl".format(
                nsweeps, filter_zero
            ),
            "wb",
        ) as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == "__main__":
    """ """
    #ARGOVERSE_DATASET_ROOT = "data/argoverse"
    ARGOVERSE_DATASET_ROOT = '/srv/share/cliu324/argoverse-tracking-readonly'
    create_argoverse_infos(
        root_path=ARGOVERSE_DATASET_ROOT, nsweeps=5 # not using 10 for Argoverse
    )

