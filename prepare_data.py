

import copy
import os.path as osp
import pdb
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from functools import reduce

import numpy as np
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat, quat_argo2scipy
from tqdm import tqdm
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")


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


def transform_matrix(
    translation: np.ndarray = np.array([0, 0, 0]),
    rotation: Quaternion = Quaternion([1, 0, 0, 0]),
    inverse: bool = False,
) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm



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


def transform_city_box_to_lidar_frame_old(box: Box, pose_record: Dict[str,Any], cs_record: Dict[str,Any]) -> Box:
    """
    """
    # Move box to ego vehicle coord system
    box.translate(-np.array(pose_record["translation"]))
    box.rotate(Quaternion(pose_record["rotation"]).inverse)

    #  Move box to sensor coord system
    box.translate(-np.array(cs_record["translation"]))
    box.rotate(Quaternion(cs_record["rotation"]).inverse)

    return box


def transform_city_box_to_lidar_frame(box: Box, pose_record: Dict[str,Any], cs_record: Dict[str,Any]) -> Box:
    """
    Args:
        box in city frame
    
    Returns:
        box in egovehicle frame
    """
    city_SE3_egovehicle = SE3(
        rotation=quat2rotmat(pose_record["rotation"]),
        translation=np.array(pose_record["translation"])
    )
    egovehicle_SE3_city = city_SE3_egovehicle.inverse()

    egovehicle_SE3_lidar = SE3(
        rotation=quat2rotmat(cs_record["rotation"]),
        translation=np.array(cs_record["translation"])
    )
    lidar_SE3_egovehicle = egovehicle_SE3_lidar.inverse()

    box.center = egovehicle_SE3_city.transform_point_cloud(box.center.reshape(1,3)).squeeze()
    box.orientation = Quaternion(rotmat2quat(egovehicle_SE3_city.rotation @ quat2rotmat(list(box.orientation))))
    box.velocity = egovehicle_SE3_city.rotation @ box.velocity

    # should become
    # box.center = array([-3.40565985, 15.44737312,  0.73430395])
    # box.orientation = Quaternion(-0.5881997166174422, -0.008993043052939156, -0.006952766600972482, -0.8086358127021093)

    box.center = lidar_SE3_egovehicle.transform_point_cloud(box.center.reshape(1,3)).squeeze()
    box.orientation = Quaternion(rotmat2quat(lidar_SE3_egovehicle.rotation @ quat2rotmat(list(box.orientation))))
    box.velocity = lidar_SE3_egovehicle.rotation @ box.velocity

    return box


def test_transform_city_box_to_lidar_frame() -> None:
    """ """
    # boxes[1]

    pose_record = {
        'token': '3388933b59444c5db71fade0bbfef470',
        'timestamp': 1531883530449377,
        'rotation': [-0.7495886280607293, -0.0077695335695504636, 0.00829759813869316, -0.6618063711504101],
        'translation': [1010.1328353833223, 610.8111652918716, 0.0]
    }
    cs_record = {
        'token': '7a0cd258d096410eb68251b4b87febf5',
        'sensor_token': 'dc8b396651c05aedbb9cdaae573bb567',
        'translation': [0.943713, 0.0, 1.84023],
        'rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817],
        'camera_intrinsic': []
    }

    city_box = Box(
        center = [9.94381e+02, 6.09330e+02, 6.67000e-01],
        orientation = Quaternion(-0.09426469466835254, 0.0, 0.0, 0.9955471698212407),
        size = [0.315, 0.338, 0.712]
    )

    lidar_box_gt = Box(
        center = [-15.44939123,  -4.28768163,  -1.30136452],
        orientation = Quaternion(0.15480463394047833, 0.0033357299433613465, 0.00023897082747060053, -0.9879394420252907),
        size = [0.315, 0.338, 0.712]
    )

    lidar_box1 = transform_city_box_to_lidar_frame_old( copy.deepcopy(city_box), copy.deepcopy(pose_record), copy.deepcopy(cs_record) )
    assert lidar_box1 == lidar_box_gt

    lidar_box2 = transform_city_box_to_lidar_frame( copy.deepcopy(city_box), copy.deepcopy(pose_record), copy.deepcopy(cs_record) )
    assert np.allclose(lidar_box2.center, lidar_box_gt.center)
    assert np.allclose(quat2rotmat(list(lidar_box2.orientation)), quat2rotmat(list(lidar_box_gt.orientation)))



def get_sample_data(nusc, sample_data_token: str, selected_anntokens: List[str] = None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.

    Args:
        nusc
        sample_data_token: token for "sample" LIDAR_TOP measurement
        selected_anntokens: If provided only return the selected annotation.

    Returns:
        data_path: path to LIDAR_TOP file
        boxes
        camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get(table_name="sample_data", token=sample_data_token)
    cs_record = nusc.get(table_name="calibrated_sensor", token=sd_record["calibrated_sensor_token"])
    sensor_record = nusc.get(table_name="sensor", token=cs_record["sensor_token"])
    pose_record = nusc.get(table_name="ego_pose", token=sd_record["ego_pose_token"])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record["modality"] == "camera":
        cam_intrinsic = np.array(cs_record["camera_intrinsic"])
        imsize = (sd_record["width"], sd_record["height"])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:

        lidar_box = transform_city_box_to_lidar_frame(box, pose_record, cs_record)
        box_list.append(lidar_box)

    return data_path, box_list, cam_intrinsic


def _fill_trainval_infos(
    nusc, train_scenes, val_scenes, test: bool = False, nsweeps: int = 10, filter_zero: bool = True
):
    """ """
    from nuscenes.utils.geometry_utils import transform_matrix

    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for sample_idx, sample in enumerate(nusc.sample):
        print(f'On {sample_idx}/{len(nusc.sample)}')
        """ Manual save info["sweeps"] """
        # Get reference pose and timestamp

        """
        sample is a dictionary with keys
            'token',
            'timestamp',
            'prev',
            'next',
            'scene_token',
            'anns',
            'data',

        'data' is itself a dictionary with keys
            'RADAR_FRONT',
            'RADAR_FRONT_LEFT',
            'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT',

            'LIDAR_TOP',

            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT'
        """
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = nusc.get(table_name="sample_data", token=ref_sd_token)
        ref_cs_rec = nusc.get(
            table_name="calibrated_sensor", token=ref_sd_rec["calibrated_sensor_token"]
        )
        ref_pose_rec = nusc.get(table_name="ego_pose", token=ref_sd_rec["ego_pose_token"])
        ref_time = 1e-6 * ref_sd_rec["timestamp"]

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample["data"]["CAM_FRONT"]
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        egot0_SE3_lidart0 = SE3(
            rotation=quat2rotmat(ref_cs_rec["rotation"]),
            translation=np.array(ref_cs_rec["translation"])
        )
        lidart0_SE3_egot0 = egot0_SE3_lidart0.inverse()

        # Homogeneous transformation matrix from global to _current_ ego car frame
        city_SE3_egot0 = SE3(
            rotation=quat2rotmat(ref_pose_rec["rotation"]),
            translation=np.array(ref_pose_rec["translation"])
        )
        egot0_SE3_city = city_SE3_egot0.inverse()

        info = {
            "lidar_path": ref_lidar_path,
            "cam_front_path": ref_cam_path,
            "cam_intrinsic": ref_cam_intrinsic,
            "token": sample["token"],
            "sweeps": [],
            "ref_from_car": lidart0_SE3_egot0.transform_matrix,
            "car_from_global": egot0_SE3_city.transform_matrix,
            "timestamp": ref_time,
        }

        sample_data_token = sample["data"][chan]
        curr_sd_rec = nusc.get("sample_data", sample_data_token)
        sweeps = []

        # should be 9 samples for each 1 sweep

        while len(sweeps) < nsweeps - 1:
            # if there are no samples before, just pad with the same sample
            if curr_sd_rec["prev"] == "":
                if len(sweeps) == 0:
                    sweep = {
                        "lidar_path": ref_lidar_path,
                        "sample_data_token": curr_sd_rec["token"],
                        "transform_matrix": None,
                        "time_lag": curr_sd_rec["timestamp"] * 0,
                        # time_lag: 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
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
            annotations = [
                nusc.get("sample_annotation", token) for token in sample["anns"]
            ]

            mask = np.array(
                [
                    (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                    for anno in annotations
                ],
                dtype=bool,
            ).reshape(-1)

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

            assert len(annotations) == len(gt_boxes) == len(velocity)

            if not filter_zero:
                info["gt_boxes"] = gt_boxes
                info["gt_boxes_velocity"] = velocity
                info["gt_names"] = np.array(
                    [general_to_detection[name] for name in names]
                )
                info["gt_boxes_token"] = tokens
            else:
                info["gt_boxes"] = gt_boxes[mask, :]
                info["gt_boxes_velocity"] = velocity[mask, :]
                info["gt_names"] = np.array(
                    [general_to_detection[name] for name in names]
                )[mask]
                info["gt_boxes_token"] = tokens[mask]

        if sample["scene_token"] in train_scenes:
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


def test_quaternion_yaw_scipy():
    """ """
    q = Quaternion(-0.10293980572965565, -0.003318673306337732, -0.00041304817784475515, 0.994681965351252)

    old_yaw = quaternion_yaw(copy.deepcopy(q))
    scipy_yaw = quaternion_yaw_scipy(copy.deepcopy(q))

    yaw_gt = -2.9353469986645324
    assert np.isclose(old_yaw, yaw_gt)
    assert np.isclose(scipy_yaw, yaw_gt)



def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec["next"] == "":
                sd_rec = nusc.get("sample_data", sd_rec["next"])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def create_nuscenes_infos(
    root_path: str,
    version: str = "v1.0-trainval",
    nsweeps: int = 10,
    filter_zero: bool = True,
):
    """ """
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)

    pdb.set_trace()
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)

    """
    Each scene has keys like:
        'token',
        'log_token',
        'nbr_samples', e.g. 40
        'first_sample_token',
        'last_sample_token',
        'name', e.g. 'scene-0001'
        'description', e.g. 'Construction, maneuver between several trucks'
    """
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, nsweeps=nsweeps, filter_zero=filter_zero
    )

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
    NUSCENES_TRAINVAL_DATASET_ROOT = "data/nuScenes"
    #version = "v1.0-trainval"
    version = "v1.0-test"
    create_nuscenes_infos(
        root_path=NUSCENES_TRAINVAL_DATASET_ROOT, version=version, nsweeps=10
    )
    # test_rotmat2quat()
    # test_transform_city_box_to_lidar_frame()
    # test_quaternion_yaw_scipy()


