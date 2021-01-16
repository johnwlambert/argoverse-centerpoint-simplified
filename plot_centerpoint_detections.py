
import argparse
import glob
import os
import pdb
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from argoverse.utils.json_utils import read_json_file, save_json_dict
from argoverse.utils.pkl_utils import load_pkl_dictionary
from argoverse.utils.se3 import SE3
from argoverse.utils.transform import quat2rotmat

from centerpoint.utils.loading import read_file, load_ply_xyzir

nuscenes_class_names = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone'
]

def get_argo_label(label: str) -> str:
    """Map the nuscenes labels to argoverse labels"""

    if 'car' in label:
        return 'VEHICLE'

    if 'truck' in label:
        return 'VEHICLE' # 'LARGE_VEHICLE'

    if 'construction_vehicle' in label:
        return 'LARGE_VEHICLE'

    if 'bus' in label:
        return 'BUS'

    if 'trailer' in label:
        return 'TRAILER'

    if 'barrier' in label:
        return 'ON_ROAD_OBSTACLE'

    if 'motorcycle' in label:
        return 'MOTORCYCLE'

    if 'bicycle' in label:
        return 'BICYCLE'

    if 'pedestrian' in label:
        return 'PEDESTRIAN'

    if 'traffic_cone' in label:
        return 'ON_ROAD_OBSTACLE'


def get_box_corners(box, wlh_factor: float = 1.0) -> np.ndarray:
    """Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
    The last four are the ones facing backwards.
    """
    w, l, h = box.wlh * wlh_factor

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(box.orientation.rotation_matrix, corners)

    # Translate
    x, y, z = box.center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def render_nuscenes_box(
    box,
    axis: Axes,
    view: np.ndarray = np.eye(3),
    normalize: bool = False,
    colors: Tuple = ("b", "r", "k"),
    linewidth: float = 2,
) -> None:
    """
    Renders the box in the provided Matplotlib axis.
    :param axis: Axis onto which the box should be drawn.
    :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
        back and sides.
    :param linewidth: Width in pixel of the box sides.
    """
    corners = view_points(get_box_corners(box), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot(
                [prev[0], corner[0]],
                [prev[1], corner[1]],
                color=color,
                linewidth=linewidth,
            )
            prev = corner

    # Draw the sides
    for i in range(4):
        axis.plot(
            [corners.T[i][0], corners.T[i + 4][0]],
            [corners.T[i][1], corners.T[i + 4][1]],
            color=colors[2],
            linewidth=linewidth,
        )

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    axis.plot(
        [center_bottom[0], center_bottom_forward[0]],
        [center_bottom[1], center_bottom_forward[1]],
        color=colors[0],
        linewidth=linewidth,
    )
    axis.text(center_bottom[0], center_bottom[1], f'{nuscenes_class_names[box.label]}', color='r')


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


class Box:
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(
        self,
        center: List[float],
        size: List[float],
        orientation: Quaternion,
        label: int = np.nan,
        score: float = np.nan,
        velocity: Tuple = (np.nan, np.nan, np.nan),
        name: str = None,
        token: str = None,
    ):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        # print(center.shape)
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token


def _second_det_to_nusc_box(detection):
    """ Boxes have shape
    
    locs
    dims
    velocity_x
    velocity_y
    yaw
    """
    box3d = detection["box3d_lidar"]
    scores = detection["scores"]
    labels = detection["label_preds"]
    # undo the phase shift from the original data file creation
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            list(box3d[i, :3]),
            list(box3d[i, 3:6]),
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list




def visual(points, gt_anno, det, i, eval_range=100, conf_th=0.5):
    """ """
    token = det['metadata']['token'].replace('/', '_')
    
    _, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
    #points = remove_close(points, radius=3)
    #points = view_points(points[:3, :], np.eye(4), normalize=False)

    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    if gt_anno is not None:
        boxes_gt = _second_det_to_nusc_box(gt_anno)
    boxes_est = _second_det_to_nusc_box(det)

    if gt_anno is not None:
        # Show GT boxes.
        for box in boxes_gt:
            render_nuscenes_box(box, ax, view=np.eye(4), colors=("r", "r", "r"), linewidth=2)

    # Show EST boxes.
    for box in boxes_est:
        if box.score >= conf_th:
            render_nuscenes_box(
                box, ax, view=np.eye(4), colors=("b", "b", "b"), linewidth=1
            )

    axes_limit = (
        eval_range + 3
    )  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    #plt.axis("off")

    plt.show()
    # plt.savefig(f"demo/{token}_file%02d.png" % i)
    # plt.close('all')

def read_file(path, tries=2, num_point_feature=4):
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            points = np.fromfile(path, dtype=np.float32)
            s = points.shape[0]
            if s % 5 != 0:
                points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None

    return points

    
def visualize_argoverse_detections(args):
    """ """
    pkl_data = load_pkl_dictionary(args.pkl_fpath)
    split = args.split

    sweep_idx = 0
    num_sweeps = len(pkl_data.keys())
    for token, sweep_output in pkl_data.items():
        print(f'On {sweep_idx}/{num_sweeps}')
        sweep_idx += 1
        print(f'\t{token}')
        lidar_subpath = sweep_output['metadata']["token"]
        log_id = lidar_subpath.split('/')[0]
        lidar_fpath = f'{args.argoverse_root}/{lidar_subpath}'
        points = load_ply_xyzir(lidar_fpath)[:,:3]
        calibration_fpath = f'{args.argoverse_root}/{log_id}/vehicle_calibration_info.json'
        calib_data = read_json_file(calibration_fpath)
        egovehicle_SE3_lidar = SE3(
            rotation=quat2rotmat(calib_data["vehicle_SE3_up_lidar_"]["rotation"]["coefficients"]),
            translation=np.array(calib_data["vehicle_SE3_up_lidar_"]["translation"])
        )
        lidar_SE3_egovehicle = egovehicle_SE3_lidar.inverse()
        points = lidar_SE3_egovehicle.transform_point_cloud(points)
        
        if split != 'test':
            gt_anno = sweep_output["annos"]
            num_boxes = len(gt_anno[0]['names'])
            gt_obj_classnames = gt_anno[0]['names']
            gt_obj_classnames = [ 'barrier' if name in ['vehicle','ignore'] else name for name in gt_obj_classnames]
            annos = {
                'box3d_lidar': gt_anno[0]['boxes'],
                'scores': np.ones(num_boxes),
                'label_preds': [ nuscenes_class_names.index(name) for name in gt_obj_classnames]
            }

        #visual(points.T, gt_anno=annos, det=pkl_data[token], i=0, eval_range=50, conf_th=0.5)
        convert_dets_to_argoverse_format(pkl_data[token], egovehicle_SE3_lidar, args.output_dataroot, conf_th=0.5)


def rotmat2quat(R: np.ndarray) -> np.ndarray:
    """  """
    q_scipy =  Rotation.from_matrix(R).as_quat()
    return quat_scipy2argo(q_scipy)


def quat_scipy2argo(q_scipy: np.ndarray) -> np.ndarray:
    """Re-order Argoverse's scalar-first [w,x,y,z] quaternion order to Scipy's scalar-last [x,y,z,w]"""
    x, y, z, w = q_scipy
    q_argo = np.array([w, x, y, z])
    return q_argo


def convert_dets_to_argoverse_format(
    dets_dict,
    egovehicle_SE3_lidar: SE3,
    output_dataroot: str,
    conf_th: float
):
    """ """
    token = dets_dict['metadata']['token']

    log_id = token.split('/')[0]
    lidar_fname = Path(token.split('/')[-1]).stem
    lidar_timestamp = int(lidar_fname.split('_')[-1])

    log_dets_dir = os.path.join(output_dataroot, log_id, f"per_sweep_annotations_amodal")
    if not os.path.exists(log_dets_dir):
        os.makedirs(log_dets_dir)

    # just for this sweep
    tracked_labels = []

    boxes_est = _second_det_to_nusc_box(dets_dict)
    for box in boxes_est:

        # move from lidar frame to egovehicle frame
        width, length, height = box.wlh.astype(np.float64)

        nuscenes_classname = nuscenes_class_names[box.label]

        lidar_SE3_object = SE3(
            rotation=quat2rotmat(list(box.orientation)),
            translation=box.center
        )

        egovehicle_SE3_object = egovehicle_SE3_lidar.compose(lidar_SE3_object)

        x, y, z = egovehicle_SE3_object.translation.astype(np.float64)
        qw, qx, qy, qz = rotmat2quat(egovehicle_SE3_object.rotation)
        
        # if box.score < 0.5:
        #     continue

        label = {
            "center": {"x": x, "y": y, "z": z},
            "rotation": {"x": qx , "y": qy, "z": qz, "w": qw},
            "length": length,
            "width": width,
            "height": height,
            "track_label_uuid": "", # none, for now
            "timestamp": lidar_timestamp,
            "label_class": get_argo_label(nuscenes_classname),
            "score": float(box.score)
        }

        #print(label['label_class'], label['center'])
        tracked_labels.append(label)
    
    json_fpath = os.path.join(log_dets_dir, f"tracked_object_labels_{lidar_timestamp}.json")
    save_json_dict(json_fpath, tracked_labels)


def visualize_nuscenes_detections(args):
    """ """
    pkl_data = load_pkl_dictionary(args.pkl_fpath)
    
    for token, sweep_output in pkl_data.items():
        print(token)
        pdb.set_trace()
        lidar_path = sweep_output['metadata']['lidar_fpath']
        points = read_file(lidar_path)
        
        visual(points.T, gt_anno=None, det=pkl_data[token], i=0, eval_range=50, conf_th=0.5)
    


if __name__ == "__main__":
    """ Example usage:
    
    nuScenes:
        pkl_fpath = "/home/ubuntu/argoverse-centerpoint-simplified/work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/prediction.pkl"
        pkl_fpath = "/Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_prediction.pkl"
    
    argoverse:
        pkl_fpath = "/Users/jlambert/Downloads/prediction.pkl"
        pkl_fpath = "/home/ubuntu/argoverse-centerpoint-simplified/work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/argoverse_val_correct_prediction.pkl"
        pkl_fpath = "/home/ubuntu/argoverse-centerpoint-simplified/work_dirs/nusc_centerpoint_voxelnet_dcn_0075voxel_flip_testset/2020-12-23-argoverse_test_prediction.pkl"
        prediction.pkl"
        pkl_fpath = '/Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_val_correct_prediction.pkl'
    
        argoverse_root = "/home/ubuntu/argoverse/argoverse-tracking/test"
        argoverse_root = "/home/ubuntu/argoverse/argoverse-tracking/val"
        argoverse_root = "/Users/jlambert/Downloads/argoverse-tracking/val"
        
        output_dataroot = "argoverse-test-predictions-corrected-egovehicle-frame-2020-12-23"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices = ['val', 'test'],
        help="which split of the dataset predictions pertain to; if val is specified, ground truth will be loaded",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="which dataset the data came from, e.g. `argoverse`, `nuScenes`",
    )
    parser.add_argument(
        '--output_dataroot',
        type=str,
        required=True,
        help="path to directory where the detections will be saved in the Argoverse data format"
    )
    parser.add_argument(
        '--pkl_fpath',
        type=str,
        required=True,
        help='path to pickle file where Centerpoint predictions are saved'
    )
    parser.add_argument(
        '--argoverse_root',
        type=str,
        help='dataroot of logs with argoverse raw data, not required for nuScenes of course'
    )
    args = parser.parse_args()
    
    if 'argoverse' in args.dataset_name:
        visualize_argoverse_detections(args)
    elif args.dataset_name == 'nuScenes':
        visualize_nuscenes_detections(args)
