
from typing import Any, Dict, Tuple

import numpy as np
from centerpoint.utils.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=20000):
        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            self._max_voxels,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size



class Voxelization(object):
    def __init__(self, **kwargs):
        """
        voxel_generator.grid_size will be populated with an array
            of shape (3,) e.g. [1440, 1440,   40]

            if 54 meter range, and voxel size 0.075, then
                54 / 0.075 = 720, becomes grid of size 1440 x 1440

        Args:
            cfg: object with attributes as follows:

                cfg.voxel_size should be a float array of shape (3,)
                    e.g. [0.075, 0.075, 0.2 ]
                cfg.point_cloud_range should be a float array of shape (6,)
                    e.g. [-54., -54.,  -5.,  54.,  54.,   3.]
        """
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = cfg.max_voxel_num

        self.double_flip = cfg.__dict__.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def __call__(
        self,
        res: Dict[str,Any],
        info: Dict[str,Any]
    ) -> Tuple[ Dict[str,Any], Dict[str,Any] ]:
        """
            Given a point cloud of shape (N, 5), e.g. N=277783,
            add 3 copies of it, mirrored along the x-dim, then mirrored
            along y-dim, and then mirrored along both the x- and y-dim.

            These are added to the res['lidar'] dictionary

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
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size
        # [352, 400]

        double_flip = self.double_flip and (res["mode"] != 'train')

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict

        # points = points[:int(points.shape[0] * 0.1), :]
        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"]
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            

        return res, info
