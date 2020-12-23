
from typing import Any, Dict, Tuple

class DoubleFlip(object):
    def __init__(self):
        pass

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
        
        Returns:
            res["lidar"]['yflip_points'] set to mirror combined (N,5) array of [x, y, z, intensity, timestamps]
            res["lidar"]['xflip_points'] as above, mirrored
            res["lidar"]["double_flip_points"] as above, mirrored
        """
        # y flip
        points = res["lidar"]["points"].copy()
        points[:, 1] = -points[:, 1]

        res["lidar"]['yflip_points'] = points

        # x flip
        points = res["lidar"]["points"].copy()
        points[:, 0] = -points[:, 0]

        res["lidar"]['xflip_points'] = points

        # x y flip
        points = res["lidar"]["points"].copy()
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]

        res["lidar"]["double_flip_points"] = points  

        return res, info 
