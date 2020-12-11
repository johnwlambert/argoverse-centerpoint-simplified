
import numpy as np

from centerpoint.utils.center_utils import draw_umich_gaussian, gaussian_radius



class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.remove_environment = cfg.remove_environment
        self.shuffle_points = cfg.shuffle_points
        self.remove_unknown = cfg.remove_unknown_examples
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.add_rgb_to_points = cfg.get("add_rgb_to_points", False)
        self.reference_detections = cfg.get("reference_detections", None)
        self.remove_outside_points = cfg.get("remove_outside_points", False)
        self.random_crop = cfg.get("random_crop", False)

        self.normalize_intensity = cfg.get("normalize_intensity", False)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.gt_rotation_noise = cfg.gt_rot_noise
            self.gt_loc_noise_std = cfg.gt_loc_noise
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_random_rot_range = cfg.global_rot_per_obj_range
            self.global_translate_noise_std = cfg.global_trans_noise
            self.gt_points_drop = (cfg.gt_drop_percentage,)
            self.remove_points_after_sample = cfg.remove_points_after_sample
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.flip_single = cfg.get("flip_single", False)
            self.npoints = cfg.get("npoints", -1)
            self.random_select = cfg.get("random_select", False)

        self.symmetry_intensity = cfg.get("symmetry_intensity", False)
        self.kitti_double = cfg.get("kitti_double", False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

            if "difficulty" not in anno_dict:
                difficulty = np.zeros([anno_dict["boxes"].shape[0]], dtype=np.int32)
                gt_dict["difficulty"] = difficulty
            else:
                gt_dict["difficulty"] = anno_dict["difficulty"]

        if "calib" in res:
            calib = res["calib"]
        else:
            calib = None

        if self.add_rgb_to_points:
            assert calib is not None and "image" in res
            image_path = res["image"]["image_path"]
            image = (
                    imgio.imread(str(pathlib.Path(root_path) / image_path)).astype(
                        np.float32
                    )
                    / 255
            )
            points_rgb = box_np_ops.add_rgb_to_points(
                points, image, calib["rect"], calib["Trv2c"], calib["P2"]
            )
            points = np.concatenate([points, points_rgb], axis=1)
            num_point_features += 3

        if self.reference_detections is not None:
            assert calib is not None and "image" in res
            C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
            frustums = box_np_ops.get_frustum_v2(reference_detections, C)
            frustums -= T
            frustums = np.einsum("ij, akj->aki", np.linalg.inv(R), frustums)
            frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
            surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
            masks = points_in_convex_polygon_3d_jit(points, surfaces)
            points = points[masks.any(-1)]

        if self.remove_outside_points:
            assert calib is not None
            image_shape = res["metadata"]["image_shape"]
            points = box_np_ops.remove_outside_points(
                points, calib["rect"], calib["Trv2c"], calib["P2"], image_shape
            )
        if self.remove_environment is True and self.mode == "train":
            selected = keep_arrays_by_name(gt_names, target_assigner.classes)
            _dict_select(gt_dict, selected)
            masks = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])
            points = points[masks.any(-1)]

        if self.mode == "train":
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)
            if self.remove_unknown:
                remove_mask = gt_dict["difficulty"] == -1
                """
                gt_boxes_remove = gt_boxes[remove_mask]
                gt_boxes_remove[:, 3:6] += 0.25
                points = prep.remove_points_in_boxes(points, gt_boxes_remove)
                """
                keep_mask = np.logical_not(remove_mask)
                _dict_select(gt_dict, keep_mask)
            gt_dict.pop("difficulty")

            if self.min_points_in_gt > 0:
                # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    self.random_crop,
                    gt_group_ids=None,
                    calib=calib,
                    road_planes=None # res["lidar"]["ground_plane"]
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    if self.remove_points_after_sample:
                        masks = box_np_ops.points_in_rbbox(points, sampled_gt_boxes)
                        points = points[np.logical_not(masks.any(-1))]

                    points = np.concatenate([sampled_points, points], axis=0)
            prep.noise_per_object_v3_(
                gt_dict["gt_boxes"],
                points,
                gt_boxes_mask,
                rotation_perturb=self.gt_rotation_noise,
                center_noise_std=self.gt_loc_noise_std,
                global_random_rot_range=self.global_random_rot_range,
                group_ids=None,
                num_try=100,
            )

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            iskitti = res["type"] in ["KittiDataset"]

            if self.kitti_double:
                assert False, "No more KITTI"
                gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points, flip_coor=70.4/2)
            elif self.flip_single or iskitti:
                assert False, "nuscenes double flip is better"
                gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"], points)
            else:
                gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )

        if self.shuffle_points:
            # shuffle is a little slow.
            np.random.shuffle(points)

        if self.mode == "train" and self.random_select:
            if self.npoints < points.shape[0]:
                pts_depth = points[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(
                    near_idxs, self.npoints - len(far_idxs_choice), replace=False
                )

                choice = (
                    np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
                    if len(far_idxs_choice) > 0
                    else near_idxs_choice
                )
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                if self.npoints > len(points):
                    extra_choice = np.random.choice(
                        choice, self.npoints - len(points), replace=False
                    )
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            points = points[choice]

        if self.symmetry_intensity:
            points[:, -1] -= 0.5  # translate intensity to [-0.5, 0.5]
            # points[:, -1] *= 2

        if self.normalize_intensity and res["type"] in ["NuScenesDataset"]:
            # print(points[:20, 3])
            assert 0, "Velocity Accuracy drops 3 percent with normalization.."
            points[:, 3] /= 255

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info



def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.no_log = assigner_cfg.get("no_log", False)

    def __call__(self, res, info):
        max_objs = self._max_objs * self.dense_reg
        class_names_by_task = [t.class_names for t in self.tasks]


        dxy = [(0, 0)]

        # Calculate output featuremap size
        grid_size = res["lidar"]["voxels"]["shape"]  # 448 x 512
        pc_range = res["lidar"]["voxels"]["range"]
        voxel_size = res["lidar"]["voxels"]["size"]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                # print("classes: ", gt_dict["gt_classes"], "name", class_name)
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 8), dtype=np.float32)
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)
                direction = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        if not (y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1]):
                            # a double check, should never happen
                            print(x, y, y * feature_map_size[0] + x)
                            assert False 

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            if not self.no_log:
                                anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                    np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                            else:
                                anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, gt_dict['gt_boxes'][idx][k][3:6],
                                    np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.sin(rot), np.cos(rot)), axis=None)

                        else:
                            raise NotImplementedError("Only Support KITTI and nuScene for Now!")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info
