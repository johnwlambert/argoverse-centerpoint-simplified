

import copy 
import logging
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from det3d.core import box_torch_ops
from det3d.models.losses.centernet_loss import FocalLoss, SmoothRegLoss, RegLoss
from det3d.core.utils.center_utils import ddd_decode
from det3d.models.utils import Sequential



class CenterHead(nn.Module):
    def __init__(
        self,
        mode="3d",
        in_channels=[128,],
        norm_cfg=None,
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        encode_rad_error_by_sin=False,
        loss_aux=None,
        direction_offset=0.0,
        direction_weight=0.0,
        name="centerhead",
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        smooth_loss=False,
        no_log=False,
        num_hm_conv=2,
        dcn_head=False,
        bn=True
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # weight between hm loss and loc loss
        self.dataset = dataset

        self.encode_background_as_zeros = True
        self.use_sigmoid_score = True
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FocalLoss()
        self.crit_reg = RegLoss()
        self.loss_aux = None
        
        self.no_log = no_log

        self.box_n_dim = 9 if dataset == 'nuscenes' else 7  # change this if your box is different
        self.num_anchor_per_locs = [n for n in num_classes]
        self.use_direction_classifier = False 

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        self.bev_only = True if mode == "bev" else False

        logger.info(
            f"num_classes: {num_classes}"
        )

        # a shared convolution 
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        self.smooth_loss = smooth_loss
        if self.smooth_loss:
            print("Use Smooth L1 Loss!!")
            self.crit_reg = SmoothRegLoss()

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3, directional_classifier=False)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3, directional_classifier=False)
                )

        logger.info("Finish CenterHead Initialization")

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset == 'nuscenes':
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['vel'], preds_dict['rot']), dim=1)
            elif self.dataset == 'waymo':
                preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                    preds_dict['rot']), dim=1)                  
            else:
                raise NotImplementedError()

            loss = 0
            ret = {}
 
            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss += hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged


    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict['hm'].shape[0]
            num_class_with_bg = self.num_classes[task_id]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, C, H, W = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, C, H, W)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[2]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[3])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[2, 3])


            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            if "anchors_mask" not in example:
                batch_anchors_mask = [None] * batch_size
            else:
                assert False 

            batch_hm = preds_dict['hm'].sigmoid_()

            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if not self.no_log:
                batch_dim = torch.exp(preds_dict['dim'])
            else:
                batch_dim = preds_dict['dim']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)
 
                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, 1] = 1 - batch_reg[:, 1, 1]
                batch_reg[:, 2, 0] = 1 - batch_reg[:, 2, 0]
                 
                batch_reg[:, 3, 0] = 1 - batch_reg[:, 3, 0]
                batch_reg[:, 3, 1] = 1 - batch_reg[:, 3, 1]
                batch_reg = batch_reg.mean(dim=1)

                batch_rots = preds_dict['rot'][:, :, 0:1]
                batch_rotc = preds_dict['rot'][:, :, 1:2]

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] = -batch_rotc[:, 1]


                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] = -batch_rots[:, 2]

                # double flip 
                batch_rots[:, 3] = -batch_rots[:, 3]
                batch_rotc[:, 3] = -batch_rotc[:, 3]

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            else:
                batch_rots = preds_dict['rot'][:, 0].unsqueeze(1)
                batch_rotc = preds_dict['rot'][:, 1].unsqueeze(1)


            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']
                if double_flip:
                    # flip vy
                    batch_vel[:, 1, 1] = - batch_vel[:, 1, 1]
                    # flip vx
                    batch_vel[:, 2, 0] = - batch_vel[:, 2, 0]

                    batch_vel[:, 3] = - batch_vel[:, 3]

                    batch_vel = batch_vel.mean(dim=1)
            else:
                batch_vel = None

            batch_dir_preds = [None] * batch_size
    
            temp = ddd_decode(
                batch_hm,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_dir_preds,
                batch_vel,
                None,
                reg=batch_reg,
                post_center_range=post_center_range,
                K=test_cfg.max_per_img,
                score_threshold=test_cfg.score_threshold,
                cfg=test_cfg,
                task_id=task_id
            )

            batch_reg_preds = [box['box3d_lidar'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['label_preds'] for box in temp]

            metas.append(meta_list)
            
            if test_cfg.get('max_pool_nms', False) or test_cfg.get('circle_nms', False):
                rets.append(temp)
                continue

            rets.append(
                self.get_task_detections(
                    task_id,
                    num_class_with_bg,
                    test_cfg,
                    batch_cls_preds,
                    batch_reg_preds,
                    batch_cls_labels,
                    batch_dir_preds,
                    batch_anchors_mask,
                    meta_list,
                )
            )

        # Merge branches results
        num_tasks = len(rets)
        ret_list = []
        num_preds = len(rets)
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 

    def get_task_detections(
        self,
        task_id,
        num_class_with_bg,
        test_cfg,
        batch_cls_preds,
        batch_reg_preds,
        batch_cls_labels,
        batch_dir_preds=None,
        batch_anchors_mask=None,
        meta_list=None,
    ):
        predictions_dicts = []
        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device,
            )

        for box_preds, cls_preds, cls_labels, dir_preds, a_mask, meta in zip(
            batch_reg_preds,
            batch_cls_preds,
            batch_cls_labels,
            batch_dir_preds,
            batch_anchors_mask,
            meta_list,
        ):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]

            box_preds = box_preds.float()
            cls_preds = cls_preds.float()

            if self.use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            if self.encode_background_as_zeros:
                # this don't support softmax
                assert self.use_sigmoid_score is True
                # total_scores = torch.sigmoid(cls_preds)
                total_scores = cls_preds
            else:
                # encode background as first element in one-hot vector
                if self.use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            # Apply NMS in birdeye view
            if test_cfg.nms.use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            assert test_cfg.nms.use_multi_class_nms is False 
            """feature_map_size_prod = (
                batch_reg_preds.shape[1] // self.num_anchor_per_locs[task_id]
            )"""
            if test_cfg.nms.use_multi_class_nms:
                assert self.encode_background_as_zeros is True
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]
                if not test_cfg.nms.use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4]
                    )
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners
                    )

                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []

                scores = total_scores
                boxes = boxes_for_nms
                selected_per_class = []
                score_threshs = [test_cfg.score_threshold] * self.num_classes[task_id]
                pre_max_sizes = [test_cfg.nms.nms_pre_max_size] * self.num_classes[
                    task_id
                ]
                post_max_sizes = [test_cfg.nms.nms_post_max_size] * self.num_classes[
                    task_id
                ]
                iou_thresholds = [test_cfg.nms.nms_iou_threshold] * self.num_classes[
                    task_id
                ]

                for class_idx, score_thresh, pre_ms, post_ms, iou_th in zip(
                    range(self.num_classes[task_id]),
                    score_threshs,
                    pre_max_sizes,
                    post_max_sizes,
                    iou_thresholds,
                ):
                    self._nms_class_agnostic = False
                    if self._nms_class_agnostic:
                        class_scores = total_scores.view(
                            feature_map_size_prod, -1, self.num_classes[task_id]
                        )[..., class_idx]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])
                        class_boxes = box_preds
                        class_dir_labels = dir_labels
                    else:
                        # anchors_range = self.target_assigner.anchors_range(class_idx)
                        anchors_range = self.target_assigners[task_id].anchors_range
                        class_scores = total_scores.view(
                            -1, self._num_classes[task_id]
                        )[anchors_range[0] : anchors_range[1], class_idx]
                        class_boxes_nms = boxes.view(-1, boxes_for_nms.shape[-1])[
                            anchors_range[0] : anchors_range[1], :
                        ]
                        class_scores = class_scores.contiguous().view(-1)
                        class_boxes_nms = class_boxes_nms.contiguous().view(
                            -1, boxes_for_nms.shape[-1]
                        )
                        class_boxes = box_preds.view(-1, box_preds.shape[-1])[
                            anchors_range[0] : anchors_range[1], :
                        ]
                        class_boxes = class_boxes.contiguous().view(
                            -1, box_preds.shape[-1]
                        )
                        if self.use_direction_classifier:
                            class_dir_labels = dir_labels.view(-1)[
                                anchors_range[0] : anchors_range[1]
                            ]
                            class_dir_labels = class_dir_labels.contiguous().view(-1)
                    if score_thresh > 0.0:
                        class_scores_keep = class_scores >= score_thresh
                        if class_scores_keep.shape[0] == 0:
                            selected_per_class.append(None)
                            continue
                        class_scores = class_scores[class_scores_keep]
                    if class_scores.shape[0] != 0:
                        if score_thresh > 0.0:
                            class_boxes_nms = class_boxes_nms[class_scores_keep]
                            class_boxes = class_boxes[class_scores_keep]
                            class_dir_labels = class_dir_labels[class_scores_keep]
                        keep = nms_func(
                            class_boxes_nms, class_scores, pre_ms, post_ms, iou_th
                        )
                        if keep.shape[0] != 0:
                            selected_per_class.append(keep)
                        else:
                            selected_per_class.append(None)
                    else:
                        selected_per_class.append(None)
                    selected = selected_per_class[-1]

                    if selected is not None:
                        selected_boxes.append(class_boxes[selected])
                        selected_labels.append(
                            torch.full(
                                [class_boxes[selected].shape[0]],
                                class_idx,
                                dtype=torch.int64,
                                device=box_preds.device,
                            )
                        )
                        if self.use_direction_classifier:
                            selected_dir_labels.append(class_dir_labels[selected])
                        selected_scores.append(class_scores[selected])
                    # else:
                    #     selected_boxes.append(torch.Tensor([], device=class_boxes.device))
                    #     selected_labels.append(torch.Tensor([], device=box_preds.device))
                    #     selected_scores.append(torch.Tensor([], device=class_scores.device))
                    #     if self.use_direction_classifier:
                    #         selected_dir_labels.append(torch.Tensor([], device=class_dir_labels.device))

                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self.use_direction_classifier:
                    selected_dir_labels = torch.cat(selected_dir_labels, dim=0)

            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long,
                    )

                else:
                    top_labels = cls_labels.long()
                    top_scores = total_scores.squeeze(-1)
                    # top_scores, top_labels = torch.max(total_scores, dim=-1)

                if test_cfg.score_threshold > 0.0:
                    thresh = torch.tensor(
                        [test_cfg.score_threshold], device=total_scores.device
                    ).type_as(total_scores)
                    top_scores_keep = top_scores >= thresh
                    top_scores = top_scores.masked_select(top_scores_keep)

                if top_scores.shape[0] != 0:
                    if test_cfg.score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self.use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    # boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]

                    # GPU NMS from PCDet(https://github.com/sshaoshuai/PCDet) 
                    boxes_for_nms = box_torch_ops.boxes3d_to_bevboxes_lidar_torch(box_preds)
                    if not test_cfg.nms.use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2],
                            boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4],
                        )
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners
                        )
                    # the nms in 3d detection just remove overlap boxes.

                    selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, top_scores, 
                                thresh=test_cfg.nms.nms_iou_threshold,
                                pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                post_max_size=test_cfg.nms.nms_post_max_size)
                else:
                    selected = []

                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self.use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

            # finally generate predictions.
            # self.logger.info(f"selected boxes: {selected_boxes.shape}")
            if selected_boxes.shape[0] != 0:
                # self.logger.info(f"result not none~ Selected boxes: {selected_boxes.shape}")
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self.use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (
                        (box_preds[..., -1] - self.direction_offset) > 0
                    ) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds),
                    )
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >= post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <= post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, self.box_n_dim], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros(
                        [0], dtype=top_labels.dtype, device=device
                    ),
                    "metadata": meta,
                }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts