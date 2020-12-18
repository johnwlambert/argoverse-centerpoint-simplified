
from typing import Any, Dict

from centerpoint.models.single_stage import SingleStageDetector

class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data: Dict[str,Any]):
        input_features = self.reader(data["features"], data["num_voxels"])
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)

        return x

    def forward(self, example: Dict[str,Any], return_loss: bool = True, **kwargs):
        """
        Args:
            example is a dictionary with keys like
            dict_keys(['metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates', 'annos'])
        
        `voxels` could be a tensor of shape [150031, 10, 5]
        `coordinates` could be a tensor of shape [150031, 4], with entries like [0, 20, 720, 698]
        `num_points_in_voxel` could be a tensor of shape [150031] with entries like [10]
        `num_voxels` could be a tensor of shape [4], with entries like [37507, 37508, 37508, 37508]
        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        # x has shape [4, 512, 180, 180], 8x downsampled from 1440 x 1440
        x = self.extract_feat(data)
        
        # get out list of length 6 (for 6 tasks)
        # each is a dictionary with keys
        #      dict_keys(['reg', 'height', 'dim', 'rot', 'vel', 'hm'])
        #      values could have shape
        #           height: [4, 1, 180, 180]
        #           dim: [4, 3, 180, 180]
        #           rot: [4, 2, 180, 180]
        #           vel: [4, 2, 180, 180]
        #           hm: [4, 1, 180, 180]
        #           reg: [4, 2, 180, 180]
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def pred_hm(self, example):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds_dicts = self.bbox_head(x)

        return preds_dicts 

    def pred_result(self, example, preds):
        return self.bbox_head.predict(example, preds, self.test_cfg)

