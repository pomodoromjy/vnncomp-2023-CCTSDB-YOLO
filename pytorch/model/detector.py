import torch
import torch.nn as nn

from pytorch.model.fpn import *
from pytorch.model.backbone.shufflenetv2 import *


class DetectorMulInputOneTensor(nn.Module):
    def __init__(self, classes, anchor_num, load_param, mode="train"):
        super(DetectorMulInputOneTensor, self).__init__()
        out_depth = 72
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.mode = mode
        self.backbone = ShuffleNetV2(stage_out_channels, load_param)
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 2 * anchor_num, 3, 2, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 3, 2, 0, bias=True)


    def bbox_iou(self, box1, box2):
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0][0], box2[0][1], box2[0][2], box2[0][3]

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = (w1 * h1 + 1e-16) + w2 * h2 - inter
        iou = inter / union

        return iou

    def eqaul(self, cls1, cls2):
        return cls1 == cls2


    def forward(self, x):
        if self.mode == "train":
            imgs = x['imgs']
        elif self.mode == "test":
            imgs = x[0:12288].reshape(1, 3, 64,64)
            position = [(x[12288].type(torch.int64), x[12289].type(torch.int64))]
            targets = [x[12290:]]

            gt_bbox = targets[0][2:6]
            gt_label = targets[0][1]
            for postion_x, postion_y in position:
                width = 3
                height = 3
                # width = 1
                # height = 1
                patch_mask = torch.ones_like(imgs)
                patch_mask[:, postion_x:postion_x + height, postion_y: postion_y + width, :] = 0
                imgs = imgs * patch_mask
        C2, C3 = self.backbone(imgs)
        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(C2, C3)

        out_reg_2 = self.output_reg_layers(reg_2)
        out_cls_2 = self.output_cls_layers(cls_2)

        out_reg_2 = out_reg_2.squeeze(3).squeeze(2)
        out_cls_2 = out_cls_2.squeeze(3).squeeze(2)

        #
        pred_cls = torch.argmax(out_cls_2, dim=1).detach()
        pred_bbox = out_reg_2.detach()

        if self.mode == "test":
            cls_res = torch.tensor(self.eqaul(pred_cls, gt_label))
            reg_res = torch.tensor(self.bbox_iou(gt_bbox, pred_bbox))

            final_out = cls_res.to(torch.long) * reg_res

        if self.mode == "train":
            return out_reg_2, out_cls_2
        elif self.mode == "test":
            return out_reg_2, out_cls_2, final_out


