import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        classification_logits = []
        for feature_map in x:
            conv_results = self.conv(feature_map)
            logits = self.cls_logits(conv_results)
            
            N, C, H, W = logits.shape
            logits = logits.view(N, C, H, W).permute(0, 2, 3, 1)
            logits = logits.reshape(N, -1, C)
            classification_logits.append(logits)

        return classification_logits


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        regression_outputs = []
        center_scores = []
        for feature_map in x:
            conv_results = self.conv(feature_map)

            reg = self.bbox_reg(conv_results)
            N, _, H, W = reg.shape
            reg = reg.view(N, 4, H, W).permute(0, 2, 3, 1)
            reg = reg.reshape(N, -1, 4)
            regression_outputs.append(reg)

            ctrness = self.bbox_ctrness(conv_results)
            ctrness = ctrness.view(N, 1, H, W).permute(0, 2, 3, 1)
            ctrness = ctrness.reshape(N, -1)
            center_scores.append(ctrness)

        return regression_outputs, center_scores


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    '''
    points: (Level,H,W,2)
    strides: (Level,)
    reg_range: (Level,2)
    cls_logits: (Level,N,HxW,C)
    reg_outputs: (Level,N,HxW,4)
    ctr_logits: (Level,N,HxW)
    targets: (N,)
    '''
    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        for level in range(len(points)):
            points[level] = points[level].reshape([-1, 2]) # Level,HxW,2

        num_points_per_level = [len(points_per_level) for points_per_level in points]
        num_objects_per_img = [len(target['boxes']) for target in targets]
        points_all_level = torch.cat(points, dim=0) # (num_points, 2)
        num_points = len(points_all_level) # num_points = LevelxHxW
        num_obj_all = sum(num_objects_per_img)

        boxes_all_im = []
        labels_all_im = []
        areas_all_im = []
        for target in targets:
            boxes_per_im = target['boxes']
            labels_per_im = target['labels']
            boxes_all_im.append(boxes_per_im)
            labels_all_im.append(labels_per_im)
            # areas
            areas_per_im = (boxes_per_im[:, 2] - boxes_per_im[:, 0]) * (boxes_per_im[:, 3] - boxes_per_im[:, 1])
            areas_all_im.append(areas_per_im)
            
        boxes_all_im = torch.cat(boxes_all_im, dim=0)
        labels_all_im = torch.cat(labels_all_im, dim=0)
        areas_all_im = torch.cat(areas_all_im, dim=0)
        
        # expand boxes
        boxes_all_im = boxes_all_im[None].expand(num_points, num_obj_all, 4) # (num_points, num_obj_all,4)
        labels_all_im = labels_all_im[None].expand(num_points, num_obj_all) # (num_points, num_obj_all)
        areas_all_im = areas_all_im[None].expand(num_points, num_obj_all) # (num_points, num_obj_all)

        #expand boxes and points x and y
        xs, ys = points_all_level[:, 1], points_all_level[:, 0]
        xs = xs[:, None].expand(num_points, num_obj_all) # (num_points, num_obj_all)
        ys = ys[:, None].expand(num_points, num_obj_all) # (num_points, num_obj_all)
        

        # expand reg range
        expanded_object_sizes_of_interest = []
        for level, points_per_level in enumerate(points):
            expanded_object_sizes_of_interest.append(reg_range[level][None].expand(len(points_per_level), -1))
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0) # (num_points, 2)
        reg_range_all = expanded_object_sizes_of_interest[:, None, :].expand(num_points, num_obj_all, 2)# (num_points, num_obj_all,2)


        # center sampling
        center_x = (boxes_all_im[..., 0] + boxes_all_im[..., 2]) / 2
        center_y = (boxes_all_im[..., 1] + boxes_all_im[..., 3]) / 2

        #expand stride
        strides_expand = torch.zeros_like(center_x) # (num_points, num_obj_all)
        strides_expand_for_targets = torch.zeros_like(center_x) # (num_points, num_obj_all)

        # project the points on current level back to the `original` sizes
        p_start = 0
        for level, n_p in enumerate(num_points_per_level):
            p_end = p_start + n_p
            strides_expand[p_start:p_end] = self.center_sampling_radius * strides[level]
            strides_expand_for_targets[p_start:p_end] = strides[level]
            p_start = p_end

        x_min = center_x - strides_expand
        y_min = center_y - strides_expand
        x_max = center_x + strides_expand
        y_max = center_y + strides_expand

        is_in_box = (xs<boxes_all_im[..., 2]) & (xs>boxes_all_im[..., 0]) & (ys<boxes_all_im[..., 3]) & (ys>boxes_all_im[..., 1]) 
        is_in_center = (xs<x_max) & (xs>x_min) & (ys<y_max) & (ys>y_min) 

        # eq1
        l = (xs - boxes_all_im[..., 0])/strides_expand_for_targets
        t = (ys - boxes_all_im[..., 1])/strides_expand_for_targets
        r = (boxes_all_im[..., 2] - xs)/strides_expand_for_targets
        b = (boxes_all_im[..., 3] - ys)/strides_expand_for_targets
        reg_targets = torch.stack([l, t, r, b], dim=2) # (num_points, num_obj_all,4)

        max_reg_targets = reg_targets.max(dim=2)[0]# (num_points, num_obj_all)
        is_in_reg_range = ((max_reg_targets > reg_range_all[..., 0]) & (max_reg_targets < reg_range_all[..., 1])) # (num_points, num_obj_all)

        valid_points = is_in_box & is_in_reg_range & is_in_center

        labels_all_im = torch.where(valid_points, labels_all_im, self.num_classes)
        areas_all_im = torch.where(valid_points, areas_all_im, 1e8)

        obj_start = 0
        reg_targets_lst = []
        labels_lst = []
        for n in range(len(targets)):
            obj_end = obj_start + num_objects_per_img[n]
            reg_targets_per_img = reg_targets[:,obj_start:obj_end,:]
            labels_per_img = labels_all_im[:,obj_start:obj_end]
            areas_per_img = areas_all_im[:,obj_start:obj_end]
            obj_start = obj_end

            min_area, min_area_inds = areas_per_img.min(dim=1) # (num_points,)
            labels_per_img = labels_per_img[range(num_points),min_area_inds]
            labels_lst.append(torch.split(labels_per_img, num_points_per_level, dim=0))
            
            reg_targets_per_img = reg_targets_per_img[range(num_points), min_area_inds] 
            reg_targets_lst.append(torch.split(reg_targets_per_img, num_points_per_level, dim=0))
        
        #level first
        labels_level_first = []
        reg_targets_level_first = []

        for level in range(len(points)):
            labels_level_first.append(torch.cat([labels_tmp[level] for labels_tmp in labels_lst], dim=0))
            reg_targets_per_level = torch.cat([reg_targets_tmp[level]for reg_targets_tmp in reg_targets_lst], dim=0)
            reg_targets_level_first.append(reg_targets_per_level)

        labels_flatten = [labels_l.reshape(-1) for labels_l in labels_level_first]
        reg_targets_flatten = [reg_targets_l.reshape(-1, 4) for reg_targets_l in reg_targets_level_first]
        cls_logits_flatten = [cls_logit.reshape(-1, self.num_classes) for cls_logit in cls_logits]
        reg_outputs_flatten = [reg_output.reshape(-1, 4) for reg_output in reg_outputs]
        ctr_logits_flatten = [ctr_logit.reshape(-1) for ctr_logit in ctr_logits]


        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        cls_logits_flatten = torch.cat(cls_logits_flatten, dim=0)
        reg_outputs_flatten = torch.cat(reg_outputs_flatten, dim=0)
        ctr_logits_flatten = torch.cat(ctr_logits_flatten, dim=0)

        # choose foreground points
        positive_mask = ((labels_flatten >= 0) & (labels_flatten < self.num_classes)).nonzero().reshape(-1)
        num_positive_points = torch.tensor(len(positive_mask), dtype=torch.float)#, device=reg_outputs_flatten[0].device)
        num_positive_points = max(num_positive_points, 1.0)
        reg_outputs_flatten = reg_outputs_flatten[positive_mask]
        ctr_logits_flatten = ctr_logits_flatten[positive_mask]
        reg_targets_flatten = reg_targets_flatten[positive_mask]
        
        labels_flatten_num_classes = torch.nn.functional.one_hot(labels_flatten,num_classes=self.num_classes+1)
        labels_flatten_num_classes = labels_flatten_num_classes[:,:self.num_classes]
        
        cls_loss = sigmoid_focal_loss(cls_logits_flatten,labels_flatten_num_classes)

        if len(positive_mask) > 0:
            num_imgs = cls_logits[0].size(0)
            flatten_points = torch.cat([points[None].expand(num_imgs, 2) for points in points_all_level], dim=0)
            pos_points = flatten_points[positive_mask]
            
            tmp_l = pos_points[...,1] - reg_outputs_flatten[..., 0]
            tmp_r = pos_points[...,1] + reg_outputs_flatten[..., 2]
            tmp_t = pos_points[...,0] - reg_outputs_flatten[..., 1]
            tmp_b = pos_points[...,0] + reg_outputs_flatten[..., 3]
            decoded_reg_outputs_flatten = torch.stack((tmp_l, tmp_t, tmp_r, tmp_b), -1)
            
            tmp_l = pos_points[...,1] - reg_targets_flatten[..., 0]
            tmp_r = pos_points[...,1] + reg_targets_flatten[..., 2]
            tmp_t = pos_points[...,0] - reg_targets_flatten[..., 1]
            tmp_b = pos_points[...,0] + reg_targets_flatten[..., 3]
            decoded_reg_targets_flatten = torch.stack((tmp_l, tmp_t, tmp_r, tmp_b), -1)
            
            reg_loss = giou_loss(decoded_reg_outputs_flatten, decoded_reg_targets_flatten)

            left_right = reg_targets_flatten[:, [0, 2]]
            top_bottom = reg_targets_flatten[:, [1, 3]]
            centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            centerness_targets =  torch.sqrt(centerness)
            ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits_flatten, centerness_targets,reduction='none') 

        else:
            reg_loss = torch.zeros_like(cls_loss)
            ctr_loss = torch.zeros_like(cls_loss)

        cls_loss = cls_loss.sum()/ num_positive_points
        reg_loss = reg_loss.sum()/ num_positive_points
        ctr_loss = ctr_loss.sum()/ num_positive_points
        final_loss = cls_loss + reg_loss + ctr_loss
        
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss,
        }

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level 
        (1) compute the object scores
        (2) filter out boxes with object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        for i in range(len(cls_logits)):
            cls_logits[i] = cls_logits[i].sigmoid()
        for i in range(len(ctr_logits)):
            ctr_logits[i] = ctr_logits[i].sigmoid()
        detections = []
        for i in range(len(image_shapes)):
            image_boxes = []
            image_scores = []
            image_labels = []
            H, W = image_shapes[i]
            for level in range(len(points)):
                # (1) Compute the object scores
                ctr_logits_expanded = ctr_logits[level][i][:,None]
                objectness_scores = torch.sqrt(cls_logits[level][i]*ctr_logits_expanded)

                # (2) Filter out boxes with object scores below the threshold
                # keep_indices shape HxW,C
                keep_indices = objectness_scores > self.score_thresh
                objectness_scores = objectness_scores[keep_indices]
                per_candidate_nonzeros = keep_indices.nonzero()
                per_box_loc = per_candidate_nonzeros[:, 0]
                per_class = per_candidate_nonzeros[:, 1] + 1 # Offset labels by +1
                per_box_regression = reg_outputs[level][i][per_box_loc]
                points[level] = points[level].reshape([-1, 2])
                per_locations = points[level][per_box_loc]
                
                # (3) Select the top K boxes
                if keep_indices.sum().item() > self.topk_candidates:
                    objectness_scores, top_k_indices = objectness_scores.topk(self.topk_candidates, sorted=False)
                    # (per_pre_nms_top_n,)
                    per_class = per_class[top_k_indices]
                    # (per_pre_nms_top_n,4)
                    per_box_regression = per_box_regression[top_k_indices]
                    # (per_pre_nms_top_n,2) 
                    per_locations = per_locations[top_k_indices]
    
                # (4) Decode the boxes
                box_x0 = per_locations[:, 1] - per_box_regression[:, 0] * strides[level]
                box_y0 = per_locations[:, 0] - per_box_regression[:, 1] * strides[level]
                box_x1 = per_locations[:, 1] + per_box_regression[:, 2] * strides[level]
                box_y1 = per_locations[:, 0] + per_box_regression[:, 3] * strides[level]

                # (5) Clip boxes outside of the image boundaries and remove small boxes

                box_x0 = box_x0.clamp(min = 0, max = W-1)
                box_x1 = box_x1.clamp(min = 0, max = W-1)
                box_y0 = box_y0.clamp(min = 0, max = H-1)
                box_y1 = box_y1.clamp(min = 0, max = H-1)

                new_keep_indices = ((box_x1 - box_x0)>0)
                new_keep_indices &= ((box_y1 - box_y0)>0)

                boxes = torch.stack([box_x0, box_y0, box_x1, box_y1], dim=-1)

                boxes = boxes[new_keep_indices]
                scores = objectness_scores[new_keep_indices]
                labels = per_class[new_keep_indices]

                # Append detections for this level
                image_boxes.append(boxes)
                image_scores.append(scores)
                image_labels.append(labels)

            # (b) Collect all candidate boxes across all pyramid levels
            image_boxes = torch.cat(image_boxes, dim = 0)
            image_scores = torch.cat(image_scores, dim = 0)
            image_labels = torch.cat(image_labels, dim = 0)

            # (c) Run non-maximum suppression to remove duplicated boxes within each category
            keep_indices = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)

            # (d) Keep the top K boxes after NMS
            keep_indices = keep_indices[:self.detections_per_img]

            # Append the final detections for this image
            detections.append(
                {
                    "boxes": image_boxes[keep_indices],
                    "scores": image_scores[keep_indices],
                    "labels": image_labels[keep_indices],  
                }
            )

        return detections

