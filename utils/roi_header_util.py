import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import List, Dict, Tuple

import utils.boxes_utils as box_op
from utils.det_utils import *
from utils.drLoss import drLoss
from config.train_config import cfg


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.
    :param class_logits: predicted class, shape=[num_anchors, num_classes]
    :param box_regression: predicted bbox regression
    :param labels: true label
    :param regression_targets: true bbox
    :return: classification_loss (Tensor)
             box_loss (Tensor)
    """
    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)

    if(cfg.use_dr_loss == 1):
        classification_loss = drLoss(class_logits, labels.int()) / (sampled_pos_inds_subset.numel() + N)  #logits, targets
    else:
        classification_loss = F.cross_entropy(class_logits, labels)
    
    '''
    labels = guesses, regression targets = regression targets
    box_cls = logits

    #https://github.com/facebookresearch/maskrcnn-benchmark/blob/57eec25b75144d9fb1a6857f32553e1574177daf/maskrcnn_benchmark/modeling/rpn/retinanet/loss.py#L99  75
    #https://github.com/leedewdew/tensorflow-dr-loss/blob/master/loss.py
    '''

    labels_pos = labels[sampled_pos_inds_subset]

    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos],
                              regression_targets[sampled_pos_inds_subset],
                              beta=1 / 9,
                              size_average=False,
                              ) / labels.numel()

    return classification_loss, box_loss

  
def dr_loss(logits, targets, num_classes):
    '''
        logits:  [1, 80, 80, 105] -> [num_anchors, 7]
        targets: [1, 80, 80, 15]  -> [num_anchors, 1]
    '''
    L = 6.0
    tau = 4.0
    margin = 0.5
    pos_lambda = 1
    neg_lambda = 0.184

    # num_classes = 80
    N = tf.shape(logits)[0]
    logits  = tf.reshape(logits, [-1, num_classes])  # [h*w*a, 7]
    targets = tf.reshape(targets, [-1, 1])  # [h*w*a, 1]

    # class_range = tf.range(1, num_classes + 1, dtype=targets.dtype)
    ## line 552 in anchors.py
    class_range = tf.range(0, num_classes, dtype=targets.dtype)
    class_range = tf.expand_dims(class_range, axis=0)

    pos_ind  = tf.equal(targets, class_range)
    neg_ind  = tf.not_equal(targets, class_range)
    pos_prob = tf.nn.sigmoid(tf.boolean_mask(logits, pos_ind))
    neg_prob = tf.nn.sigmoid(tf.boolean_mask(logits, neg_ind))

    neg_q    = tf.nn.softmax(neg_prob / neg_lambda, axis=0)
    neg_dist = tf.reduce_sum(neg_q * neg_prob)

    def true_fn():
        pos_q    = tf.nn.softmax(-pos_prob / pos_lambda, axis=0)
        pos_dist = tf.reduce_sum(pos_q * pos_prob)
        loss = tau * tf.math.log(1. + tf.math.exp(L * (neg_dist - pos_dist + margin))) / L
        return loss
    def false_fn():
        loss = tau * tf.math.log(1. + tf.math.exp(L * (neg_dist - 1. + margin))) / L
        return loss
    numel = tf.size(pos_prob)
    loss = tf.cond(numel > 0, true_fn, false_fn)
    loss /= tf.cast((numel + N), tf.float32)

    return loss


def add_gt_proposals(proposals, gt_boxes):
    """
    concate gt_box and proposals
    :param proposals: bboxes of predicted by rpn
    :param gt_boxes: true bbox
    :return:
    """

    proposals = [
        torch.cat((proposal, gt_box))
        for proposal, gt_box in zip(proposals, gt_boxes)
    ]
    return proposals


def check_targets(targets):
    assert targets is not None
    assert all(["boxes" in t for t in targets])
    assert all(["labels" in t for t in targets])


class RoIHeads(torch.nn.Module):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,

                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,

                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detection_per_img):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_op.box_iou

        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 0.5
            bg_iou_thresh,  # 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,  # 512
            positive_fraction)  # 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """
        get the matched gt_bbox for every anchors, and set positive/negative samples
        :param proposals:
        :param gt_boxes:
        :param gt_labels:
        :return:
        """

        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # iou of bbox and anchors
                match_quality_matrix = box_op.box_iou(gt_boxes_in_image, proposals_in_image)

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img, as_tuple=False).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self,
                                proposals,
                                targets
                                ):

        check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        proposals = add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,
                               box_regression,
                               proposals,
                               image_shapes
                               ):
        """
        Post-processing the forecast data of the network, including
        （1）Calculate the final bbox coordinates according to the proposal and the predicted regression parameters
        （2）Perform softmax processing on the predicted category results
        （3）Crop the predicted boxes information and adjust the out-of-bounds coordinates to the picture boundary
        （4）Remove all background information
        （5）Remove low-probability targets
        （6）Remove small size targets
        （7）Execute nms processing and sort by scores
        （8）Return the top topk targets according to the scores sorting
        Args:
            class_logits: network prediction class probability information
            box_regression: bounding box regression parameters predicted by the network
            proposals: The proposal output by rpn
            image_shapes: The width and height of each image before packing into batch

        Returns:

        """
        device = class_logits.device
        # Predict the number of target categories
        num_classes = class_logits.shape[-1]

        # Get the number of predicted bboxes for each image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # Calculate the final bbox coordinates according to the proposal and the predicted regression parameters
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # Perform softmax processing on the predicted category results
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # Segment the result according to the predicted number of bboxes for each image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        #  Traverse the prediction information of each image
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            #Crop the predicted boxes information and adjust the out-of-bounds coordinates to the picture boundary
            boxes = box_op.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # Remove all information with index 0 (0 stands for background)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # Remove low probability targets，self.scores_thresh=0.05
            inds = torch.nonzero(scores > self.score_thresh, as_tuple=False).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # Remove small targets
            keep = box_op.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # Execute nms processing, and the results after execution will be sorted and returned according to scores from largest to smallest
            keep = box_op.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # Get the topk prediction targets ranked by scores
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None
                ):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                # assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # Divide the positive and negative samples, and count the labels corresponding to gt and the bounding box regression information
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # Pass the collected sample through the roi_pooling layer
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # Two fully connected layers after roi_pooling
        box_features = self.box_head(box_features)
        # Then predict the target category and bounding box regression parameters separately
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses
