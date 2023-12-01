import mmcv
import numpy as np
from skimage import morphology

def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray): Prediction segmentation map.
        label (ndarray): Ground truth segmentation map.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=np.float)
    total_area_union = np.zeros((num_classes, ), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=np.float)
    total_area_label = np.zeros((num_classes, ), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category IoU, shape (num_classes, ).
    """

    all_acc, acc, iou = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category dice, shape (num_classes, ).
    """

    all_acc, acc, dice = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray]): List of prediction segmentation maps.
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, ).
         ndarray: Per category evalution metrics, shape (num_classes, ).
    """

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    acc = total_area_intersect / total_area_label
    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics

def eval_metrics_unpaved(results, gt_seg_maps):
    size = results[0].shape
    kernal_size = 9
    precision_total = np.zeros(2)
    recall_total = np.zeros(2)
    precision_num = np.zeros(2)
    recall_num = np.zeros(2)

    for idx in range(len(results)):
        result = results[idx]
        gt_seg_map = gt_seg_maps[idx]
        # results for unpaved and paved
        result_unpaved = np.array(result)
        result_unpaved[result_unpaved == 2] = 0

        result_paved = np.array(result)
        result_paved[result_paved == 1] = 0
        result_paved[result_paved != 0] = 1

        # segmentation for unpaved and paved
        seg_unpaved = np.array(gt_seg_map)
        seg_unpaved[seg_unpaved == 2] = 0

        seg_paved = np.array(gt_seg_map)
        seg_paved[seg_paved == 1] = 0
        seg_paved[seg_paved != 0] = 1

        result_unpaved_skel = morphology.skeletonize(result_unpaved.astype(np.uint8), method='lee')
        result_paved_skel = morphology.skeletonize(result_paved.astype(np.uint8), method='lee')

        result_unpaved_non_index = list(zip(*np.nonzero(result_unpaved_skel)))
        result_paved_non_index = list(zip(*np.nonzero(result_paved_skel)))

        seg_unpaved_skel = morphology.skeletonize(seg_unpaved, method='lee')
        seg_paved_skel = morphology.skeletonize(seg_paved, method='lee')

        seg_unpaved_non_index = list(zip(*np.nonzero(seg_unpaved_skel)))
        seg_paved_non_index = list(zip(*np.nonzero(seg_paved_skel)))

        # ---------------------------计算非铺装道路指标---------------------------
        unpaved_TP = 0
        unpaved_FP = 0
        unpaved_FN = 0

        for pre_unpaved_index in result_unpaved_non_index:
            pre_left_bound = max(0, pre_unpaved_index[1] - kernal_size)
            pre_right_bound = min(size[1], pre_unpaved_index[1] + kernal_size)
            pre_up_bound = min(size[0], pre_unpaved_index[0] + kernal_size)
            pre_down_bound = max(0, pre_unpaved_index[0] - kernal_size)

            if not seg_unpaved_skel[pre_down_bound:pre_up_bound, pre_left_bound:pre_right_bound].any():
                unpaved_FN += 1

        for seg_unpaved_index in seg_unpaved_non_index:
            seg_left_bound = max(0, seg_unpaved_index[1] - kernal_size)
            seg_right_bound = min(size[1], seg_unpaved_index[1] + kernal_size)
            seg_up_bound = min(size[0], seg_unpaved_index[0] + kernal_size)
            seg_down_bound = max(0, seg_unpaved_index[0] - kernal_size)

            if not result_unpaved_skel[seg_down_bound:seg_up_bound, seg_left_bound:seg_right_bound].any():
                unpaved_FP += 1
            else:
                unpaved_TP += 1

        # ---------------------------计算铺装道路指标---------------------------
        paved_TP = 0
        paved_FP = 0
        paved_FN = 0

        for pre_paved_index in result_paved_non_index:
            pre_left_bound = max(0, pre_paved_index[1] - kernal_size)
            pre_right_bound = min(size[1], pre_paved_index[1] + kernal_size)
            pre_up_bound = min(size[0], pre_paved_index[0] + kernal_size)
            pre_down_bound = max(0, pre_paved_index[0] - kernal_size)

            if not seg_paved_skel[pre_down_bound:pre_up_bound, pre_left_bound:pre_right_bound].any():
                paved_FN += 1

        for seg_paved_index in seg_paved_non_index:
            seg_left_bound = max(0, seg_paved_index[1] - kernal_size)
            seg_right_bound = min(size[1], seg_paved_index[1] + kernal_size)
            seg_up_bound = min(size[0], seg_paved_index[0] + kernal_size)
            seg_down_bound = max(0, seg_paved_index[0] - kernal_size)

            if not result_paved_skel[seg_down_bound:seg_up_bound, seg_left_bound:seg_right_bound].any():
                paved_FP += 1
            else:
                paved_TP += 1

        precision = np.zeros(2)
        recall = np.zeros(2)

        if not any(seg_unpaved_non_index):
            precision[0] = None
        else:
            precision[0] = unpaved_TP / (unpaved_TP + unpaved_FP)
            precision_total[0] += precision[0]
            precision_num[0] += 1

        if not any(result_unpaved_non_index):
            recall[0] = None
        else:
            recall[0] = unpaved_TP / (unpaved_TP + unpaved_FN)
            recall_total[0] += recall[0]
            recall_num[0] += 1

        if not any(seg_paved_non_index):
            precision[1] = None
        else:
            precision[1] = paved_TP / (paved_TP + paved_FP)
            precision_total[1] += precision[1]
            precision_num[1] += 1

        if not any(result_paved_non_index):
            recall[1] = None
        else:
            recall[1] = paved_TP / (paved_TP + paved_FN)
            recall_total[1] += recall[1]
            recall_num[1] += 1

    return [precision_total / precision_num, recall_total / recall_num]
