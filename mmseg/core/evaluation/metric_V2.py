import numpy as np
from skimage import morphology

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
