import cv2
import math
import random
import numpy as np
from functools import lru_cache
from .sample import DSample


class BasePointSampler:
    def __init__(self):
        self._selected_mask = None
        self._selected_masks = None

    def sample_object(self, sample: DSample):
        raise NotImplementedError

    def sample_points(self):
        raise NotImplementedError

    @property
    def selected_mask(self):
        assert self._selected_mask is not None
        return self._selected_mask

    @selected_mask.setter
    def selected_mask(self, mask):
        # print('------------', mask.shape)
        self._selected_mask = mask[np.newaxis, :].astype(np.float32)


class MultiPointSampler(BasePointSampler):
    def __init__(self, max_num_points, prob_gamma=0.7, expand_ratio=0.1,
                 positive_erode_prob=0.9, positive_erode_iters=3,
                 negative_bg_prob=0.1, negative_other_prob=0.4, negative_border_prob=0.5,
                 merge_objects_prob=0.0, max_num_merged_objects=2,
                 use_hierarchy=False, soft_targets=False,
                 first_click_center=False, only_one_first_click=False,
                 sfc_inner_k=1.7, sfc_full_inner_prob=0.0):
        super().__init__()
        self.max_num_points = max_num_points
        self.expand_ratio = expand_ratio
        self.positive_erode_prob = positive_erode_prob
        self.positive_erode_iters = positive_erode_iters
        self.merge_objects_prob = merge_objects_prob
        self.use_hierarchy = use_hierarchy
        self.soft_targets = soft_targets
        self.first_click_center = first_click_center
        self.only_one_first_click = only_one_first_click
        self.sfc_inner_k = sfc_inner_k
        self.sfc_full_inner_prob = sfc_full_inner_prob

        if max_num_merged_objects == -1:
            max_num_merged_objects = max_num_points
        self.max_num_merged_objects = max_num_merged_objects

        self.neg_strategies = ['bg', 'other', 'border']
        self.neg_strategies_prob = [negative_bg_prob, negative_other_prob, negative_border_prob]
        assert math.isclose(sum(self.neg_strategies_prob), 1.0)

        self._pos_probs = generate_probs(max_num_points, gamma=prob_gamma)
        self._neg_probs = generate_probs(max_num_points + 1, gamma=prob_gamma)
        self._neg_masks = None

    def sample_object(self, mask):
        gt_mask, pos_masks, neg_masks = self._sample_mask(mask)
        binary_gt_mask = gt_mask
        self.selected_mask = gt_mask
        self._selected_masks = pos_masks
        neg_mask_bg = np.logical_not(binary_gt_mask)
        neg_mask_border = self._get_border_mask(binary_gt_mask)
        neg_mask_other = neg_mask_bg
        self._neg_masks = {
            'bg': neg_mask_bg,
            'other': neg_mask_other,
            'border': neg_mask_border,
            'required': neg_masks
        }

    def _sample_mask(self, mask):

        pos_segments = []
        neg_segments = []
        obj_gt_mask, obj_pos_segments, obj_neg_segments = mask, [mask], []
        pos_segments.extend(obj_pos_segments)
        neg_segments.extend(obj_neg_segments)

        pos_masks = [self._positive_erode(x) for x in pos_segments]
        neg_masks = [self._positive_erode(x) for x in neg_segments]
        return mask, pos_masks, neg_masks

    def sample_points(self):
        assert self._selected_mask is not None
        pos_points = self._multi_mask_sample_points(self._selected_masks,
                                                    is_negative=[False] * len(self._selected_masks),
                                                    with_first_click=self.first_click_center)

        neg_strategy = [(self._neg_masks[k], prob)
                        for k, prob in zip(self.neg_strategies, self.neg_strategies_prob)]
        neg_masks = self._neg_masks['required'] + [neg_strategy]
        neg_points = self._multi_mask_sample_points(neg_masks, is_negative=[False] * len(self._neg_masks['required']) + [True])

        return pos_points + neg_points

    def _multi_mask_sample_points(self, selected_masks, is_negative, with_first_click=False):
        selected_masks = selected_masks[:self.max_num_points]

        each_obj_points = [
            self._sample_points(mask, is_negative=is_negative[i],
                                with_first_click=with_first_click)
            for i, mask in enumerate(selected_masks)
        ]
        each_obj_points = [x for x in each_obj_points if len(x) > 0]

        points = []
        if len(each_obj_points) == 1:
            points = each_obj_points[0]
        elif len(each_obj_points) > 1:
            if self.only_one_first_click:
                each_obj_points = each_obj_points[:1]

            points = [obj_points[0] for obj_points in each_obj_points]

            aggregated_masks_with_prob = []
            for indx, x in enumerate(selected_masks):
                if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
                    for t, prob in x:
                        aggregated_masks_with_prob.append((t, prob / len(selected_masks)))
                else:
                    aggregated_masks_with_prob.append((x, 1.0 / len(selected_masks)))

            other_points_union = self._sample_points(aggregated_masks_with_prob, is_negative=True)
            if len(other_points_union) + len(points) <= self.max_num_points:
                points.extend(other_points_union)
            else:
                points.extend(random.sample(other_points_union, self.max_num_points - len(points)))

        if len(points) < self.max_num_points:
            points.extend([(-1, -1, -1)] * (self.max_num_points - len(points)))

        return points

    def _sample_points(self, mask, is_negative=False, with_first_click=False):
        if is_negative:
            num_points = np.random.choice(np.arange(self.max_num_points + 1), p=self._neg_probs)
        else:
            num_points = 1 + np.random.choice(np.arange(self.max_num_points), p=self._pos_probs)

        indices_probs = None
        if isinstance(mask, (list, tuple)):
            indices_probs = [x[1] for x in mask]
            indices = [(np.argwhere(x), prob) for x, prob in mask]
            if indices_probs:
                assert math.isclose(sum(indices_probs), 1.0)
        else:
            indices = np.argwhere(mask)

        points = []
        for j in range(num_points):
            first_click = with_first_click and j == 0 and indices_probs is None

            if first_click:
                point_indices = get_point_candidates(mask, k=self.sfc_inner_k, full_prob=self.sfc_full_inner_prob)
            elif indices_probs:
                point_indices_indx = np.random.choice(np.arange(len(indices)), p=indices_probs)
                point_indices = indices[point_indices_indx][0]
            else:
                point_indices = indices

            num_indices = len(point_indices)
            if num_indices > 0:
                point_indx = 0 if first_click else 100
                click = point_indices[np.random.randint(0, num_indices)].tolist() + [point_indx]
                points.append(click)

        return points

    def _positive_erode(self, mask):
        if random.random() > self.positive_erode_prob:
            return mask

        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(mask.astype(np.uint8),
                                kernel, iterations=self.positive_erode_iters).astype(np.bool)

        if eroded_mask.sum() > 10:
            return eroded_mask
        else:
            return mask

    def _get_border_mask(self, mask):
        expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
        kernel = np.ones((3, 3), np.uint8)
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
        expanded_mask[mask.astype(np.bool)] = 0
        return expanded_mask


@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    probs = []
    last_value = 1
    for i in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs)
    probs /= probs.sum()

    return probs


def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
    if full_prob > 0 and random.random() < full_prob:
        return obj_mask

    padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    if k > 0:
        inner_mask = dt > dt.max() / k
        return np.argwhere(inner_mask)
    else:
        prob_map = dt.flatten()
        prob_map /= max(prob_map.sum(), 1e-6)
        click_indx = np.random.choice(len(prob_map), p=prob_map)
        click_coords = np.unravel_index(click_indx, dt.shape)
        return np.array([click_coords])
