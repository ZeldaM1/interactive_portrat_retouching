from functools import lru_cache

import cv2
import numpy as np


def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def draw_probmap(x):
    return cv2.applyColorMap((x).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 5
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    


