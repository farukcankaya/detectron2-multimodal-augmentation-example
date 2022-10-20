import copy
import random
from typing import Callable

import numpy as np
from PIL import Image

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from augmentation import MultiModalTransform, MultiModalAugmentation, MultiModalAugInput
import pycocotools.mask as mask_util


class InstanceColorJitterTransform(MultiModalTransform):

    def __init__(self, color_operation: Callable, instance_rate: float, min_count_to_apply: int) -> None:
        if not callable(color_operation):
            raise ValueError("color_operation parameter should be callable")

        super().__init__()
        self._set_attributes(locals())

    def apply_multi_modal(self, img, annos, *args):
        instance_count = len(annos)
        apply_count = max(self.min_count_to_apply, int(instance_count * self.instance_rate))
        selected_idx = random.sample(list(range(instance_count)), apply_count)

        for idx in selected_idx:
            segm = annos[idx]["segmentation"]
            bitmask = MultiModalTransform._get_bitmask(segm, img.shape[:2])

            augmented_instance = self.color_operation(Image.fromarray(img * bitmask))
            img = img * (1 - bitmask) + np.asarray(augmented_instance) * bitmask

        return img, annos


class InstanceColorJitterAugmentation(MultiModalAugmentation):

    def __init__(self, color_operation: Callable, instance_rate=0.5, min_count_to_apply=1) -> None:
        super().__init__()
        self._init(locals())

    def get_transform(self, image):
        return InstanceColorJitterTransform(self.color_operation, self.instance_rate, self.min_count_to_apply)


class CopyPasteTransform(MultiModalTransform):

    def __init__(self, paste_dict, paste_img, pre_augs, instance_rate, min_count_to_apply, bbox_occluded_thr,
                 mask_occluded_thr) -> None:
        super().__init__()
        self._set_attributes(locals())

    def apply_multi_modal(self, img, annos, *args):
        masks = []
        labels = []
        for anno in annos:
            mask = MultiModalTransform._get_bitmask(anno["segmentation"], img.shape[:2])
            masks.append(mask)
            labels.append(anno["category_id"])

        paste_masks = []
        paste_labels = []
        for anno in self.paste_dict["annotations"]:
            paste_masks.append(anno["segmentation"])
            paste_labels.append(anno["category_id"])

        src_masks = np.array(np.where(np.array(paste_masks) == 1, 1, 0), dtype=np.uint8)
        dst_masks = np.array(np.where(np.array(masks) == 1, 1, 0), dtype=np.uint8)
        dst_labels = np.array(labels)
        src_labels = np.array(paste_labels)

        composed_mask = np.array(np.where(np.any(src_masks, axis=0), 1, 0), dtype=np.uint8, order='F')
        updated_masks = np.array(np.where(composed_mask, 0, dst_masks), dtype=np.uint8, order='F')
        src_bboxes = self._get_boxes(src_masks)
        dst_bboxes = self._get_boxes(dst_masks)
        updated_dst_bboxes = self._get_boxes(updated_masks)
        assert len(updated_dst_bboxes) == len(updated_masks)

        # filter
        bboxes_inds = np.all(np.abs((updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr, axis=-1)
        masks_inds = updated_masks.sum(axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | np.squeeze(masks_inds)

        # copy paste
        img = img * (1 - composed_mask) + self.paste_img * composed_mask

        # update
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate([updated_masks[valid_inds], src_masks])

        composed_annos = []
        mask_type = 'polygons' if isinstance(annos[0]["segmentation"], list) else 'bitmask'
        for i in range(len(bboxes)):
            anno = {}
            anno["bbox_mode"] = BoxMode.XYXY_ABS
            anno["bbox"] = bboxes[i]
            anno["segmentation"] = MultiModalTransform._to_polygons(
                masks[i]) if mask_type == 'polygons' else mask_util.encode(masks[i])
            anno["category_id"] = labels[i]

            composed_annos.append(anno)

        return img, composed_annos

    def _get_boxes(self, masks):
        """
        Adapted from https://github.com/open-mmlab/mmdetection/blob/1376e77e6ecbaad609f6003725158de24ed42e84/mmdet/core/mask/structures.py#L532
        :param masks:
        :return:
        """
        num_masks = len(masks)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        x_any = masks.any(axis=1)
        y_any = masks.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                         dtype=np.float32)
        return boxes


class CopyPasteAugmentation(MultiModalAugmentation):
    def __init__(self,
                 dataset,
                 image_format: str,
                 pre_augs=(),
                 instance_rate=0.5,
                 min_count_to_apply=1,
                 bbox_occluded_thr=10,
                 mask_occluded_thr=300,
                 ) -> None:
        super().__init__()
        self._init(locals())
        self.pre_augs = copy.deepcopy(T.AugmentationList(pre_augs))

    def get_transform(self, image):
        paste_dict, paste_img, transforms = self._get_paste_instance()

        return CopyPasteTransform(paste_dict, paste_img, transforms, self.instance_rate,
                                  self.min_count_to_apply, self.bbox_occluded_thr, self.mask_occluded_thr)

    def _get_paste_instance(self):
        data_count = len(self.dataset)
        paste_idx = random.sample(list(range(data_count)), 1)[0]
        paste_dict = copy.deepcopy(self.dataset[paste_idx])

        self.paste_img = utils.read_image(paste_dict["file_name"], format=self.image_format)
        self.image_size = self.paste_img.shape[:2]
        aug_input = T.AugInput(self.paste_img)
        transforms = self.pre_augs(aug_input)
        self.paste_img = aug_input.image

        instance_count = len(paste_dict["annotations"])
        apply_count = max(self.min_count_to_apply, int(instance_count * self.instance_rate))
        selected_idx = random.sample(list(range(instance_count)), apply_count)

        paste_annos = []
        for idx in selected_idx:
            anno = copy.deepcopy(paste_dict["annotations"][idx])
            mask = MultiModalTransform._get_bitmask(paste_dict["annotations"][idx]["segmentation"], self.image_size)
            ones = np.ones_like(mask)
            anno["segmentation"] = transforms.apply_segmentation(mask)
            anno["bbox"] = transforms.apply_box(paste_dict["annotations"][idx]["bbox"])
            paste_annos.append(anno)

        paste_dict["annotations"] = paste_annos

        return paste_dict, self.paste_img, transforms
