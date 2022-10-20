import copy
import random
from typing import Callable

import numpy as np
from PIL import Image

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from augmentation import MultiModalTransform, MultiModalAugmentation, MultiModalAugInput


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

    def __init__(self, paste_img, paste_masks, pre_augs, instance_rate, min_count_to_apply) -> None:
        super().__init__()
        self._set_attributes(locals())

    def apply_multi_modal(self, img, annos, *args):
        masks = []
        for anno in annos:
            mask = MultiModalTransform._get_bitmask(anno["segmentation"], img.shape[:2])
            masks.append(mask)

        composed_mask = np.array(np.where(np.any(np.array(self.paste_masks), axis=0), 1, 0), dtype=np.uint8, order='F')
        updated_masks = np.array(np.where(composed_mask, 0, np.array(masks)), dtype=np.uint8, order='F')

        # TODO: discard totally occluded masks and bboxes

        img = img * (1 - composed_mask) + self.paste_img * composed_mask

        # TODO: update annotations by masks and bboxes

        return img, annos


class CopyPasteAugmentation(MultiModalAugmentation):
    def __init__(self, dataset, image_format: str, pre_augs=(), instance_rate=0.5, min_count_to_apply=1, ) -> None:
        super().__init__()
        self._init(locals())
        self.pre_augs = T.AugmentationList(pre_augs)

    def get_transform(self, image):
        paste_img, paste_masks = self._get_paste_instance()

        return CopyPasteTransform(paste_img, paste_masks, self.pre_augs, self.instance_rate,
                                  self.min_count_to_apply)

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

        paste_masks = []
        for idx in selected_idx:
            mask = MultiModalTransform._get_bitmask(paste_dict["annotations"][idx]["segmentation"], self.image_size)
            paste_masks.append(transforms.apply_segmentation(mask))

        return self.paste_img, paste_masks
