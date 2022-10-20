import random
from abc import abstractmethod
from copy import deepcopy
from typing import Optional, List, Callable

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from fvcore.transforms.transform import Transform, NoOpTransform

import detectron2.data.transforms as T
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.structures import polygons_to_bitmask

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


class MultiModalAugInput(T.AugInput):

    def __init__(self,
                 image: np.ndarray,
                 *,
                 annos: Optional[List[dict]] = None,
                 keypoint_hflip_indices: Optional[np.ndarray] = None,
                 boxes: Optional[np.ndarray] = None,
                 sem_seg: Optional[np.ndarray] = None
                 ):
        super().__init__(image, boxes=boxes, sem_seg=sem_seg)
        self.annos = deepcopy(annos)
        self.keypoint_hflip_indices = keypoint_hflip_indices

    def transform(self, tfm: Transform) -> None:
        if isinstance(tfm, MultiModalTransform):
            self.image, self.annos = tfm.apply_multi_modal(self.image, self.annos)
        else:
            super().transform(tfm)
            if self.annos is not None:
                for annotation in self.annos:
                    if "segmentation" in annotation:
                        # each instance contains 1 or more polygons
                        segm = annotation["segmentation"]
                        if isinstance(segm, list):
                            # polygons
                            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                            annotation["segmentation"] = [
                                p.reshape(-1) for p in tfm.apply_polygons(polygons)
                            ]
                        elif isinstance(segm, dict):
                            # RLE
                            mask = mask_util.decode(segm)
                            mask = tfm.apply_segmentation(mask)
                            annotation["segmentation"] = mask
                        else:
                            raise ValueError(
                                "Cannot transform segmentation of type '{}'!"
                                "Supported types are: polygons as list[list[float] or ndarray],"
                                " COCO-style RLE as a dict.".format(type(segm))
                            )

                    if "keypoints" in annotation:
                        keypoints = transform_keypoint_annotations(
                            annotation["keypoints"], tfm, self.image.shape[:2], self.keypoint_hflip_indices
                        )
                        annotation["keypoints"] = keypoints


class SingleOperationNotSupported(NotImplementedError):

    def __init__(self, *args: object) -> None:
        super().__init__("single operation is not supported in MultiModalTransform. "
                         "If you apply a sequence of transformations, you have to apply multimodal "
                         "transformations separately by simply calling `apply_multi_modal(image, masks, boxes)`")


class MultiModalTransform(Transform):
    def apply_image(self, img: np.ndarray):
        return NoOpTransform()

    def apply_coords(self, coords: np.ndarray):
        return NoOpTransform()

    @abstractmethod
    def apply_multi_modal(self, img: np.ndarray, annos, *args):
        """
        implement
        """

    @staticmethod
    def _get_bitmask(segm, image_size):
        if isinstance(segm, list):
            # polygons
            mask = polygons_to_bitmask(segm, *image_size)
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )
        return np.asarray(mask[:, :, None], dtype=np.uint8, order='F')

    @staticmethod
    def _to_polygons(mask):
        # This conversion comes from D4809743 and D5171122,
        # when Mask-RCNN was first developed.
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
            -2
        ]
        polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]
        return polygons


class MultiModalAugmentation(T.Augmentation):
    pass

# Doesn't work!. It only allows to take single type of argument.
# @MultiModalTransform.register_type("multi_modal")
# def func(multi_modal_transform: MultiModalTransform, *args: Any):
# do the work
#    return args
