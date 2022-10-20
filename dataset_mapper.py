import copy

import numpy as np
import torch

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from augmentation import MultiModalAugmentation, MultiModalAugInput
from example_multi_modal_augmentations import CopyPasteAugmentation, CopyPasteTransform


class MultiModalDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        annotations = None
        if any([isinstance(aug, MultiModalAugmentation) for aug in self.augmentations.augs]):
            if "annotations" in dataset_dict:
                annotations = dataset_dict['annotations']

            aug_input = MultiModalAugInput(image, annos=annotations, keypoint_hflip_indices=self.keypoint_hflip_indices)
            transforms = self.augmentations(aug_input)
            # update annotations in-place
            image, dataset_dict['annotations'] = aug_input.image, aug_input.annos

           # copy_paste_tfm = list(filter(lambda tfm: isinstance(tfm, CopyPasteTransform), transforms))[0]

            #copy_paste_tfm.apply_multi_modal(self, img, annos, *args)

            # Since annotations are already changed, they cannot be applied later onto the image
            transforms = []
        else:
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
