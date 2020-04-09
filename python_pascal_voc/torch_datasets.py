import os
import sys
import numpy as np

import torch
import torch.utils.data
from maskrcnn_benchmark.structures.bounding_box import BoxList
from PIL import Image
from torchvision.datasets import VisionDataset

from python_pascal_voc import datasets, voc_utils, pascal_part


from python_pascal_voc import pascal_part_annotation, datasets


class CroppedPascalPartsDataset(datasets.CroppedPascalPartsDataset):
    def __init__(
        self,
        data_dir,
        data_csv_dir,
        split,
        use_difficult=False,
        use_occluded=False,
        use_truncated=False,
        transforms=None,
    ):
        """dataset compliant with maskrcnn_benchmark interface """
        super(CroppedPascalPartsDataset, self).__init__(
            data_dir,
            data_csv_dir,
            split,
            use_difficult=use_difficult,
            use_occluded=use_occluded,
            use_truncated=use_truncated,
        )
        self.transforms = transforms

    def __getitem__(self, index):
        img, target_values, index = super(CroppedPascalPartsDataset, self).__getitem__(
            index
        )

        boxes = target_values["boxes"]
        labels = target_values["labels"]
        object_bbox = target_values["object_bbox"]
        im_info = target_values["im_info"]
        height, width = im_info

        boxes = torch.tensor(boxes, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = BoxList(boxes, (width, height), mode="xyxy")
        target.add_field("labels", labels)
        # TODO: currently, all boxes are easy, even if they are not.
        # The reason for this is because I could not establish the mapping between
        # the pascal part annotation and the pascal voc annotation
        target.add_field("difficult", torch.zeros_like(labels, dtype=labels.dtype))

        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index
