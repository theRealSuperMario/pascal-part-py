import os
import numpy as np
from pascal_part_py.VOClabelcolormap import color_map
from pascal_part_py.anno import ImageAnnotation
import glob

from pascal_part_py import voc_utils

import pandas as pd
import os
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io

from pascal_part_py import voc_utils


class PascalPartDataset(voc_utils.PascalVOCDataset):
    def __init__(
        self, VOC_root_dir, dir_Annotations_Part, object_class, data_split,
    ):
        """Dataset to iterate over Pascal Parts Dataset
        
        Parameters
        ----------
        VOC_root_dir : str
            directory from VOC 2010 or later dataset, subdirs `JPEGImages`, `Annotations`, `ImageSets/Main`
        dir_Annotations_Part : str
            directory `Annotations_Part` from pascal parts dataset, contains .mat files
        """
        self.dir_Annotations_Part = dir_Annotations_Part
        super(PascalPartDataset, self).__init__(VOC_root_dir, object_class, data_split)

    def __getitem__(self, i):
        example = super(PascalPartDataset, self).__getitem__(i)
        fname = example["annotation"]["filename"]  # .jpg file
        fname = os.path.splitext(fname)[0]
        fname_im = fname + ".jpg"
        fname_part_anno = fname + ".mat"
        an = ImageAnnotation(
            os.path.join(self.voc.dir_JPEGImages, fname_im),
            os.path.join(self.dir_Annotations_Part, fname_part_anno),
        )
        return {
            "annotations_part": an,
            "image": an.im,
            "class_segmentation": an.cls_mask,
            "instance_segmentation": an.inst_mask,
            "part_segmentation": an.part_mask,
            "annotation": example["annotation"],
        }


import functools


class CroppedPascalPartDataset(voc_utils.CroppedPascalVOCDataset):
    def __init__(
        self,
        VOC_root_dir,
        dir_cropped_csv,
        dir_Annotations_Part,
        object_class,
        data_split,
    ):
        self.dir_Annotations_Part = dir_Annotations_Part
        super(CroppedPascalPartDataset, self).__init__(
            VOC_root_dir, dir_cropped_csv, object_class, data_split
        )

    def __getitem__(self, i):
        example = super(CroppedPascalPartDataset, self).__getitem__(i)
        fname = example["fname"]
        fname_im = fname + ".jpg"
        fname_part_anno = fname + ".mat"
        an = ImageAnnotation(
            os.path.join(self.voc.dir_JPEGImages, fname_im),
            os.path.join(self.dir_Annotations_Part, fname_part_anno),
        )
        bbox = {
            "xmin": int(example["xmin"]),
            "xmax": int(example["xmax"]),
            "ymin": int(example["ymin"]),
            "ymax": int(example["ymax"]),
        }
        crop = functools.partial(voc_utils.crop_box, **bbox)
        example["image"] = crop(an.im)
        example["class_segmentation"] = crop(an.cls_mask)
        example["instance_segmentation"] = crop(an.inst_mask)
        example["part_segmentation"] = crop(an.part_mask)
        return example


if __name__ == "__main__":

    for image_set in voc_utils.OBJECT_CLASSES_NAMES:
        for split in voc_utils.DATA_SPLIT:
            dset = voc_utils.PascalPartDataset(
                "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010",
                "/media/sandro/Volume/datasets/PascalVOC/python_voc_devkit/csv/",
                "/media/sandro/Volume/datasets/PascalVOC/PascalParts/trainval/Annotations_Part",
                data_type=split,
                category=image_set,
            )

