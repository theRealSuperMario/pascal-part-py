import os
import numpy as np
from pascal_part_py.VOClabelcolormap import color_map
from pascal_part_py.anno import ImageAnnotation
import glob
from pascal_part_py.voc_utils import crop_box


import pandas as pd
import os
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io

from pascal_part_py.voc_utils import PascalVOCDataset


class PascalPartDataset(PascalVOCDataset):
    def __init__(
        self,
        VOC_root_dir,
        dir_pascal_csv,
        dir_Annotations_Part,
        data_type="train",
        category="bicycle",
    ):
        """Dataset to iterate over Pascal Parts Dataset
        
        Parameters
        ----------
        VOC_root_dir : str
            directory from VOC 2010 or later dataset, subdirs `JPEGImages`, `Annotations`, `ImageSets/Main`
        dir_Annotations_Part : str
            directory `Annotations_Part` from pascal parts dataset, contains .mat files
        """
        super(PascalPartDataset, self).__init__(VOC_root_dir, dir_pascal_csv)
        self.dir_Annotations_Part = dir_Annotations_Part

        # returns dataframe
        self.labels = self._load_data(category, data_type)
        # converts dataframe to list of dicts
        self.labels = self.labels.to_dict("records")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        example = self.labels[i]
        fname = example["fname"]
        fname = os.path.splitext(fname)[0]
        fname_im = fname + ".jpg"
        fname_anno = fname + ".mat"
        an = ImageAnnotation(
            os.path.join(self.dir_JPEGImages, fname_im),
            os.path.join(self.dir_Annotations_Part, fname_anno),
        )
        xmax = example["xmax"]
        xmin = example["xmin"]
        ymax = example["ymax"]
        ymin = example["ymin"]
        bbox = {
            "xmax": xmax,
            "xmin": xmin,
            "ymax": ymax,
            "ymin": ymin,
        }
        return {
            "annotations_part": an,
            "image": an.im,
            "class_mask": an.inst_mask,
            "instance_mask": an.inst_mask,
            "part_mask": an.part_mask,
            "fname": fname,
            "bbox": bbox,
        }


import functools


class CroppedPascalPartDataset(PascalPartDataset):
    def __getitem__(self, i):
        example = super(CroppedPascalPartDataset, self).__getitem__(i)
        bbox = example["bbox"]
        crop = functools.partial(crop_box, **bbox)
        example["image"] = crop(example["image"])
        example["class_mask"] = crop(example["class_mask"])
        example["instance_mask"] = crop(example["instance_mask"])
        example["part_mask"] = crop(example["part_mask"])
        return example


if __name__ == "__main__":

    for image_set in PascalVOCDataset.list_image_sets():
        for split in PascalVOCDataset.SPLITS():
            dset = PascalPartDataset(
                "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010",
                "/media/sandro/Volume/datasets/PascalVOC/python_voc_devkit/csv/",
                "/media/sandro/Volume/datasets/PascalVOC/PascalParts/trainval/Annotations_Part",
                data_type=split,
                category=image_set,
            )

