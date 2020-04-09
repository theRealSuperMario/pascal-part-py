import pytest
from python_pascal_voc import datasets
from python_pascal_voc import voc_utils, pascal_part
from python_pascal_voc import pascal_part_annotation
import pandas as pd
import os
import collections
from matplotlib import pyplot as plt


DIR_VOC_ROOT = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010/"


class Test_PascalPartLoader:
    def test_test(self, tmpdir):
        csv_dir = tmpdir.mkdir("csvdir")
        loader = pascal_part.PascalPartLoader(DIR_VOC_ROOT)
        df = loader._make_object_class_cropped_data(
            voc_utils.ANNOTATION_CLASS.person, voc_utils.DATA_SPLIT.train, csv_dir
        )

