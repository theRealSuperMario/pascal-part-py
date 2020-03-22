import pytest
from python_pascal_voc import datasets, voc_utils, pascal_part_annotation
import pandas as pd
import os
import collections

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


class Test_PascalObject:
    def test_objects(self):
        dset = datasets.PascalPartDataset(
            DIR_VOC_ROOT,
            DIR_ANNOTATIONS_PART,
            voc_utils.ANNOTATION_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        ex = dset[0]
        image_annotation = ex[
            "annotations_part"
        ]  # type : pascal_part_annotation.ImageAnnotation
        assert isinstance(
            image_annotation.objects[0].object_class, voc_utils.ANNOTATION_CLASS
        )
