import pytest
from python_pascal_voc import datasets, voc_utils, pascal_part_annotation
import pandas as pd
import os
import collections


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

    def test_filter_objects(self):
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

        # in this special case, the image_annotation is "aeroplane"
        assert len(image_annotation.objects) == 1

        filter_ = lambda x: x.object_class.name in ["bicycle"]
        image_annotation = pascal_part_annotation.filter_objects(
            filter_, image_annotation
        )
        assert len(image_annotation.objects) == 0


class Test_PartAnnotation:
    def test_objects(self):
        mat_file = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/trainval/Annotations_Part/2008_000002.mat"
        part_anno = pascal_part_annotation.PartAnnotation.from_mat(mat_file)
