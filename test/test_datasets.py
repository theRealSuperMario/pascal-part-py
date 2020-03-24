import pytest
from python_pascal_voc import datasets
from python_pascal_voc import voc_utils, pascal_part
import pandas as pd
import os
import collections

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


class Test_VOCutils:
    def test_load_annotation(self, tmpdir):
        """ load single annotation """
        voc = voc_utils.VOCLoader(DIR_VOC_ROOT)
        anno = voc.load_annotation(voc.annotation_files[0])
        df = pd.DataFrame.from_dict(anno, orient="index")

        assert len(df.iloc[0].object) == 1  # pascal VOC 2010

        anno = voc.load_annotation(voc.annotation_files[2])
        df = pd.DataFrame.from_dict(anno, orient="index")
        assert len(df.iloc[0].object) == 3  # pascal VOC 2010

    def test_load_object_class_cropped(self, tmpdir):
        """ load single bounding box annotations """
        csv_dir = tmpdir.mkdir("csv")
        voc = voc_utils.VOCLoader(DIR_VOC_ROOT)
        df = voc.load_object_class_cropped(
            voc_utils.ANNOTATION_CLASS.aeroplane, voc_utils.DATA_SPLIT.train, csv_dir
        )
        assert len(df) == 403  # pascal VOC 2010
        assert len(os.listdir(csv_dir)) == 1

        df = voc.load_object_class_cropped(None, voc_utils.DATA_SPLIT.train, csv_dir)
        assert len(df) == 13339  # pascal VOC 2010, all annotated objects
        assert len(os.listdir(csv_dir)) == 2


class Test_CroppedPascalVoc:
    def test_croppedPascalVOC(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        dset = datasets.CroppedPascalVOCDataset(
            DIR_VOC_ROOT,
            csv_dir,
            voc_utils.ANNOTATION_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        ex = dset[0]
        assert len(dset) == 403  # pascal VOC 2010


class Test_PascalVOCDataset:
    def test_dataset(self):
        dset = datasets.PascalVOCDataset(
            DIR_VOC_ROOT,
            voc_utils.ANNOTATION_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        assert len(dset) == 283  # pascal VOC 2010

        dset = datasets.PascalVOCDataset(DIR_VOC_ROOT, None, voc_utils.DATA_SPLIT.train)
        assert len(dset) == 4998  # pascal VOC 2010


@pytest.mark.pascalpart
class Test_PascalPartDataset:
    def test_dataset(self):
        dset = datasets.PascalPartDataset(
            DIR_VOC_ROOT,
            DIR_ANNOTATIONS_PART,
            voc_utils.ANNOTATION_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        assert len(dset) == 283  # pascal VOC 2010

        object_classes = set([])
        for i in range(4):
            ex = dset[i]
            image_annotation = ex["annotations_part"]
            for obj in image_annotation.objects:
                object_classes.add(obj.object_class)
        assert len(set(object_classes)) == 1

        dset = datasets.PascalPartDataset(
            DIR_VOC_ROOT, DIR_ANNOTATIONS_PART, None, voc_utils.DATA_SPLIT.train,
        )
        assert len(dset) == 4998  # pascal VOC 2010
        image_annotation = ex["annotations_part"]

        object_classes = set([])
        for i in range(10):
            ex = dset[i]
            image_annotation = ex["annotations_part"]
            for obj in image_annotation.objects:
                object_classes.add(obj.object_class)
        assert len(set(object_classes)) > 1


@pytest.mark.pascalpart
class Test_CroppedPascalPartDataset:
    def test_dataset(self, tmpdir):
        csv_dir = tmpdir.mkdir("csvs")
        dset = datasets.CroppedPascalPartDataset(
            DIR_VOC_ROOT,
            csv_dir,
            DIR_ANNOTATIONS_PART,
            voc_utils.ANNOTATION_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        ex = dset[0]
        assert len(dset) == 403  # pascal VOC 2010

        object_classes = set([])
        for i in range(4):
            ex = dset[i]
            image_annotation = ex["annotations_part"]
            for obj in image_annotation.objects:
                object_classes.add(obj.object_class)
        assert len(set(object_classes)) == 1

        dset = datasets.CroppedPascalPartDataset(
            DIR_VOC_ROOT,
            csv_dir,
            DIR_ANNOTATIONS_PART,
            None,
            voc_utils.DATA_SPLIT.train,
        )
        assert len(dset) == 13339  # pascal VOC 2010
        image_annotation = ex["annotations_part"]

        object_classes = set([])
        for i in range(10):
            ex = dset[i]
            image_annotation = ex["annotations_part"]
            for obj in image_annotation.objects:
                object_classes.add(obj.object_class)
        assert len(set(object_classes)) > 1


import cv2
from python_pascal_voc import voc_utils, datasets, pascal_part_annotation


import numpy as np


class Test_FilteredCroppedPascalParts:
    def test_dataset(self, tmpdir):
        csv_dir = tmpdir.mkdir("csvs")
        part_remapping = {
            "head": [
                pascal_part.HORSE_PARTS.head,
                pascal_part.HORSE_PARTS.leye,
                pascal_part.HORSE_PARTS.reye,
                pascal_part.HORSE_PARTS.lear,
                pascal_part.HORSE_PARTS.rear,
                pascal_part.HORSE_PARTS.muzzle,
            ],
            "neck": [pascal_part.HORSE_PARTS.neck],
            "torso": [pascal_part.HORSE_PARTS.torso],
            "legs": [
                pascal_part.HORSE_PARTS.lfuleg,
                pascal_part.HORSE_PARTS.lflleg,
                pascal_part.HORSE_PARTS.rfuleg,
                pascal_part.HORSE_PARTS.rflleg,
                pascal_part.HORSE_PARTS.lbuleg,
                pascal_part.HORSE_PARTS.lblleg,
                pascal_part.HORSE_PARTS.rbuleg,
                pascal_part.HORSE_PARTS.rblleg,
                pascal_part.HORSE_PARTS.lfho,
                pascal_part.HORSE_PARTS.rfho,
                pascal_part.HORSE_PARTS.blho,
                pascal_part.HORSE_PARTS.rbho,
            ],
            "tail": [pascal_part.HORSE_PARTS.tail],
            "background": [voc_utils.ANNOTATION_CLASS.background],
        }
        dset = datasets.FilteredCroppedPascalParts(
            DIR_VOC_ROOT,
            csv_dir,
            DIR_ANNOTATIONS_PART,
            voc_utils.ANNOTATION_CLASS.horse,
            voc_utils.DATA_SPLIT.train,
            part_remapping,
        )
        ex = dset[0]

        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1)
        ax.imshow(ex["part_segmentation"].as_rgb())
