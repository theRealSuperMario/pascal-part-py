import pytest
from pascal_part_py.datasets import PascalPartDataset
from pascal_part_py import datasets
from pascal_part_py import voc_utils
import pandas as pd
import os
import collections

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_PASCAL_CSV = os.environ["DIR_PASCAL_CSV"]
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
            voc_utils.OBJECT_CLASS.aeroplane, voc_utils.DATA_SPLIT.train, csv_dir
        )
        assert len(df) == 403  # pascal VOC 2010
        assert len(os.listdir(csv_dir)) == 1

        df = voc.load_object_class_cropped(None, voc_utils.DATA_SPLIT.train, csv_dir)
        assert len(df) == 13339  # pascal VOC 2010, all annotated objects
        assert len(os.listdir(csv_dir)) == 2


class Test_CroppedPascalVoc:
    def test_croppedPascalVOC(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        dset = voc_utils.CroppedPascalVOC(
            DIR_VOC_ROOT,
            csv_dir,
            voc_utils.OBJECT_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        ex = dset[0]
        assert len(dset) == 403  # pascal VOC 2010


class Test_PascalVOCDataset:
    def test_dataset(self):
        dset = voc_utils.PascalVOCDataset(
            DIR_VOC_ROOT, voc_utils.OBJECT_CLASS.aeroplane, voc_utils.DATA_SPLIT.train
        )
        assert len(dset) == 283  # pascal VOC 2010

        dset = voc_utils.PascalVOCDataset(
            DIR_VOC_ROOT, None, voc_utils.DATA_SPLIT.train
        )
        assert len(dset) == 4998  # pascal VOC 2010


class Test_PascalPartDataset:
    def test_dataset(self):
        dset = datasets.PascalPartDataset(
            DIR_VOC_ROOT,
            DIR_ANNOTATIONS_PART,
            voc_utils.OBJECT_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        ex = dset[0]


class Test_CroppedPascalPartDataset:
    def test_dataset(self, tmpdir):
        csv_dir = tmpdir.mkdir("csvs")
        dset = datasets.CroppedPascalPartDataset(
            DIR_VOC_ROOT,
            csv_dir,
            DIR_ANNOTATIONS_PART,
            voc_utils.OBJECT_CLASS.aeroplane,
            voc_utils.DATA_SPLIT.train,
        )
        ex = dset[0]
        assert len(dset) == 403  # pascal VOC 2010
