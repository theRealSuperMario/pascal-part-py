import pytest
from pascal_part_py.datasets import PascalPartDataset, PascalVOCDataset
from pascal_part_py import voc_utils
import pandas as pd
import os
import collections

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_PASCAL_CSV = os.environ["DIR_PASCAL_CSV"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


class Test_PascalPartDataset:
    def test_bicycle(self):
        split = "train"
        image_set = "bicycle"
        dset = PascalPartDataset(
            DIR_VOC_ROOT,
            DIR_PASCAL_CSV,
            DIR_ANNOTATIONS_PART,
            data_type=split,
            category=image_set,
        )
        ex = dset[0]
        assert list(ex.keys()) == ["annotations_part", "fname", "bbox"]


class Test_VOCutils:
    def test_datasets(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        voc = voc_utils.VOCUtils(DIR_VOC_ROOT, DIR_PASCAL_CSV, csv_dir)
        voc._load_data("bicycle", "train")
        assert len(os.listdir(csv_dir)) == 1

    def test_load_all_annotations_df(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        voc = voc_utils.VOCUtils(DIR_VOC_ROOT, csv_dir)
        # df = voc.load_all_annotations_as_df()
        anno = voc.load_annotation(voc.annotation_files[0])
        df = pd.DataFrame.from_dict(anno, orient="index")

        # item 0 only has a single annotations, which is why it gets returned as single OrderedDict
        assert len(df.iloc[0].object) == 1

        anno = voc.load_annotation(voc.annotation_files[2])
        df = pd.DataFrame.from_dict(anno, orient="index")
        # item 2 has multiple annotations, which is why it gets returned as a list of OrderedDicts
        assert len(df.iloc[0].object) == 3


class Test_PascalVOCDataset:
    def test_voc_dataset(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        dset = PascalVOCDataset(DIR_VOC_ROOT, DIR_PASCAL_CSV)
        assert len(os.listdir(csv_dir)) == 1

        ex = dset[0]

        assert list(ex.keys()) == ["annotations_part", "fname", "bbox"]

