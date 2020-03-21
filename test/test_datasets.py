import pytest
from pascal_part_py.datasets import PascalPartDataset
from pascal_part_py import voc_utils
import pandas as pd
import os
import collections

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_PASCAL_CSV = os.environ["DIR_PASCAL_CSV"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


# class Test_PascalPartDataset:
#     def test_bicycle(self):
#         split = "train"
#         image_set = "bicycle"
#         dset = PascalPartDataset(
#             DIR_VOC_ROOT,
#             DIR_PASCAL_CSV,
#             DIR_ANNOTATIONS_PART,
#             data_type=split,
#             category=image_set,
#         )
#         ex = dset[0]
#         assert list(ex.keys()) == ["annotations_part", "fname", "bbox"]


class Test_VOCutils:
    def test_load_annotation(self, tmpdir):
        voc = voc_utils.VOCUtils(DIR_VOC_ROOT)
        anno = voc.load_annotation(voc.annotation_files[0])
        df = pd.DataFrame.from_dict(anno, orient="index")

        # item 0 only has a single annotations, which is why it gets returned as single OrderedDict
        assert len(df.iloc[0].object) == 1

        anno = voc.load_annotation(voc.annotation_files[2])
        df = pd.DataFrame.from_dict(anno, orient="index")
        # item 2 has multiple annotations, which is why it gets returned as a list of OrderedDicts
        assert len(df.iloc[0].object) == 3

    def test_load_object_class_cropped(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        voc = voc_utils.VOCUtils(DIR_VOC_ROOT)
        df = voc.load_object_class_cropped(
            voc_utils.OBJECT_CLASS.aeroplane, voc_utils.DATA_SPLITS.train, csv_dir
        )
        assert len(df) == 403  # pascal VOC 2010
        assert len(os.listdir(csv_dir)) == 1

        df = voc.load_object_class_cropped(None, voc_utils.DATA_SPLITS.train, csv_dir)
        assert len(df) == 13339  # pascal VOC 2010, all annotationted objects
        assert len(os.listdir(csv_dir)) == 2


class Test_CroppedPascalVoc:
    def test_croppedPascalVOC(self, tmpdir):
        csv_dir = tmpdir.mkdir("csv")
        dset = voc_utils.CroppedPascalVOC(
            DIR_VOC_ROOT,
            csv_dir,
            voc_utils.OBJECT_CLASS.aeroplane,
            voc_utils.DATA_SPLITS.train,
        )
        ex = dset[0]
        assert len(dset) == 403


# class Test_PascalVOCDataset:
#     def test_voc_dataset(self, tmpdir):
#         csv_dir = tmpdir.mkdir("csv")
#         dset = PascalVOCDataset(DIR_VOC_ROOT, DIR_PASCAL_CSV)
#         assert len(os.listdir(csv_dir)) == 1

#         ex = dset[0]

#         assert list(ex.keys()) == ["annotations_part", "fname", "bbox"]

