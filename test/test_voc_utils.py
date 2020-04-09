from python_pascal_voc import datasets
from python_pascal_voc import voc_utils, pascal_part
from python_pascal_voc import pascal_part_annotation
import pandas as pd
import os
import collections


DIR_VOC_ROOT = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010/"


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

        assert "object_id" in df.columns

