import pytest
from python_pascal_voc import torch_datasets
from maskrcnn_benchmark.structures.bounding_box import BoxList
from python_pascal_voc import voc_utils
import numpy as np

DIR_VOC_ROOT = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010/"
csv_dir = (
    "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010/part_csvs"
)


class Test_CroppedPascalPartsDataset:
    def test_croppedPascalPartsDataset(self, tmpdir):
        # csv_dir = tmpdir.mkdir("csv_dir")
        split = "train"
        dset = torch_datasets.CroppedPascalPartsDataset(DIR_VOC_ROOT, csv_dir, split)
        img, target, index = dset[0]

        overlay = voc_utils.overlay_boxes_without_labels(
            np.array(img), target.bbox.numpy()
        )
        from matplotlib import pyplot as plt

        plt.imshow(overlay)
        plt.show()
        assert isinstance(target, BoxList)

        im_info = dset.get_img_info(0)
        assert im_info["height"] == img.size[1]
        assert im_info["width"] == img.size[0]
