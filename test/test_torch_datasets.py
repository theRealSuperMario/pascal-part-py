import pytest
from python_pascal_voc import torch_datasets
from maskrcnn_benchmark.structures.bounding_box import BoxList
from python_pascal_voc import voc_utils
import numpy as np

DIR_VOC_ROOT = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010/"
csv_dir = (
    "/media/sandro/Volume/datasets/PascalVOC/PascalParts/VOCdevkit/VOC2010/part_csvs"
)
import tqdm
import pandas as pd
import os


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

    def test_iterate_dataset_once_save_errors(self, tmpdir):
        split = "trainval"
        dset = torch_datasets.CroppedPascalPartsDataset(DIR_VOC_ROOT, csv_dir, split)
        fault_indices = []
        for i in tqdm.tqdm(range(len(dset))):
            try:
                img, target, index = dset[i]
                assert isinstance(target, BoxList)

                im_info = dset.get_img_info(i)
                assert im_info["height"] == img.size[1]
                assert im_info["width"] == img.size[0]
            except:
                fault_indices.append(i)
        if fault_indices:
            faulty_fnames = set([dset.ids[i]["fname"] for i in fault_indices])
            pd.DataFrame(faulty_fnames).to_csv(
                os.path.join(csv_dir, "faulty_fnames.csv"), index=False, header=False
            )

    def test_iterate_dataset_once(self, tmpdir):
        split = "trainval"
        dset = torch_datasets.CroppedPascalPartsDataset(DIR_VOC_ROOT, csv_dir, split)
        for i in tqdm.tqdm(range(len(dset))):
            img, target, index = dset[i]
            assert isinstance(target, BoxList)

            im_info = dset.get_img_info(i)
            assert im_info["height"] == img.size[1]
            assert im_info["width"] == img.size[0]
