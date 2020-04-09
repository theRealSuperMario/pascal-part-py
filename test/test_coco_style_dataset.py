import pytest
from python_pascal_voc import coco_style_datasets
from python_pascal_voc import datasets
from python_pascal_voc import voc_utils, pascal_part
from python_pascal_voc import pascal_part_annotation
import pandas as pd
import os
import collections
from matplotlib import pyplot as plt
import numpy as np


class Test_PascalPartCropped:
    @pytest.mark.mpl_image_compare
    def test_bounding_boxes(self, tmpdir):

        csv_dir = tmpdir.mkdir("csv")
        part_remapping = {
            "head": [
                pascal_part.PERSON_PARTS.head,
                pascal_part.PERSON_PARTS.hair,
                pascal_part.PERSON_PARTS.leye,
                pascal_part.PERSON_PARTS.reye,
                pascal_part.PERSON_PARTS.lear,
                pascal_part.PERSON_PARTS.rear,
                pascal_part.PERSON_PARTS.nose,
                pascal_part.PERSON_PARTS.mouth,
                pascal_part.PERSON_PARTS.neck,
                pascal_part.PERSON_PARTS.lebrow,
                pascal_part.PERSON_PARTS.rebrow,
            ],
            "torso": [pascal_part.PERSON_PARTS.torso],
            "legs": [
                pascal_part.PERSON_PARTS.ruleg,
                pascal_part.PERSON_PARTS.rlleg,
                pascal_part.PERSON_PARTS.llleg,
                pascal_part.PERSON_PARTS.luleg,
            ],
            "foot": [pascal_part.PERSON_PARTS.lfoot, pascal_part.PERSON_PARTS.rfoot],
            "arm": [
                pascal_part.PERSON_PARTS.ruarm,
                pascal_part.PERSON_PARTS.rlarm,
                pascal_part.PERSON_PARTS.llarm,
                pascal_part.PERSON_PARTS.luarm,
            ],
            "hand": [pascal_part.PERSON_PARTS.lhand, pascal_part.PERSON_PARTS.rhand],
            "background": [voc_utils.ANNOTATION_CLASS.background],
        }
        dset = coco_style_datasets.VOCPartsCropped(
            DIR_VOC_ROOT,
            csv_dir,
            DIR_ANNOTATIONS_PART,
            voc_utils.ANNOTATION_CLASS.person,
            voc_utils.DATA_SPLIT.train,
            part_remapping,
        )

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        for i in range(8):
            ax = axes[i]
            image, boxlist, idx = dset[i]

            overlay = voc_utils.overlay_boxes_without_labels(
                np.array(image), boxlist.bbox.numpy().astype(np.int32)
            )

            ax.imshow(overlay)

        return fig
