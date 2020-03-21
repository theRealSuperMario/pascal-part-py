#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pylab as plt
from skimage.io import imread
from pascal_part_py.VOClabelcolormap import color_map
from pascal_part_py.anno import ImageAnnotation

from pascal_part_py.datasets import PascalPartDataset, CroppedPascalPartDataset
from pascal_part_py.voc_utils import PascalVOCDataset, crop_box
import click

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_PASCAL_CSV = os.environ["DIR_PASCAL_CSV"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


@click.command()
@click.option("-i", default=0)
@click.option("-image-set", default="horse")
@click.option("-split", default="train")
def main(i, image_set, split):
    dset = PascalPartDataset(
        DIR_VOC_ROOT,
        DIR_PASCAL_CSV,
        DIR_ANNOTATIONS_PART,
        data_type=split,
        category=image_set,
    )
    ex = dset[i]

    bbox_coords = ex["bbox"]

    an = ex["annotations_part"]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(an.im)
    ax1.set_title("Image")
    ax1.axis("off")
    ax2.imshow(an.cls_mask, cmap=color_map(N=np.max(an.cls_mask) + 1))
    ax2.set_title("Class mask")
    ax2.axis("off")
    ax3.imshow(an.inst_mask, cmap=color_map(N=np.max(an.inst_mask) + 1))
    ax3.set_title("Instance mask")
    ax3.axis("off")
    if np.max(an.part_mask) == 0:
        ax4.imshow(an.part_mask, cmap="gray")
    else:
        ax4.imshow(an.part_mask, cmap=color_map(N=np.max(an.part_mask) + 1))
    ax4.set_title("Part mask")
    ax4.axis("off")
    plt.savefig("pascal_part.png")

    dset = CroppedPascalPartDataset(
        DIR_VOC_ROOT,
        DIR_PASCAL_CSV,
        DIR_ANNOTATIONS_PART,
        data_type=split,
        category=image_set,
    )
    ex = dset[i]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(ex["image"])
    ax1.set_title("Image")
    ax1.axis("off")
    ax2.imshow(ex["class_mask"], cmap=color_map(N=np.max(an.cls_mask) + 1))
    ax2.set_title("Class mask")
    ax2.axis("off")
    ax3.imshow(
        ex["instance_mask"], cmap=color_map(N=np.max(an.inst_mask) + 1),
    )
    ax3.set_title("Instance mask")
    ax3.axis("off")
    part_mask = ex["part_mask"]
    if np.max(part_mask) == 0:
        ax4.imshow(part_mask, cmap="gray")
    else:
        ax4.imshow(part_mask, cmap=color_map(N=np.max(part_mask) + 1))
    ax4.set_title("Part mask")
    ax4.axis("off")
    plt.savefig("pascal_part_cropped.png")


if __name__ == "__main__":
    main()
