#!/usr/bin/env python
import os

import click
import cv2
import matplotlib.pylab as plt
import numpy as np

from python_pascal_voc import datasets, voc_utils
from python_pascal_voc.voc_utils import VOCLoader, crop_box
from python_pascal_voc.voc_utils import color_map

DIR_VOC_ROOT = os.environ["DIR_VOC_ROOT"]
DIR_PASCAL_CSV = os.environ["DIR_PASCAL_CSV"]
DIR_ANNOTATIONS_PART = os.environ["DIR_ANNOTATIONS_PART"]


@click.command()
@click.option("--i", default=0)
@click.option("--object-class", default="horse")
@click.option("--data-split", default="train")
@click.option("--out-path", default="demo.png")
def main(i, object_class, data_split, out_path):
    if object_class == "None":
        object_class = None
    else:
        object_class = voc_utils.ANNOTATION_CLASS[object_class]
    data_split = voc_utils.DATA_SPLIT[data_split]
    image_set = voc_utils.get_image_set(object_class, data_split)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    dset = datasets.PascalVOCDataset(
        DIR_VOC_ROOT, object_class=object_class, data_split=data_split,
    )
    examples = [dset[i] for i in range(i, i + 4)]
    images = [cv2.resize(e["image"], (128, 128)) for e in examples]
    images = np.stack(images, axis=0)

    canvas = voc_utils.batch_to_canvas(images, cols=2)

    axes[0].set_title("PascalVOCDataset - {}, len : {}".format(image_set, len(dset)))
    axes[0].imshow(canvas)
    axes[0].set_axis_off()

    dset = datasets.CroppedPascalVOCDataset(
        DIR_VOC_ROOT, DIR_PASCAL_CSV, object_class=object_class, data_split=data_split,
    )
    examples = [dset[i] for i in range(i, i + 4)]
    images = [cv2.resize(e["image"], (128, 128)) for e in examples]
    images = np.stack(images, axis=0)

    canvas = voc_utils.batch_to_canvas(images, cols=2)

    axes[1].set_title(
        "CroppedPascalVOCDataset - {}, len : {}".format(image_set, len(dset))
    )
    axes[1].imshow(canvas)
    axes[1].set_axis_off()

    #%% Pascal Parts

    dset = datasets.PascalPartDataset(
        DIR_VOC_ROOT,
        DIR_ANNOTATIONS_PART,
        object_class=object_class,
        data_split=data_split,
    )
    examples = [dset[i] for i in range(i, i + 4)]
    images = [cv2.resize(e["image"], (128, 128)) / 255.0 for e in examples]
    images = np.stack(images, axis=0)

    class_segmentation = [
        cv2.resize(e["class_segmentation"].as_rgb(), (128, 128), cv2.INTER_NEAREST)
        for e in examples
    ]
    class_segmentation = np.stack(class_segmentation, axis=0)

    instance_segmentation = [
        cv2.resize(e["instance_segmentation"].as_rgb(), (128, 128), cv2.INTER_NEAREST)
        for e in examples
    ]
    instance_segmentation = np.stack(instance_segmentation, axis=0)

    part_segmentation = [
        cv2.resize(e["part_segmentation"].as_rgb(), (128, 128), cv2.INTER_NEAREST)
        for e in examples
    ]
    part_segmentation = np.stack(part_segmentation, axis=0)

    canvas = voc_utils.batch_to_canvas(
        np.concatenate(
            [images, class_segmentation, instance_segmentation, part_segmentation],
            axis=0,
        ),
        cols=4,
    )

    axes[2].set_title(
        "PascalPartDataset - {}, len : {}".format(image_set, len(dset))
        + "\n"
        + "image, \n class_segmentation, \n instance_segmentation, \n part_segmentation"
    )
    axes[2].imshow(canvas)
    axes[2].set_axis_off()

    #%% Cropped Pascal Parts

    dset = datasets.CroppedPascalPartDataset(
        DIR_VOC_ROOT,
        DIR_PASCAL_CSV,
        DIR_ANNOTATIONS_PART,
        object_class=object_class,
        data_split=data_split,
    )
    examples = [dset[i] for i in range(i, i + 4)]
    images = [cv2.resize(e["image"], (128, 128)) / 255.0 for e in examples]
    images = np.stack(images, axis=0)

    class_segmentation = [
        cv2.resize(e["class_segmentation"].as_rgb(), (128, 128), cv2.INTER_NEAREST)
        for e in examples
    ]
    class_segmentation = np.stack(class_segmentation, axis=0)

    instance_segmentation = [
        cv2.resize(e["instance_segmentation"].as_rgb(), (128, 128), cv2.INTER_NEAREST)
        for e in examples
    ]
    instance_segmentation = np.stack(instance_segmentation, axis=0)

    part_segmentation = [
        cv2.resize(e["part_segmentation"].as_rgb(), (128, 128), cv2.INTER_NEAREST)
        for e in examples
    ]
    part_segmentation = np.stack(part_segmentation, axis=0)

    canvas = voc_utils.batch_to_canvas(
        np.concatenate(
            [images, class_segmentation, instance_segmentation, part_segmentation],
            axis=0,
        ),
    )

    axes[3].set_title(
        "CroppedPascalPartDataset - {}, len : {}".format(image_set, len(dset))
        + "\n"
        + "image, \n class_segmentation, \n instance_segmentation, \n part_segmentation"
    )
    axes[3].imshow(canvas)
    axes[3].set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == "__main__":
    main()
