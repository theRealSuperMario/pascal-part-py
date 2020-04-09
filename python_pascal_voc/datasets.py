import functools
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from python_pascal_voc import voc_utils, pascal_part_annotation
from PIL import Image

import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from python_pascal_voc import pascal_part


class CroppedPascalPartsDataset:
    PART_REMAPPING = {
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
    transforms = None

    def __init__(
        self,
        data_dir,
        dir_cropped_csv,
        split,
        use_difficult=False,
        use_occluded=False,
        use_truncated=False,
    ):
        self.root = data_dir
        self.voc_loader = voc_utils.VOCLoader(self.root)
        self.image_set = "person_" + split
        self.keep_difficult = use_difficult
        self.use_occluded = use_occluded

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._partannopath = os.path.join(
            self.root, "Annotations_Part", "%s.mat"
        )  # TODO: make this correct
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        data_split = voc_utils.DATA_SPLIT[split]
        ids_df = self.voc_loader.load_object_class_cropped(
            voc_utils.ANNOTATION_CLASS.person, data_split, dir_cropped_csv
        )
        if not use_difficult:
            ids_df = ids_df[ids_df.difficult != 1]
        if not use_occluded:
            ids_df = ids_df[ids_df.occluded != 1]
        if not use_truncated:
            ids_df = ids_df[ids_df.truncated != 1]

        self.ids = ids_df.to_dict("records")

        unique_new_labels = list(set(list(self.PART_REMAPPING.keys())))
        if "background" in unique_new_labels:
            unique_new_labels.remove("background")
        unique_new_labels = [
            "background"
        ] + unique_new_labels  # keep background as 0 label

        partindex2partname = dict(enumerate(unique_new_labels))
        partname2partindex = {
            v: k for k, v in dict(enumerate(unique_new_labels)).items()
        }
        self.class_to_ind = partname2partindex
        self.categories = partindex2partname

    def __getitem__(self, index):
        example = self.ids[index]
        fname = example["fname"]
        img = Image.open(self._imgpath % fname).convert("RGB")

        target = self.get_groundtruth(index)
        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        example = self.ids[index]
        fname = example["fname"]
        object_id = example["object_id"]
        anno = ET.parse(self._annopath % fname).getroot()
        anno = self._preprocess_annotation(anno, object_id)
        part_anno = self._preprocess_part_annotation(fname, object_id)

        target = {}
        # target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        # target.add_field("labels", anno["labels"])
        # target.add_field("difficult", anno["difficult"])
        target.update(anno)
        target.update(part_anno)
        return target

    def _preprocess_part_annotation(self, fname, object_id):
        mat_file = self._partannopath % fname
        part_anno = pascal_part_annotation.PartAnnotation.from_mat(mat_file)
        selected_object = part_anno.objects[object_id]

        new_object = pascal_part_annotation.remap_parts(
            selected_object, self.PART_REMAPPING
        )

        part_index_map = {new_object.object_class.value: new_object.partname2partid}
        new_part_segmentation = pascal_part_annotation.get_part_mask(
            [new_object], part_index_map
        )
        new_segmentation = pascal_part_annotation.get_class_mask([new_object])
        new_instance_segmentation = pascal_part_annotation.get_instance_mask(
            [new_object]
        )
        new_anno = pascal_part_annotation.PartAnnotation(
            [new_object],
            new_segmentation,
            new_instance_segmentation,
            new_part_segmentation,
        )

        boxes = [p.bbox for p in new_object.parts]
        gt_classes = [self.class_to_ind[p.part_name] for p in new_object.parts]
        res = {"boxes": boxes, "labels": gt_classes, "part_anno": new_anno}
        return res

    def _preprocess_annotation(self, target, object_id):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        objects = [obj for obj in target.iter("object")]
        target_object = objects[object_id]

        name = target_object.find("name").text.lower().strip()
        bb = target_object.find("bndbox")
        # Make pixel indexes 0-based
        # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
        box = [
            bb.find("xmin").text,
            bb.find("ymin").text,
            bb.find("xmax").text,
            bb.find("ymax").text,
        ]
        bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, box))))

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {"object_bbox": bndbox, "im_info": im_info}
        return res

    def get_img_info(self, index):
        example = self.ids[index]
        img_id = example["fname"]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        pass
        # return PascalVOCDataset.CLASSES[class_id]
