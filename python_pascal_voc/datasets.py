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
    # TODO: make remapping into a parameter to feed into the class from outside
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
        self.pascal_part_loader = pascal_part.PascalPartLoader(self.root)
        self.voc_loader = voc_utils.VOCLoader(self.root)
        self.image_set = "person_" + split
        self.keep_difficult = use_difficult
        self.use_occluded = use_occluded

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._partannopath = os.path.join(self.root, "Annotations_Part", "%s.mat")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        data_split = voc_utils.DATA_SPLIT[split]
        ids_df_pascal_parts = self.pascal_part_loader.load_object_class_cropped(
            voc_utils.ANNOTATION_CLASS.person, data_split, dir_cropped_csv
        )
        ids_df_voc = self.voc_loader.load_object_class_cropped(
            voc_utils.ANNOTATION_CLASS.person,
            data_split,
            os.path.join(dir_cropped_csv, "voc"),
        )

        # Do NOT use images with difficult, truncated or occluded objects
        # Also do not use fnames which have some error while iterating the dataset
        # see `test_iterate_dataset_once` for details
        remove_set = set()
        if not use_difficult:
            remove_set.update(ids_df_voc.fname[ids_df_voc.difficult == 1])
        if not use_occluded:
            remove_set.update(ids_df_voc.fname[ids_df_voc.occluded == 1])
        if not use_truncated:
            remove_set.update(ids_df_voc.fname[ids_df_voc.truncated == 1])

        if os.path.exists(os.path.join(dir_cropped_csv, "faulty_fnames.csv")):
            df_faulty = pd.read_csv(
                os.path.join(dir_cropped_csv, "faulty_fnames.csv"), names=["fname"]
            )
            remove_set.update(df_faulty.fname)

        ids_df_pascal_parts = ids_df_pascal_parts[
            np.logical_not(ids_df_pascal_parts.fname.isin(remove_set))
        ]

        self.ids = ids_df_pascal_parts.to_dict("records")

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
        target_values = target

        boxes = target_values["boxes"]
        labels = target_values["labels"]
        object_bbox = target_values["object_bbox"]
        img = img.crop(box=object_bbox)

        offset = np.concatenate([object_bbox[:2], object_bbox[:2]], axis=-1)
        boxes = [b - offset for b in boxes]

        target["boxes"] = boxes

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        example = self.ids[index]
        fname = example["fname"]
        object_id = example["object_id"]
        # anno = ET.parse(self._annopath % fname).getroot()
        # anno = self._preprocess_annotation(anno, object_id)
        part_anno = self._preprocess_part_annotation(fname, object_id)
        im_info = self.get_img_info(index)

        target = {
            "im_info": [im_info["height"], im_info["width"]],
        }
        target.update(part_anno)
        """
            boxes : List[Tuple[int]]
            labels: List[int]
            part_anno : pascal_part_annotation.PartAnnotation
            object_bbox: Tuple[int]
            im_info: [height, width]
        """
        return target

    def _preprocess_part_annotation(self, fname, object_id):
        TO_REMOVE = 1  # bounding box correction by 1 pixel

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
        if len(boxes) == 0:
            boxes = [[]]
        gt_classes = [self.class_to_ind[p.part_name] for p in new_object.parts]

        object_bbox = new_object.bbox

        bndbox = tuple(map(lambda x: x - TO_REMOVE, list(map(int, object_bbox))))
        res = {
            "boxes": boxes,
            "labels": gt_classes,
            "part_anno": new_anno,
            "object_bbox": bndbox,
        }
        return res

    def _preprocess_annotation(self, target, object_id):
        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {"im_info": im_info}
        return res

    def get_img_info(self, index):
        example = self.ids[index]
        img_id = example["fname"]
        xmin = example["xmin"]
        xmax = example["xmax"]
        ymin = example["ymin"]
        ymax = example["ymax"]

        height = ymax - ymin
        width = xmax - xmin
        # height, width
        # Because this dataset returns the cropped objects,
        # we take the bounding box as the image info
        # anno = ET.parse(self._annopath % img_id).getroot()
        # size = anno.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        # return {"height": im_info[0], "width": im_info[1]}

        im_info = {"height": height, "width": width}
        return im_info

    def map_class_id_to_class_name(self, class_id):
        return self.categories[class_id]
