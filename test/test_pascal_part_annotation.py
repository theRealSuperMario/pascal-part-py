import pytest
from python_pascal_voc import datasets, voc_utils, pascal_part_annotation, pascal_part
import pandas as pd
import os
import collections


# class Test_PascalObject:
#     def test_objects(self):
#         dset = datasets.PascalPartDataset(
#             DIR_VOC_ROOT,
#             DIR_ANNOTATIONS_PART,
#             voc_utils.ANNOTATION_CLASS.aeroplane,
#             voc_utils.DATA_SPLIT.train,
#         )
#         ex = dset[0]
#         image_annotation = ex[
#             "annotations_part"
#         ]  # type : pascal_part_annotation.ImageAnnotation
#         assert isinstance(
#             image_annotation.objects[0].object_class, voc_utils.ANNOTATION_CLASS
#         )

#     def test_filter_objects(self):
#         dset = datasets.PascalPartDataset(
#             DIR_VOC_ROOT,
#             DIR_ANNOTATIONS_PART,
#             voc_utils.ANNOTATION_CLASS.aeroplane,
#             voc_utils.DATA_SPLIT.train,
#         )
#         ex = dset[0]
#         image_annotation = ex[
#             "annotations_part"
#         ]  # type : pascal_part_annotation.ImageAnnotation

#         # in this special case, the image_annotation is "aeroplane"
#         assert len(image_annotation.objects) == 1

#         filter_ = lambda x: x.object_class.name in ["bicycle"]
#         image_annotation = pascal_part_annotation.filter_objects(
#             filter_, image_annotation
#         )
#         assert len(image_annotation.objects) == 0


class Test_PartAnnotation:
    def test_objects(self):
        mat_file = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/trainval/Annotations_Part/2008_000008.mat"
        part_anno = pascal_part_annotation.PartAnnotation.from_mat(mat_file)
        fig, _ = part_anno.show()

    def test_remap_parts(self):
        mat_file = "/media/sandro/Volume/datasets/PascalVOC/PascalParts/trainval/Annotations_Part/2008_000008.mat"
        part_anno = pascal_part_annotation.PartAnnotation.from_mat(mat_file)

        obj = part_anno.objects[0]
        remapping = {
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
        new_object = pascal_part_annotation.remap_parts(obj, remapping)
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
        fig, _ = new_anno.show()

