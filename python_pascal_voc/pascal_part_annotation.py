# import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from skimage.io import imread
from skimage.measure import regionprops

from python_pascal_voc import voc_utils
from python_pascal_voc.pascal_part import get_pimap
from python_pascal_voc.voc_utils import color_map

PART_INDEX_MAP = get_pimap()
import numpy as np
from typing import *


class PartAnnotation(object):
    cls_mask = None
    inst_mask = None
    part_mask = None
    objects = None

    @classmethod
    def from_mat(cls, mat_file):
        data = loadmat(mat_file)["anno"][0, 0]
        objects = []
        for obj in data["objects"][0, :]:
            objects.append(PascalObject.from_mat_data(obj))

        cls_mask = get_class_mask(objects)
        inst_mask = get_instance_mask(objects)
        part_mask = get_part_mask(objects, PART_INDEX_MAP)
        return cls(objects, cls_mask, inst_mask, part_mask)

    def __init__(self, objects, cls_mask, inst_mask, part_mask):
        # read image
        # parse objects and parts
        self.objects = objects

        # create masks for objects and parts
        # self._mat2map()
        self.cls_mask = cls_mask
        self.inst_mask = inst_mask
        self.part_mask = part_mask

    def show(self, fig=None, ax1=None, ax2=None, ax3=None):
        if ax1 is None or ax2 is None or ax3 is None or fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.ravel()
            ax1, ax2, ax3 = axes[:3]

        axes[0].imshow(self.cls_mask.as_rgb())
        axes[1].imshow(self.inst_mask.as_rgb())
        axes[2].imshow(self.part_mask.as_rgb())
        return fig, (ax1, ax2, ax3)


def get_class_mask(objects):
    cls_mask = None
    for i, obj in enumerate(objects):
        object_class_index = obj.object_class.value
        mask = obj.mask
        shape = mask.shape
        if cls_mask is None:
            cls_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
        cls_mask[mask > 0] = object_class_index
    return cls_mask


def get_instance_mask(objects):
    inst_mask = None
    for i, obj in enumerate(objects):
        mask = obj.mask
        shape = mask.shape
        if inst_mask is None:
            inst_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
        inst_mask[mask > 0] = i + 1
    return inst_mask


def get_part_mask(objects, PART_INDEX_MAP):
    part_mask = None
    for i, obj in enumerate(objects):
        mask = obj.mask
        shape = mask.shape
        object_class_index = obj.object_class.value
        if len(obj.parts) > 0:
            for p in obj.parts:
                part_name = p.part_name
                pid = PART_INDEX_MAP[object_class_index][part_name]
                if part_mask is None:
                    part_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
                part_mask[p.mask > 0] = pid
    return part_mask


class SemanticAnnotation(np.ndarray):
    def as_rgb(self):
        colors = color_map(N=np.max(self) + 1)

        return colors(self / np.max(self))[..., :3]  # no alpha

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return


class PascalBase(object):
    mask: SemanticAnnotation = None
    props = None
    bbox = None

    @classmethod
    def from_mat_data(cls, data):
        mask = data["mask"]
        mask = SemanticAnnotation(mask)
        return cls(mask)

    def __init__(self, mask):
        self.mask = mask
        self.props = self._get_region_props()
        ymin, xmin, ymax, xmax = self.props.bbox
        self.bbox = (xmin, ymin, xmax, ymax)

    def _get_region_props(self):
        """ useful properties
        It includes: area, bbox, bbox_Area, centroid
        It can also extract: filled_image, image, intensity_image, local_centroid
        """
        return regionprops(self.mask)[0]


class PascalPart(PascalBase):
    part_name = None
    part_index = None

    def __init__(self, mask, partname):
        super(PascalPart, self).__init__(mask)
        self.part_name = partname

    @classmethod
    def from_mat_data(cls, data):
        mask = data["mask"]
        part_name = data["part_name"][0]
        return cls(mask, part_name)


def parts_from_part_segmentation(part_segmentation, partindex2partname):
    """ --> List[parts] """
    # iterate over part_segmentation_labels and generate parts
    parts = []
    for u in np.unique(part_segmentation):
        if u == 0:
            continue  # skip background label
        mask = (part_segmentation == u) * 1
        part_name = partindex2partname[u]
        part = PascalPart(mask, part_name)
        parts.append(part)
    return parts


def remap_parts(obj, remapping: dict):
    part_segmentation = get_part_mask([obj], PART_INDEX_MAP)
    (
        remapped_part_segmentation,
        partindex2partname,
        partname2partindex,
    ) = remap_part_segmentation(part_segmentation, remapping)

    parts = parts_from_part_segmentation(remapped_part_segmentation, partindex2partname)
    new_object = PascalObject(
        obj.mask, obj.object_class, parts, partname2partindex, partindex2partname
    )
    return new_object


def remap_part_segmentation(part_segmentation, remapping):
    unique_new_labels = list(set(list(remapping.keys())))
    if "background" in unique_new_labels:
        unique_new_labels.remove("background")
    unique_new_labels = ["background"] + unique_new_labels  # keep background as 0 label
    part_maps = {p: part_segmentation == p for p in np.unique(part_segmentation)}
    new_part_segmentation = np.zeros_like(part_segmentation)
    for new_label, parts in remapping.items():
        for p in parts:
            if p.value in np.unique(part_segmentation):
                new_part_segmentation[part_maps[p.value]] = unique_new_labels.index(
                    new_label
                )
            else:
                pass
    partindex2partname = dict(enumerate(unique_new_labels))
    partname2partindex = {v: k for k, v in dict(enumerate(unique_new_labels)).items()}
    return (
        SemanticAnnotation(new_part_segmentation),
        partindex2partname,
        partname2partindex,
    )


class PascalObject(PascalBase):
    object_class: voc_utils.ANNOTATION_CLASS = None
    parts: List[PascalPart] = None
    partname2partid: dict = None
    partid2partname: dict = None

    def __init__(self, mask, object_class, parts, partname2partid, partid2partname):
        super(PascalObject, self).__init__(mask)

        self.object_class = object_class
        self.parts = parts
        self.partid2partname = partid2partname
        self.partname2partid = partname2partid

    @classmethod
    def from_mat_data(cls, data):
        mask = data["mask"]
        object_class = voc_utils.ANNOTATION_CLASS[str(data["class"][0])]
        n_parts = data["parts"].shape[1]
        parts = []
        if n_parts > 0:
            for part_data in data["parts"][0, :]:
                parts.append(PascalPart.from_mat_data(part_data))

        partname2partid = PART_INDEX_MAP[object_class.value]
        partid2partname = {v: k for k, v in PART_INDEX_MAP[object_class.value].items()}
        return cls(mask, object_class, parts, partname2partid, partid2partname)

