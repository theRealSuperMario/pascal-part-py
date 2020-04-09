# import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from skimage.io import imread
from skimage.measure import regionprops

from python_pascal_voc import voc_utils
from python_pascal_voc.pascal_part import get_pimap
from python_pascal_voc.voc_utils import color_map

PIMAP = get_pimap()
import numpy as np


def filter_objects(func, image_annotation):
    """ callable, ImageAnnotation --> ImageAnnotation 
    Iterate over image_annotation.objects and keep those objects satisfying the condition in `func`

    Examples
    --------

    filter_ = lambda x: x.object_class.name in ["aeroplane", "bicycle"]
    image_annotation = filter_objects(filter_, image_annotation)
    """
    new_objects = list(filter(func, image_annotation.objects))
    new_image_annotation = ImageAnnotation(
        image_annotation.impath,
        image_annotation.annopath,
        image_annotation.im,
        image_annotation.imname,
        new_objects,
    )
    return new_image_annotation


def filter_objects2(func, objects):
    """ callable, ImageAnnotation --> ImageAnnotation 
    Iterate over image_annotation.objects and keep those objects satisfying the condition in `func`

    Examples
    --------

    filter_ = lambda x: x.object_class.name in ["aeroplane", "bicycle"]
    image_annotation = filter_objects(filter_, image_annotation)
    """
    new_objects = list(filter(func, objects))
    return new_objects


class ImageAnnotation(object):
    @classmethod
    def from_file(cls, impath, annopath):
        im = imread(impath)
        data = loadmat(annopath)["anno"][0, 0]
        objects = []
        for obj in data["objects"][0, :]:
            objects.append(PascalObject(obj))
        imname = data["imname"][0]

        return cls(impath, annopath, im, imname, objects)

    def __init__(self, impath, annopath, im, imname, objects):
        # read image
        self.impath = impath
        self.annopath = annopath
        self.im = im
        self.imsize = self.im.shape

        self.imname = imname

        # parse objects and parts
        self.objects = objects
        self.n_objects = len(objects)

        # create masks for objects and parts
        self._mat2map()

    def _mat2map(self):
        """ Create masks from the annotations
        Python implementation based on
        http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz

        Read the annotation and present it in terms of 3 segmentation mask maps (
        i.e., the class maks, instance maks and part mask). pimap defines a
        mapping between part name and index (See part2ind.py).
        """
        shape = self.imsize[:-1]  # first two dimensions, ignore color channel
        self.cls_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
        self.inst_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
        self.part_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
        for i, obj in enumerate(self.objects):
            object_class_index = obj.object_class.value
            mask = obj.mask

            self.inst_mask[mask > 0] = i + 1
            self.cls_mask[mask > 0] = object_class_index

            if obj.n_parts > 0:
                for p in obj.parts:
                    part_name = p.part_name
                    pid = PIMAP[object_class_index][part_name]
                    self.part_mask[p.mask > 0] = pid


class PartAnnotation(object):
    @classmethod
    def from_mat(cls, mat_file):
        data = loadmat(mat_file)["anno"][0, 0]
        objects = []
        for obj in data["objects"][0, :]:
            objects.append(PascalObject(obj))
        return cls(objects)

    def __init__(self, objects):
        # read image
        # parse objects and parts
        self.objects = objects
        self.n_objects = len(objects)

        # create masks for objects and parts
        self._mat2map()

    def _mat2map(self):
        """ Create masks from the annotations
        Python implementation based on
        http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz

        Read the annotation and present it in terms of 3 segmentation mask maps (
        i.e., the class maks, instance maks and part mask). pimap defines a
        mapping between part name and index (See part2ind.py).
        """
        self.cls_mask = None
        self.inst_mask = None
        self.part_mask = None
        for i, obj in enumerate(self.objects):
            object_class_index = obj.object_class.value
            mask = obj.mask
            shape = mask.shape
            if self.cls_mask is None:
                self.cls_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
            if self.inst_mask is None:
                self.inst_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))
            if self.part_mask is None:
                self.part_mask = SemanticAnnotation(np.zeros(shape, dtype=np.uint8))

            self.inst_mask[mask > 0] = i + 1
            self.cls_mask[mask > 0] = object_class_index

            if obj.n_parts > 0:
                for p in obj.parts:
                    part_name = p.part_name
                    pid = PIMAP[object_class_index][part_name]
                    self.part_mask[p.mask > 0] = pid


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
    def __init__(self, obj):
        self.mask = obj["mask"]
        self.props = self._get_region_props()

    def _get_region_props(self):
        """ useful properties
        It includes: area, bbox, bbox_Area, centroid
        It can also extract: filled_image, image, intensity_image, local_centroid
        """
        return regionprops(self.mask)[0]


class PascalObject(PascalBase):
    def __init__(self, obj):
        super(PascalObject, self).__init__(obj)

        self.object_class = voc_utils.ANNOTATION_CLASS[str(obj["class"][0])]
        # type : pascal_part_annotation.ImageAnnotation
        # TODO: why is obj["class"] a numpy array?

        # self.object_class_index = self.object_class.value

        self.n_parts = obj["parts"].shape[1]
        self.parts = []
        if self.n_parts > 0:
            for part in obj["parts"][0, :]:
                self.parts.append(PascalPart(part))


class PascalPart(PascalBase):
    def __init__(self, obj):
        super(PascalPart, self).__init__(obj)
        # TODO: introduce part enumeration for pascal part
        self.part_name = obj["part_name"][0]


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, label=None):
        self.coords = np.array([xmin, ymin, xmax, ymax])
        self.label = label

    @classmethod
    def from_segmentation(cls, segmentation: np.ndarray, label=None):
        """ segmentation must be binary """
        if segmentation.dtype != np.bool:
            raise TypeError("segmentation has to be boolean type")
        props = regionprops(segmentation.astype(np.int32))[0]
        ymin, xmin, ymax, xmax = props.bbox
        return cls(xmin, ymin, xmax, ymax, label)


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
    return SemanticAnnotation(new_part_segmentation)

