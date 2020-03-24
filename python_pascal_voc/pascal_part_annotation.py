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
