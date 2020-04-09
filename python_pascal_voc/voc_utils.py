import enum
import functools
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xmltodict
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import *


def crop_box(image, xmin, xmax, ymin, ymax):
    """crop image or mask to bbox coordinates
    
    Parameters
    ----------
    image : np.ndarray
        image or mask, shaped [H, W] or [H, W, 3]
    bbox_coords : dict
        {"xmin" : int, "xmax" : int, "ymin" : int, "ymax" : int}
    """
    return image[ymin:ymax, xmin:xmax, ...]


class ANNOTATION_CLASS(enum.Enum):
    """ background + 20 object classes + void class = 21 classes """

    background = 0
    aeroplane = 1
    bicycle = 2
    bird = 3
    boat = 4
    bottle = 5
    bus = 6
    car = 7
    cat = 8
    chair = 9
    cow = 10
    table = 11
    dog = 12
    horse = 13
    motorbike = 14
    person = 15
    pottedplant = 16
    sheep = 17
    sofa = 18
    train = 19
    tvmonitor = 20
    void = 21


ANNOTATION_CLASS_NAMES = [o.name for o in ANNOTATION_CLASS]


OBJECT_CLASSES = [ANNOTATION_CLASS(i) for i in range(1, 21)]  # aeroplane ... tvmonitor
OBJECT_CLASS_NAMES = [o.name for o in OBJECT_CLASSES]


class DATA_SPLIT(enum.Enum):
    train = 0
    trainval = 1
    val = 2


DATA_SPLIT_NAMES = [o.name for o in DATA_SPLIT]


class VOCLoader:
    def __init__(self, dir_VOC_root):
        """Utility class to work with PASCAL VOC20xx dataset
        
        Parameters
        ----------
        dir_VOC_root : str
            path to VOC root, i.e. 'xxx/VOCdevkit/VOC20xx'.
        """

        self.dir_VOC_root = dir_VOC_root
        self.dir_JPEGImages = os.path.join(dir_VOC_root, "JPEGImages")
        self.dir_Annotations = os.path.join(dir_VOC_root, "Annotations")
        self.dir_ImageSets = os.path.join(dir_VOC_root, "ImageSets")
        self.dir_ImageSetAction = os.path.join(self.dir_ImageSets, "Action")
        self.dir_ImageSetMain = os.path.join(self.dir_ImageSets, "Main")
        self.dir_ImageSetLayout = os.path.join(self.dir_ImageSets, "Layout")
        self.dir_ImageSetSegmentation = os.path.join(self.dir_ImageSets, "Segmentation")

        files = os.listdir(self.dir_Annotations)
        files = sorted(list(map(lambda x: os.path.splitext(x)[0], files)))
        self._files = files

    @property
    def annotation_files(self):
        """ returns absolute paths of annotation xml files """
        files = self.filenames
        files = sorted(
            list(map(lambda x: os.path.join(self.dir_Annotations, x + ".xml"), files))
        )
        return files

    @property
    def JPEG_files(self):
        """ returns absolute paths of .jpg image files """
        files = self.filenames
        files = sorted(
            list(map(lambda x: os.path.join(self.dir_JPEGImages, x + ".jpg"), files))
        )
        return files

    @property
    def filenames(self):
        """ returns list of filenames within the dataset (without extension) """
        return self._files

    def load_main_image_set(
        self, image_set,
    ):
        """ load image set file from ImageSets/Main folder.
        
        Parameters
        ----------
        image_set : str
            filename without extentions to image set.
        
        Returns
        -------
        pd.DataFrame
            dataFrame with image set contents
        """
        # TODO: maybe pandas is smart enough so that I don't need the if case here. But I am not sure.
        if image_set in [d.name for d in DATA_SPLIT]:
            "train, val and trainval do not have column which indicates if object class is in image"
            df = pd.read_csv(
                os.path.join(self.dir_ImageSetMain, image_set + ".txt"),
                names=["fname"],
                delim_whitespace=True,
                header=None,
            )
        else:
            df = pd.read_csv(
                os.path.join(self.dir_ImageSetMain, image_set + ".txt"),
                names=["fname", "is_in_image"],
                delim_whitespace=True,
                header=None,
            )
            # only keep fnames where object is in image
            df = df[df.is_in_image == 1]
        return df

    def load_image_set_as_list(self, image_set):
        """
        Get a list of filenames for images in a particular category
        as a list rather than a pandas dataframe.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)

        Returns:
            list of srings: all filenames from that category
        """
        df = self.load_main_image_set(image_set)
        return df.fname.values

    def get_annotationpath_from_fname(self, fname):
        """
        Given an image name `img_name` (without .jpg extensions), get the annotation file for that image.(dir_Annotations/img_name + .xml).

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            string: file path to the annotation file
        """
        return os.path.join(self.dir_Annotations, fname) + ".xml"

    def get_jpegpath_from_fname(self, fname):
        return os.path.join(self.dir_JPEGImages, fname) + ".jpg"

    def load_annotation(self, annotation_file):
        """
        Load annotation file for a given image.

        Args:
            annotation_file (string): abs path to annotation file (.xml)

        Returns:
            BeautifulSoup structure: the annotation labels loaded as a BeautifulSoup data structure
        """
        with open(annotation_file) as fd:
            annotation = xmltodict.parse(fd.read())
        annotation = self._patch_annotation(annotation)
        return annotation

    def _patch_annotation(self, annotation):
        """patch OrderedDict returned by `self.load_annotation` to fit into a standard format

        applied patches:
            * make sure that ["annotation"]["objects"] is a list, even if it contains a single object.
        
        Parameters
        ----------
        annotation : collections.OrderedDict
            annotation as OrderedDict
        
        Returns
        -------
        collections.OrderedDict
            annotation as OrderedDict
        """
        objects = annotation["annotation"]["object"]
        if not isinstance(objects, list):
            annotation["annotation"]["object"] = [objects]
        return annotation

    def _load_all_annotations(self):
        all_annotations = [self.load_annotation(a) for a in self.annotation_files]
        return all_annotations

    def load_all_annotations_as_df(self):
        """ 
        load all annotations and return as a pandas Dataframe. Index individual annotations using df.iloc[i].

        NOTE: loading all annotations into a dataframe takes a lot of time.
        """
        print(
            "loading all annotations into a dataframe takes some time. Consider dumping the dataframe somewhere if you need it more often."
        )
        all_annotations = self._load_all_annotations()
        df = pd.concat(
            [pd.DataFrame.from_dict(a, orient="index") for a in all_annotations], axis=0
        )
        return df

    def load_object_class_cropped(self, object_class, data_split, dir_cropped_csv):
        """Loads each object box annotation individually. Creates intermediate csv files, see `voc_utils.VOCLoader._make_object_class_cropped_data`.

        Parameters
        ----------
        object_class : ANNOTATION_CLASS or None
            object class to use. If `None`, will use entire data split. See `voc_utils.OBJECT_CLASSES` for possible object classes.
        data_split : DATA_SPLIT
            data split to use, i.e. "train", "val", "trainval"
        dir_cropped_csv : str
            path to directory where to store intermediate csv files. Preferrably NOT within VOC subdirectory.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with ["fname", "xmin", "ymin", "xmax", "ymax"] annotations for each individual object annotation.


        See Also
        --------
        .. voc_utils.VOCLoader._make_object_class_cropped_data
        """

        image_set = get_image_set(object_class, data_split)

        filename = os.path.join(dir_cropped_csv, image_set + ".csv")
        if os.path.isfile(filename):
            return pd.read_csv(filename)
        else:
            # Make data and then return them
            df = self._make_object_class_cropped_data(
                object_class, data_split, dir_cropped_csv
            )
            return df

    def load_object_class_cropped_as_list(
        self, object_class, data_split, dir_cropped_csv
    ):
        df = self.load_object_class_cropped(object_class, data_split, dir_cropped_csv)
        return df.to_dict("records")

    def _make_object_class_cropped_data(
        self, object_class, data_split, dir_cropped_csv
    ):
        """Iterate over annotations and collect each object box annotation individually.
        Then save the resulting dataset into a csv file in `dir_cropped_csv`
        
        Parameters
        ----------
        object_class : ANNOTATION_CLASS or None
            object class to use. If `None`, will use entire data split. See `voc_utils.OBJECT_CLASSES` for possible object classes.
        data_split : DATA_SPLIT
            data split to use, i.e. "train", "val", "trainval"
        dir_cropped_csv : str
            path to directory where to store intermediate csv files. Preferrably NOT within VOC subdirectory.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with ["fname", "xmin", "ymin", "xmax", "ymax"] annotations for each individual object annotation.
        """
        image_set = get_image_set(object_class, data_split)

        filename_csv = os.path.join(dir_cropped_csv, image_set + ".csv")
        file_list = self.load_image_set_as_list(image_set)
        data = []

        for fname in file_list:
            annotation_filename = self.get_annotationpath_from_fname(fname)
            anno = self.load_annotation(annotation_filename)

            # Iterate over objects and append each object with filename into dataframe
            objs = anno["annotation"]["object"]
            for object_id, obj in enumerate(objs):
                if object_class is None:
                    # just take all objects
                    bbox = obj["bndbox"]
                    xmin = int(bbox["xmin"])
                    ymin = int(bbox["ymin"])
                    xmax = int(bbox["xmax"])
                    ymax = int(bbox["ymax"])
                    occluded = int(obj["occluded"])
                    truncated = int(obj["truncated"])
                    difficult = int(obj["difficult"])
                    data.append(
                        [
                            fname,
                            xmin,
                            ymin,
                            xmax,
                            ymax,
                            object_id,
                            occluded,
                            truncated,
                            difficult,
                        ]
                    )
                else:
                    if obj["name"] == object_class.name:
                        bbox = obj["bndbox"]
                        xmin = int(bbox["xmin"])
                        ymin = int(bbox["ymin"])
                        xmax = int(bbox["xmax"])
                        ymax = int(bbox["ymax"])
                        occluded = int(obj["occluded"])
                        truncated = int(obj["truncated"])
                        difficult = int(obj["difficult"])
                        data.append(
                            [
                                fname,
                                xmin,
                                ymin,
                                xmax,
                                ymax,
                                object_id,
                                occluded,
                                truncated,
                                difficult,
                            ]
                        )
        df = pd.DataFrame(
            data,
            columns=[
                "fname",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "object_id",
                "occluded",
                "truncated",
                "difficult",
            ],
        )
        df.to_csv(filename_csv, index=False)
        return df


def get_image_set(object_class, data_split):
    """get image set from folder ImageSets/Main from object class and data split
    
    Parameters
    ----------
    object_class : ANNOTATION_CLASS or None
        object class to use. If `None`, will use entire data split. See `voc_utils.OBJECT_CLASSES` for possible object classes.
    data_split : DATA_SPLIT
        data split to use, i.e. "train", "val", "trainval"
    
    Returns
    -------
    str
        filename without .txt extension for Image set
    
    Raises
    ------
    ValueError
        object class cannot be background or void
    """
    if object_class in [ANNOTATION_CLASS.background, ANNOTATION_CLASS.void]:
        raise ValueError("object class cannot be background or void")
    if object_class is not None:
        image_set = "{}_{}".format(object_class.name, data_split.name)
    else:
        # allows to use train, val and trainval without specifying object class.
        # this allows using the entire set to be used.
        image_set = data_split.name
    return image_set


def object_class_name2object_class_id(object_class_name):
    """
    Transform a category name to an id number alphabetically.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())

    Returns:
        int: the integer that corresponds to the category name
    """
    return ANNOTATION_CLASS[object_class_name].value


def batch_to_canvas(X, cols=None):
    """convert batch of images to canvas
    
    Parameters
    ----------
    X : np.ndarray
        tensor of images shaped [N, H, W, C]. Images have to be in range [-1, 1]
    cols : int, optional
        number of columns for the final canvas, by default None
    
    Returns
    -------
    np.ndarray
        canvas with images as grid
    """
    if len(X.shape) == 5:
        # tile
        oldX = np.array(X)
        n_tiles = X.shape[3]
        side = math.ceil(math.sqrt(n_tiles))
        X = np.zeros(
            (oldX.shape[0], oldX.shape[1] * side, oldX.shape[2] * side, oldX.shape[4]),
            dtype=oldX.dtype,
        )
        # cropped images
        for i in range(oldX.shape[0]):
            inx = oldX[i]
            inx = np.transpose(inx, [2, 0, 1, 3])
            X[i] = tile(inx, side, side)
    n_channels = X.shape[3]
    if n_channels > 4:
        X = X[:, :, :, :3]
    if n_channels == 1:
        X = np.tile(X, [1, 1, 1, 3])
    rc = math.sqrt(X.shape[0])
    if cols is None:
        rows = cols = math.ceil(rc)
    else:
        cols = max(1, cols)
        rows = math.ceil(X.shape[0] / cols)
    canvas = tile(X, rows, cols)
    return canvas


def tile(X, rows, cols):
    """Tile images for display.
    
    Parameters
    ----------
    X : np.ndarray
        tensor of images shaped [N, H, W, C]. Images have to be in range [-1, 1]
    rows : int
        number of rows for final canvas
    cols : int
        number of rows for final canvas
    
    Returns
    -------
    np.ndarray
        canvas with images as grid
    """
    tiling = np.zeros((rows * X.shape[1], cols * X.shape[2], X.shape[3]), dtype=X.dtype)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < X.shape[0]:
                img = X[idx, ...]
                tiling[
                    i * X.shape[1] : (i + 1) * X.shape[1],
                    j * X.shape[2] : (j + 1) * X.shape[2],
                    :,
                ] = img
    return tiling


"""
Adapted from: https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae

Python implementation of the color map function for the PASCAL VOC data set.
Official Matlab version can be found in the PASCAL VOC devkit
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
"""


def color_map(N=256, normalized=True, matplotlib=True):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    if matplotlib:
        assert normalized is True
        return LinearSegmentedColormap.from_list("VOClabel_cmap", cmap)
    else:
        return cmap


def color_map_viz():
    labels = ANNOTATION_CLASS_NAMES
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty(
        (row_size * (nclasses + 1), col_size, cmap.shape[1]), dtype=cmap.dtype
    )
    for i in range(nclasses):
        array[i * row_size : i * row_size + row_size, :] = cmap[i]
    array[nclasses * row_size : nclasses * row_size + row_size, :] = cmap[-1]

    plt.imshow(array)
    plt.yticks([row_size * i + row_size / 2 for i in range(nclasses + 1)], labels)
    plt.xticks([])
    plt.show()


def overlay_boxes_without_labels(
    image: np.ndarray,
    bboxes: List[np.ndarray],
    colors: Union[list, np.ndarray, None] = None,
) -> np.ndarray:
    """Adds the predicted boxes on top of the image
    
    Parameters
    ----------
    image : np.ndarray
        an image as returned by OpenCV
    bboxes : List[np.ndarray]
        list of bounding box coordinates. Each bounding box is an np.array of shape 4 with [xmin, ymin, xmax ymax]
    colors : Union[list, np.ndarray, None], optional
        optional list of colors to choose for bounding box overlay, by default will only use red, by default None
    
    Returns
    -------
    np.ndarray
        image with overlaid boxes

    """
    if colors is None:
        # choose red as color for all boxes
        colors = [[255, 0, 0]] * len(bboxes)

    for box, color in zip(bboxes, colors):
        box = box.astype(np.int32)
        top_left, bottom_right = list(box[:2]), list(box[2:])
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image
