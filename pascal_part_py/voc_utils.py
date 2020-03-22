import enum
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import xmltodict
from more_itertools import unique_everseen
from skimage import io


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


class OBJECT_CLASS(enum.Enum):
    aeroplane = 0
    bicycle = 1
    bird = 2
    boat = 3
    bottle = 4
    bus = 5
    car = 6
    cat = 7
    chair = 8
    cow = 9
    diningtable = 10
    dog = 11
    horse = 12
    motorbike = 13
    person = 14
    pottedplant = 15
    sheep = 16
    sofa = 17
    train = 18
    tvmonitor = 19


OBJECT_CLASS_NAMES = [o.name for o in OBJECT_CLASS]


class DATA_SPLIT(enum.Enum):
    train = 0
    trainval = 1
    val = 2


DATA_SPLIT_NAMES = [o.name for o in DATA_SPLIT]


# 1. Docstrings
# 2. Pascal Parts dataset


class PascalVOCDataset:
    def __init__(self, dir_VOC_root, object_class, data_split):
        """Dataset class for PASCAL VOC 20xx. 
        Iterates over single images. Annotations contain bounding boxes for objects in the image. 
        There can be multiple objects in an image, therefore annotations may contain multiple object annotations.
        Returns only annotations, not images. Inherit your own subclass for data loading.

        For instance, Pascal VOC2010 train set contains 4998 images, thus the length of this dataset is 4998

        See examples and demo for further usage.
        
        Parameters
        ----------
        dir_VOC_root : str
            path to VOC root, i.e. 'xxx/VOCdevkit/VOC20xx'.
        object_class : OBJECT_CLASS or None
            object class to use. If `None`, will use entire data split.
        data_split : DATA_SPLIT
            data split to use, i.e. "train", "val", "trainval"


        Examples
        --------

            # Load only aeroplane class from train split
            dset = voc_utils.PascalVOCDataset(
                DIR_VOC_ROOT, voc_utils.OBJECT_CLASS.aeroplane, voc_utils.DATA_SPLIT.train
            )
            assert len(dset) == 283  # pascal VOC 2010

            # load entire train set
            dset = voc_utils.PascalVOCDataset(
                DIR_VOC_ROOT, None, voc_utils.DATA_SPLIT.train
            )
            assert len(dset) == 4998  # pascal VOC 2010

        """
        self.voc = VOCLoader(dir_VOC_root)
        self.image_set = self.voc.get_image_set(object_class, data_split)
        self.files = self.voc.load_image_set_as_list(self.image_set)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        annotation_file = self.voc.get_annotationpath_from_fname(fname)
        annotation = self.voc.load_annotation(annotation_file)
        return annotation


class CroppedPascalVOC:
    def __init__(self, dir_VOC_root, dir_cropped_csv, object_class, data_split):
        """Dataset class for iterating over every single annotated object box in the Pascal dataset.
        
        In the Pascal dataset, annotations per image can contain multiple object bounding boxes.
        If you want to crop every object out of the image and iterate over those crops, use this class.

        For instance, Pascal VOC2010 train set contains 4998 images, but 13339 annotated objects. 
        Thus the length of this dataset is 13339.

        You can filter by `object_class` and data split.

        
        To prevent figuring out the object boxes every time this class is instantiated,
        the data is stored in separate csv files in `dir_cropped_csv` and reloaded.
        I recommend NOT setting `dir_cropped_csv` to a subdir within the vOC dataset, but 
        to some other path.

        
        Parameters
        ----------
        dir_VOC_root : str
            path to VOC root, i.e. 'xxx/VOCdevkit/VOC20xx'.
        dir_cropped_csv : str
            path to directory where to store intermediate csv files. Preferrably NOT within VOC subdirectory.
        object_class : OBJECT_CLASS or None
            object class to use. If `None`, will use entire data split.
        data_split : DATA_SPLIT
            data split to use, i.e. "train", "val", "trainval"

        Examples
        --------

            csv_dir = tmpdir.mkdir("csv")
            dset = voc_utils.CroppedPascalVOC(
                DIR_VOC_ROOT,
                csv_dir,
                voc_utils.OBJECT_CLASS.aeroplane,
                voc_utils.DATA_SPLIT.train,
            )
            ex = dset[0]
            assert len(dset) == 403  # pascal VOC 2010
        """
        self.voc = VOCLoader(dir_VOC_root)
        self.dir_cropped_csv = dir_cropped_csv
        self.files = self.voc.load_object_class_cropped_as_list(
            object_class, data_split, dir_cropped_csv
        )
        # files is a list of {"fname" : xxx.jpg, "xmin" : xmin, "ymin" : ymin, ...}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.files[i]


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
        """ 
        patch OrderedDict returned by `self.load_annotation` to fit into a standard format

        applied patches:
            * make sure that ["annotation"]["objects"] is a list, even if it contains a single object.
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
        """
        Loads all the data as a pandas DataFrame for a particular category.

        Args:
            category (string): Category name as a string (from list_image_sets())
            data_type (string, optional): "train" or "val"

        Raises:
            ValueError: when you don't give "train" or "val" as data_type

        Returns:
            pandas DataFrame: df of filenames and bounding boxes
        """

        image_set = self.get_image_set(object_class, data_split)

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

    def get_image_set(self, object_class, data_split):
        if object_class is not None:
            image_set = "{}_{}".format(object_class.name, data_split.name)
        else:
            # allows to use train, val and trainval without specifying object class.
            # this allows using the entire set to be used.
            image_set = data_split.name
        return image_set

    def _make_object_class_cropped_data(
        self, object_class, data_split, dir_cropped_csv
    ):
        """Iterate over annotations and collect each object box annotation individually.
        Then save the resulting dataset into a csv file in `dir_cropped_csv`
        
        Parameters
        ----------
        object_class : OBJECT_CLASS or None
            object class to use. If `None`, will use entire data split.
        data_split : DATA_SPLIT
            data split to use, i.e. "train", "val", "trainval"
        dir_cropped_csv : str
            path to directory where to store intermediate csv files. Preferrably NOT within VOC subdirectory.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with ["fname", "xmin", "ymin", "xmax", "ymax"] annotations for each individual object annotation.
        """
        image_set = self.get_image_set(object_class, data_split)

        filename_csv = os.path.join(dir_cropped_csv, image_set + ".csv")
        file_list = self.load_image_set_as_list(image_set)
        data = []

        for fname in file_list:
            annotation_filename = self.get_annotationpath_from_fname(fname)
            anno = self.load_annotation(annotation_filename)

            # Iterate over objects and append each object with filename into dataframe
            objs = anno["annotation"]["object"]
            for obj in objs:
                if object_class is None:
                    # just take all objects
                    bbox = obj["bndbox"]
                    xmin = bbox["xmin"]
                    ymin = bbox["ymin"]
                    xmax = bbox["xmax"]
                    ymax = bbox["ymax"]
                    data.append([fname, xmin, ymin, xmax, ymax])
                else:
                    if obj["name"] == object_class.name:
                        bbox = obj["bndbox"]
                        xmin = bbox["xmin"]
                        ymin = bbox["ymin"]
                        xmax = bbox["xmax"]
                        ymax = bbox["ymax"]
                        data.append([fname, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(data, columns=["fname", "xmin", "ymin", "xmax", "ymax"])
        df.to_csv(filename_csv)
        return df


def object_class_name2object_class_id(object_class_name):
    """
    Transform a category name to an id number alphabetically.

    Args:
        cat_name (string): Category name as a string (from list_image_sets())

    Returns:
        int: the integer that corresponds to the category name
    """
    return OBJECT_CLASS[object_class_name].value

