import pandas as pd
import os
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import xmltodict
import enum


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


# TODO: Filter out individual object classes
# -- some images contain multiple objects. If you want to train on individual croopped images, you need to add the images multiple times int he training set insted of once.
# image sets files only indicate if an image contains an object class or not, not how many
# therefore, we still need to build the custom csv files
# TODO: remove enum. It is too complicated


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

from itertools import product


class DATA_SPLITS(enum.Enum):
    train = 0
    trainval = 1
    val = 2


DATA_SPLIT_NAMES = [o.name for o in DATA_SPLITS]


class PascalVOCDataset:
    def __init__(self, dir_VOC_root, object_class, data_split):
        self.voc = VOCUtils(dir_VOC_root)
        self.image_set = self.voc.get_image_set(object_class, data_split)
        self.files = self.voc.load_image_set_as_list(self.image_set)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        annotation_file = self.voc.get_annotationpath_from_fname(fname)
        annotation = self.voc.load_annotation(annotation_file)
        return annotation


class CroppedPascalVOC(PascalVOCDataset):
    def __init__(self, dir_VOC_root, dir_cropped_csv, object_class, data_split):
        self.voc = VOCUtils(dir_VOC_root)
        self.dir_cropped_csv = dir_cropped_csv
        self.files = self.voc.load_object_class_cropped_as_list(
            object_class, data_split, dir_cropped_csv
        )
        # files is a list of {"fname" : xxx.jpg, "bbox" : {"xmin" : xmin, "ymin" : ymin}}

    def __getitem__(self, i):
        return self.files[i]


class VOCUtils:
    def __init__(self, dir_VOC_root):
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
        """ returns absolute paths of annotation xml files """
        files = self.filenames
        files = sorted(
            list(map(lambda x: os.path.join(self.dir_JPEGImages, x + ".jpg"), files))
        )
        return files

    @property
    def filenames(self):
        """ returns list of filenames within the dataset (without extension) """
        return self._files

    def load_image_set(
        self, image_set,
    ):
        # TODO: maybe pandas is smart enough so that I don't need the if case here. But I am not sure.
        if image_set in [d.name for d in DATA_SPLITS]:
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
        df = self.load_image_set(image_set)
        return df.fname.values

    def get_annotationpath_from_fname(self, img_name):
        """
        Given an image name `img_name` (without .jpg extensions), get the annotation file for that image.(dir_Annotations/img_name + .xml).

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            string: file path to the annotation file
        """
        return os.path.join(self.dir_Annotations, img_name) + ".xml"

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
            # allows to use train, val and trainval without specifying object class, to entire set is used.
            image_set = data_split.name
        return image_set

    def _make_object_class_cropped_data(
        self, object_class, data_split, dir_cropped_csv
    ):
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

    def get_image_url_list(self, category, data_type=None):
        """
        For a given data type, returns a list of filenames.

        Args:
            category (string): Category name as a string (from list_image_sets())
            data_type (string, optional): "train" or "val"

        Returns:
            list of strings: list of all filenames for that particular category
        """
        df = self.load_object_class_cropped(category, split=data_type)
        image_url_list = list(unique_everseen(list(self.dir_JPEGImages + df["fname"])))
        return image_url_list

    def get_masks(self, cat_name, data_type, mask_type=None):
        """
        Return a list of masks for a given category and data_type.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            data_type (string, optional): "train" or "val"
            mask_type (string, optional): either "bbox1" or "bbox2" - whether to
                sum or add the masks for multiple objects

        Raises:
            ValueError: if mask_type is not valid

        Returns:
            list of np arrays: list of np arrays that are masks for the images
                in the particular category.
        """
        # change this to searching through the df
        # for the bboxes instead of relying on the order
        # so far, should be OK since I'm always loading
        # the df from disk anyway
        # mask_type should be bbox1 or bbox
        if mask_type is None:
            raise ValueError("Must provide mask_type")
        df = self.load_object_class_cropped(cat_name, split=data_type)
        # load each image, turn into a binary mask
        masks = []
        prev_url = ""
        blank_img = None
        for row_num, entry in df.iterrows():
            img_url = os.path.join(self.dir_JPEGImages, entry["fname"])
            if img_url != prev_url:
                if blank_img is not None:
                    # TODO: options for how to process the masks
                    # make sure the mask is from 0 to 1
                    max_val = blank_img.max()
                    if max_val > 0:
                        min_val = blank_img.min()
                        # print "min val before normalizing: ", min_val
                        # start at zero
                        blank_img -= min_val
                        # print "max val before normalizing: ", max_val
                        # max val at 1
                        blank_img /= max_val
                    masks.append(blank_img)
                prev_url = img_url
                img = self.load_img(img_url)
                blank_img = np.zeros((img.shape[0], img.shape[1], 1))
            bbox = [entry["xmin"], entry["ymin"], entry["xmax"], entry["ymax"]]
            if mask_type == "bbox1":
                blank_img[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 1.0
            elif mask_type == "bbox2":
                blank_img[bbox[1] : bbox[3], bbox[0] : bbox[2]] += 1.0
            else:
                raise ValueError("Not a valid mask type")
        # TODO: options for how to process the masks
        # make sure the mask is from 0 to 1
        max_val = blank_img.max()
        if max_val > 0:
            min_val = blank_img.min()
            # print "min val before normalizing: ", min_val
            # start at zero
            blank_img -= min_val
            # print "max val before normalizing: ", max_val
            # max val at 1
            blank_img /= max_val
        masks.append(blank_img)
        return np.array(masks)

    def get_imgs(self, cat_name, data_type=None):
        """
        Load and return all the images for a particular category.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            data_type (string, optional): "train" or "val"

        Returns:
            np array of images: np array of loaded images for the category
                and data_type.
        """
        image_url_list = self.get_image_url_list(cat_name, data_type=data_type)
        imgs = []
        for url in image_url_list:
            imgs.append(self.load_img(url))
        return np.array(imgs)

    def display_image_and_mask(self, img, mask):
        """
        Display an image and it's mask side by side.

        Args:
            img (np array): the loaded image as a np array
            mask (np array): the loaded mask as a np array
        """
        plt.figure(1)
        plt.clf()
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.imshow(img)
        ax1.set_title("Original image")
        ax2.imshow(mask)
        ax2.set_title("Mask")
        plt.show(block=False)

    def cat_name_to_cat_id(self, object_class):
        """
        Transform a category name to an id number alphabetically.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())

        Returns:
            int: the integer that corresponds to the category name
        """
        return OBJECT_CLASS_NAMES.index[object_class]

    def display_img_and_masks(self, img, true_mask, predicted_mask, block=False):
        """
        Display an image and it's two masks side by side.

        Args:
            img (np array): image as a np array
            true_mask (np array): true mask as a np array
            predicted_mask (np array): predicted_mask as a np array
            block (bool, optional): whether to display in a blocking manner or not.
                Default to False (non-blocking)
        """
        m_predicted_color = predicted_mask.reshape(
            predicted_mask.shape[0], predicted_mask.shape[1]
        )
        m_true_color = true_mask.reshape(true_mask.shape[0], true_mask.shape[1])
        # m_predicted_color = predicted_mask
        # m_true_color = true_mask
        # plt.close(1)
        plt.figure(1)
        plt.clf()
        plt.axis("off")
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, num=1)
        # f.clf()
        ax1.get_xaxis().set_ticks([])
        ax2.get_xaxis().set_ticks([])
        ax3.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])

        ax1.imshow(img)
        ax2.imshow(m_true_color)
        ax3.imshow(m_predicted_color)
        plt.draw()
        plt.show(block=block)

    def load_data_multilabel(self, data_type=None):
        """
        Returns a data frame for all images in a given set in multilabel format.

        Args:
            data_type (string, optional): "train" or "val"

        Returns:
            pandas DataFrame: filenames in multilabel format
        """
        if data_type is None:
            raise ValueError("Must provide data_type = train or val")
        filename = os.path.join(self.dir_ImageSets, data_type + ".txt")
        cat_list = OBJECT_CLASS_NAMES
        df = pd.read_csv(
            filename, delim_whitespace=True, header=None, names=["filename"]
        )
        # add all the blank rows for the multilabel case
        for cat_name in cat_list:
            df[cat_name] = 0
        for info in df.itertuples():
            index = info[0]
            fname = info[1]
            anno = self.load_annotation(fname)
            objs = anno.findAll("object")
            for obj in objs:
                obj_names = obj.findChildren("name")
                for name_tag in obj_names:
                    tag_name = str(name_tag.contents[0])
                    if tag_name in cat_list:
                        df.at[index, tag_name] = 1
        return df

