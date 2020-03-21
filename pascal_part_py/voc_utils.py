import pandas as pd
import os
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
import xmltodict


# For backwards support
# https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.items())
    enums["reverse_mapping"] = reverse
    return type("Enum", (), enums)


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

OBJECT_CLASS = enum(
    aeroplane=0,
    bicycle=1,
    bird=2,
    boat=3,
    bottle=4,
    bus=5,
    car=6,
    cat=7,
    chair=8,
    cow=9,
    diningtable=10,
    dog=11,
    horse=12,
    motorbike=13,
    person=14,
    pottedplant=15,
    sheep=16,
    sofa=17,
    train=18,
    tvmonitor=19,
)

OBJECT_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


IMAGE_SET_VOC_2012 = enum("Action", "Main", "Layout", "Segmentation")
from itertools import product


# list of "aeroplain_train", "aeroplain_val", ...
MAIN_IMAGE_SETS = list(product(OBJECT_CLASS_NAMES, ["train", "trainval", "val"]))
MAIN_IMAGE_SETS = list(map(lambda x: "_".join(x), MAIN_IMAGE_SETS))
MAIN_IMAGE_SETS += ["train", "trainval", "val"]
MAIN_IMAGE_SETS = enum(*MAIN_IMAGE_SETS)


class VOCUtils:
    def __init__(self, dir_VOC_root, dir_pascal_csv):
        self.dir_VOC_root = dir_VOC_root
        self.dir_pascal_csv = dir_pascal_csv
        self.dir_JPEGImages = os.path.join(dir_VOC_root, "JPEGImages")
        self.dir_Annotations = os.path.join(dir_VOC_root, "Annotations")
        self.dir_ImageSets = os.path.join(dir_VOC_root, "ImageSets")
        self.dir_ImageSetAction = os.path.join(self.dir_ImageSets, "Action")
        self.dir_ImageSetMain = os.path.join(self.dir_ImageSets, "Main")
        self.dir_ImageSetLayout = os.path.join(self.dir_ImageSets, "Layout")
        self.dir_ImageSetSegmentation = os.path.join(self.dir_ImageSets, "Segmentation")

    @staticmethod
    def OBJECT_CLASSES():
        """
        List all the image sets from Pascal VOC. Don't bother computing
        this on the fly, just remember it. It's faster.
        """
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    @staticmethod
    def SUBSETS():
        return ["train", "val"]

    @property
    def annotation_files(self):
        """ returns absolute paths of annotation xml files """
        files = os.listdir(self.dir_Annotations)
        files = sorted(
            list(map(lambda x: os.path.join(self.dir_Annotations, x), files))
        )
        return files

    @property
    def JPEG_files(self):
        """ returns absolute paths of annotation xml files """
        files = os.listdir(self.dir_Annotations)
        files = sorted(list(map(lambda x: os.path.join(self.dir_JPEGImages, x), files)))
        return files

    @property
    def filenames(self):
        """ returns list of filenames within the dataset (without extension) """
        files = os.listdir(self.dir_Annotations)
        files = sorted(list(map(lambda x: os.path.splitext(x)[0], files)))
        return files

    def imgs_from_category(self, cat_name, dataset):
        """
        Get a list of filenames for images in a particular category as a pandas dataframe.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)

        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.dir_ImageSets, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename, delim_whitespace=True, header=None, names=["filename", "true"]
        )
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Get a list of filenames for images in a particular category
        as a list rather than a pandas dataframe.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)

        Returns:
            list of srings: all filenames from that category
        """
        df = self.imgs_from_category(cat_name, dataset)
        df = df[df["true"] == 1]
        return df["filename"].values

    def annotation_file_from_img(self, img_name):
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
        all_annotations = self._load_all_annotations()
        df = pd.concat(
            [pd.DataFrame.from_dict(a, orient="index") for a in all_annotations], axis=0
        )
        return df

    def load_img(self, img_filename):
        """
        Load image from the filename. Default is to load in color if
        possible.

        Args:
            img_name (string): string of the image name, relative to
                the image directory.

        Returns:
            np array of float32: an image as a numpy array of float32
        """
        img_filename = os.path.join(self.dir_JPEGImages, img_filename + ".jpg")
        img = skimage.img_as_float(io.imread(img_filename)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def load_imgs(self, img_filenames):
        """
        Load a bunch of images from disk as np array.

        Args:
            img_filenames (list of strings): string of the image name, relative to
                the image directory.

        Returns:
            np array of float32: a numpy array of images. each image is
                a numpy array of float32
        """
        return np.array([self.load_img(fname) for fname in img_filenames])

    def _load_data(self, object_class, split=None):
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
        if split is None:
            raise ValueError("Must provide data_type = `train` or `val`")
        filename = os.path.join(
            self.dir_pascal_csv, split + "_" + object_class + ".csv"
        )
        if os.path.isfile(filename):
            return pd.read_csv(filename)
        else:
            # Make data and then return them
            df = self._make_data(object_class, split, filename)
            return df

    def _make_data(self, category, data_type, filename):
        train_img_list = self.imgs_from_category_as_list(category, data_type)
        data = []
        for item in train_img_list:
            anno = self.load_annotation(item)
            objs = anno.findAll("object")
            for obj in objs:
                obj_names = obj.findChildren("name")
                for name_tag in obj_names:
                    if str(name_tag.contents[0]) == category:
                        fname = anno.findChild("filename").contents[0]
                        bbox = obj.findChildren("bndbox")[0]
                        xmin = int(bbox.findChildren("xmin")[0].contents[0])
                        ymin = int(bbox.findChildren("ymin")[0].contents[0])
                        xmax = int(bbox.findChildren("xmax")[0].contents[0])
                        ymax = int(bbox.findChildren("ymax")[0].contents[0])
                        data.append([fname, xmin, ymin, xmax, ymax])
        df = pd.DataFrame(data, columns=["fname", "xmin", "ymin", "xmax", "ymax"])
        df.to_csv(filename)
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
        df = self._load_data(category, split=data_type)
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
        df = self._load_data(cat_name, split=data_type)
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

    def cat_name_to_cat_id(self, cat_name):
        """
        Transform a category name to an id number alphabetically.

        Args:
            cat_name (string): Category name as a string (from list_image_sets())

        Returns:
            int: the integer that corresponds to the category name
        """
        cat_list = self.OBJECT_CLASSES()
        cat_id_dict = dict(zip(cat_list, range(len(cat_list))))
        return cat_id_dict[cat_name]

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
        cat_list = self.OBJECT_CLASSES()
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

