from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from python_pascal_voc import datasets
from PIL import Image
import torch
from torchvision.datasets import VisionDataset

"""Implements datasets compliant with CocoDataset interface as used in maskrcnn_benchmark 


References
----------
..[1] https://github.com/facebookresearch/maskrcnn-benchmark
"""


class VOCPartsCropped(VisionDataset):
    def __init__(
        self,
        root,
        dir_cropped_csv,
        dir_Annotations_Part,
        object_class,
        data_split,
        remapping,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        # as you would do normally
        super(VOCPartsCropped, self).__init__(
            root, transforms, transform, target_transform
        )
        self.dset = datasets.FilteredCroppedPascalParts(
            root,
            dir_cropped_csv,
            dir_Annotations_Part,
            object_class,
            data_split,
            remapping,
        )

    def __getitem__(self, idx):
        # load the image as a PIL Image
        example = self.dset[idx]
        image = Image.fromarray(example["image"])

        boxes = example[
            "part_bboxes"
        ]  # boxes ist List[np.ndarray], but we need List[List]
        box_coords = [list(b.coords) for b in boxes]
        box_coords = torch.as_tensor(box_coords).reshape(
            -1, 4
        )  # guard against no boxes
        box_labels = [b.label for b in boxes]

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # and labels
        labels = torch.tensor(box_labels)

        # create a BoxList from the boxes
        boxlist = BoxList(box_coords, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        # TODO: add segmentations

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        example = self.dset[idx]
        image = example["image"]
        img_height = image.shape[0]
        img_width = image.shape[1]
        return {"height": img_height, "width": img_width}

