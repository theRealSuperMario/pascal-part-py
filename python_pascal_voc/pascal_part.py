""" Get class names and part names associated with each class
Python implementation based on
http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz

Define the part index of each objects.
One can merge different parts by using the same index for the
parts that are desired to be merged.
For example, one can merge
the left lower leg (llleg) and the left upper leg (luleg) of person by setting:
pimap[15]['llleg']      = 19;               # left lower leg
pimap[15]['luleg']      = 19;               # left upper leg
"""
from python_pascal_voc import voc_utils
from python_pascal_voc.voc_utils import ANNOTATION_CLASS
import numpy as np
import os


def get_class_names():
    """ exclude void and background annotation """
    classes = {}
    for o in voc_utils.ANNOTATION_CLASS:
        if o not in [
            voc_utils.ANNOTATION_CLASS.void,
            voc_utils.ANNOTATION_CLASS.background,
        ]:
            classes[o.value] = o.name
    return classes


def get_pimap():
    part_index_map = {}

    # [aeroplane]
    v = ANNOTATION_CLASS.aeroplane.value
    part_index_map[v] = {}
    part_index_map[v]["body"] = 1
    part_index_map[v]["stern"] = 2
    part_index_map[v]["lwing"] = 3  # left wing
    part_index_map[v]["rwing"] = 4  # right wing
    part_index_map[v]["tail"] = 5
    for ii in range(1, 10 + 1):
        part_index_map[v][("engine_%d" % ii)] = 10 + ii  # multiple engines
    for ii in range(1, 10 + 1):
        part_index_map[v][("wheel_%d" % ii)] = 20 + ii  # multiple wheels

    # [bicycle]
    v = ANNOTATION_CLASS.bicycle.value
    part_index_map[v] = {}
    part_index_map[v]["fwheel"] = 1  # front wheel
    part_index_map[v]["bwheel"] = 2  # back wheel
    part_index_map[v]["saddle"] = 3
    part_index_map[v]["handlebar"] = 4  # handle bar
    part_index_map[v]["chainwheel"] = 5  # chain wheel
    for ii in range(1, 10 + 1):
        part_index_map[v][("headlight_%d" % ii)] = 10 + ii

    # [bird]
    v = ANNOTATION_CLASS.bird.value
    part_index_map[v] = {}
    part_index_map[v]["head"] = 1
    part_index_map[v]["leye"] = 2  # left eye
    part_index_map[v]["reye"] = 3  # right eye
    part_index_map[v]["beak"] = 4
    part_index_map[v]["torso"] = 5
    part_index_map[v]["neck"] = 6
    part_index_map[v]["lwing"] = 7  # left wing
    part_index_map[v]["rwing"] = 8  # right wing
    part_index_map[v]["lleg"] = 9  # left leg
    part_index_map[v]["lfoot"] = 10  # left foot
    part_index_map[v]["rleg"] = 11  # right leg
    part_index_map[v]["rfoot"] = 12  # right foot
    part_index_map[v]["tail"] = 13

    # [boat]
    # only has silhouette mask
    part_index_map[ANNOTATION_CLASS.boat.value] = {}

    # [bottle]
    v = ANNOTATION_CLASS.bottle.value
    part_index_map[v] = {}
    part_index_map[v]["cap"] = 1
    part_index_map[v]["body"] = 2

    # [bus]
    v = ANNOTATION_CLASS.bus.value
    part_index_map[v] = {}
    part_index_map[v]["frontside"] = 1
    part_index_map[v]["leftside"] = 2
    part_index_map[v]["rightside"] = 3
    part_index_map[v]["backside"] = 4
    part_index_map[v]["roofside"] = 5
    part_index_map[v]["leftmirror"] = 6
    part_index_map[v]["rightmirror"] = 7
    part_index_map[v]["fliplate"] = 8  # front license plate
    part_index_map[v]["bliplate"] = 9  # back license plate
    for ii in range(1, 10 + 1):
        part_index_map[6][("door_%d" % ii)] = 10 + ii
    for ii in range(1, 10 + 1):
        part_index_map[6][("wheel_%d" % ii)] = 20 + ii
    for ii in range(1, 10 + 1):
        part_index_map[6][("headlight_%d" % ii)] = 30 + ii
    for ii in range(1, 20 + 1):
        part_index_map[6][("window_%d" % ii)] = 40 + ii

    # [car]
    part_index_map[ANNOTATION_CLASS.car.value] = part_index_map[
        ANNOTATION_CLASS.bus.value
    ].copy()  # car has the same set of parts with bus

    # [cat]
    v = ANNOTATION_CLASS.cat.value
    part_index_map[v] = {}
    part_index_map[v]["head"] = 1
    part_index_map[v]["leye"] = 2  # left eye
    part_index_map[v]["reye"] = 3  # right eye
    part_index_map[v]["lear"] = 4  # left ear
    part_index_map[v]["rear"] = 5  # right ear
    part_index_map[v]["nose"] = 6
    part_index_map[v]["torso"] = 7
    part_index_map[v]["neck"] = 8
    part_index_map[v]["lfleg"] = 9  # left front leg
    part_index_map[v]["lfpa"] = 10  # left front paw
    part_index_map[v]["rfleg"] = 11  # right front leg
    part_index_map[v]["rfpa"] = 12  # right front paw
    part_index_map[v]["lbleg"] = 13  # left back leg
    part_index_map[v]["lbpa"] = 14  # left back paw
    part_index_map[v]["rbleg"] = 15  # right back leg
    part_index_map[v]["rbpa"] = 16  # right back paw
    part_index_map[v]["tail"] = 17

    # [chair]
    # only has sihouette mask
    part_index_map[ANNOTATION_CLASS.chair.value] = {}

    # [cow]
    v = ANNOTATION_CLASS.cow.value
    part_index_map[v] = {}
    part_index_map[v]["head"] = 1
    part_index_map[v]["leye"] = 2  # left eye
    part_index_map[v]["reye"] = 3  # right eye
    part_index_map[v]["lear"] = 4  # left ear
    part_index_map[v]["rear"] = 5  # right ear
    part_index_map[v]["muzzle"] = 6
    part_index_map[v]["lhorn"] = 7  # left horn
    part_index_map[v]["rhorn"] = 8  # right horn
    part_index_map[v]["torso"] = 9
    part_index_map[v]["neck"] = 10
    part_index_map[v]["lfuleg"] = 11  # left front upper leg
    part_index_map[v]["lflleg"] = 12  # left front lower leg
    part_index_map[v]["rfuleg"] = 13  # right front upper leg
    part_index_map[v]["rflleg"] = 14  # right front lower leg
    part_index_map[v]["lbuleg"] = 15  # left back upper leg
    part_index_map[v]["lblleg"] = 16  # left back lower leg
    part_index_map[v]["rbuleg"] = 17  # right back upper leg
    part_index_map[v]["rblleg"] = 18  # right back lower leg
    part_index_map[v]["tail"] = 19

    # [table]
    # only has silhouette mask
    part_index_map[ANNOTATION_CLASS.table.value] = {}

    # [dog]
    part_index_map[ANNOTATION_CLASS.dog.value] = part_index_map[
        ANNOTATION_CLASS.cat.value
    ].copy()  # dog has the same set of parts with cat,
    # except for the additional
    # muzzle
    part_index_map[ANNOTATION_CLASS.dog.value]["muzzle"] = 20

    # [horse]
    v = ANNOTATION_CLASS.horse.value
    part_index_map[v] = part_index_map[
        ANNOTATION_CLASS.cow.value
    ].copy()  # horse has the same set of parts with cow,
    # except it has hoof instead of horn
    del part_index_map[v]["lhorn"]
    del part_index_map[v]["rhorn"]
    part_index_map[v]["lfho"] = 30
    part_index_map[v]["rfho"] = 31
    part_index_map[v]["lbho"] = 32
    part_index_map[v]["rbho"] = 33

    # [motorbike]
    v = ANNOTATION_CLASS.motorbike.value
    part_index_map[v] = {}
    part_index_map[v]["fwheel"] = 1
    part_index_map[v]["bwheel"] = 2
    part_index_map[v]["handlebar"] = 3
    part_index_map[v]["saddle"] = 4
    for ii in range(1, 10 + 1):
        part_index_map[v][("headlight_%d" % ii)] = 10 + ii

    # [person]
    v = ANNOTATION_CLASS.person.value
    part_index_map[v] = {}
    part_index_map[v]["head"] = 1
    part_index_map[v]["leye"] = 2  # left eye
    part_index_map[v]["reye"] = 3  # right eye
    part_index_map[v]["lear"] = 4  # left ear
    part_index_map[v]["rear"] = 5  # right ear
    part_index_map[v]["lebrow"] = 6  # left eyebrow
    part_index_map[v]["rebrow"] = 7  # right eyebrow
    part_index_map[v]["nose"] = 8
    part_index_map[v]["mouth"] = 9
    part_index_map[v]["hair"] = 10

    part_index_map[v]["torso"] = 11
    part_index_map[v]["neck"] = 12
    part_index_map[v]["llarm"] = 13  # left lower arm
    part_index_map[v]["luarm"] = 14  # left upper arm
    part_index_map[v]["lhand"] = 15  # left hand
    part_index_map[v]["rlarm"] = 16  # right lower arm
    part_index_map[v]["ruarm"] = 17  # right upper arm
    part_index_map[v]["rhand"] = 18  # right hand

    part_index_map[v]["llleg"] = 19  # left lower leg
    part_index_map[v]["luleg"] = 20  # left upper leg
    part_index_map[v]["lfoot"] = 21  # left foot
    part_index_map[v]["rlleg"] = 22  # right lower leg
    part_index_map[v]["ruleg"] = 23  # right upper leg
    part_index_map[v]["rfoot"] = 24  # right foot

    # [pottedplant]
    v = ANNOTATION_CLASS.pottedplant.value
    part_index_map[v] = {}
    part_index_map[v]["pot"] = 1
    part_index_map[v]["plant"] = 2

    # [sheep]
    v = ANNOTATION_CLASS.sheep.value
    part_index_map[v] = part_index_map[
        ANNOTATION_CLASS.cow.value
    ].copy()  # sheep has the same set of parts with cow

    # [sofa]
    # only has sihouette mask
    part_index_map[ANNOTATION_CLASS.sofa.value] = {}

    # [train]
    v = ANNOTATION_CLASS.train.value
    part_index_map[v] = {}
    part_index_map[v]["head"] = 1
    part_index_map[v]["hfrontside"] = 2  # head front side
    part_index_map[v]["hleftside"] = 3  # head left side
    part_index_map[v]["hrightside"] = 4  # head right side
    part_index_map[v]["hbackside"] = 5  # head back side
    part_index_map[v]["hroofside"] = 6  # head roof side

    for ii in range(1, 10 + 1):
        part_index_map[v][("headlight_%d" % ii)] = 10 + ii

    for ii in range(1, 10 + 1):
        part_index_map[v][("coach_%d" % ii)] = 20 + ii

    for ii in range(1, 10 + 1):
        part_index_map[v][("cfrontside_%d" % ii)] = 30 + ii  # coach front side

    for ii in range(1, 10 + 1):
        part_index_map[v][("cleftside_%d" % ii)] = 40 + ii  # coach left side

    for ii in range(1, 10 + 1):
        part_index_map[v][("crightside_%d" % ii)] = 50 + ii  # coach right side

    for ii in range(1, 10 + 1):
        part_index_map[v][("cbackside_%d" % ii)] = 60 + ii  # coach back side

    for ii in range(1, 10 + 1):
        part_index_map[v][("croofside_%d" % ii)] = 70 + ii  # coach roof side

    # [tvmonitor]
    v = ANNOTATION_CLASS.tvmonitor.value
    part_index_map[v] = {}
    part_index_map[v]["screen"] = 1

    return part_index_map


import enum


class COW_PARTS(enum.Enum):
    head = 1
    leye = 2
    reye = 3
    lear = 4
    rear = 5
    muzzle = 6
    lhorn = 7
    rhorn = 8
    torso = 9
    neck = 10
    lfuleg = 11
    lflleg = 12
    rfuleg = 13
    rflleg = 14
    lbuleg = 15
    lblleg = 16
    rbuleg = 17
    rblleg = 18
    tail = 19


class HORSE_PARTS(enum.Enum):
    head = 1
    leye = 2
    reye = 3
    lear = 4
    rear = 5
    muzzle = 6
    torso = 9
    neck = 10
    lfuleg = 11
    lflleg = 12
    rfuleg = 13
    rflleg = 14
    lbuleg = 15
    lblleg = 16
    rbuleg = 17
    rblleg = 18
    tail = 19
    lfho = 30
    rfho = 31
    blho = 32
    rbho = 33


class PERSON_PARTS(enum.Enum):
    head = 1
    leye = 2
    reye = 3
    lear = 4
    rear = 5
    lebrow = 6
    rebrow = 7
    nose = 8
    mouth = 9
    hair = 10
    torso = 11
    neck = 12
    llarm = 13
    luarm = 14
    lhand = 15
    rlarm = 16
    ruarm = 17
    rhand = 18
    llleg = 19
    luleg = 20
    lfoot = 21
    rlleg = 22
    ruleg = 23
    rfoot = 24
