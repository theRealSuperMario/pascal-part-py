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
from pascal_part_py import voc_utils
from pascal_part_py.voc_utils import ANNOTATION_CLASS


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
    pimap = {}

    # [aeroplane]
    v = ANNOTATION_CLASS.aeroplane.value
    pimap[v] = {}
    pimap[v]["body"] = 1
    pimap[v]["stern"] = 2
    pimap[v]["lwing"] = 3  # left wing
    pimap[v]["rwing"] = 4  # right wing
    pimap[v]["tail"] = 5
    for ii in range(1, 10 + 1):
        pimap[v][("engine_%d" % ii)] = 10 + ii  # multiple engines
    for ii in range(1, 10 + 1):
        pimap[v][("wheel_%d" % ii)] = 20 + ii  # multiple wheels

    # [bicycle]
    v = ANNOTATION_CLASS.bicycle.value
    pimap[v] = {}
    pimap[v]["fwheel"] = 1  # front wheel
    pimap[v]["bwheel"] = 2  # back wheel
    pimap[v]["saddle"] = 3
    pimap[v]["handlebar"] = 4  # handle bar
    pimap[v]["chainwheel"] = 5  # chain wheel
    for ii in range(1, 10 + 1):
        pimap[v][("headlight_%d" % ii)] = 10 + ii

    # [bird]
    v = ANNOTATION_CLASS.bird.value
    pimap[v] = {}
    pimap[v]["head"] = 1
    pimap[v]["leye"] = 2  # left eye
    pimap[v]["reye"] = 3  # right eye
    pimap[v]["beak"] = 4
    pimap[v]["torso"] = 5
    pimap[v]["neck"] = 6
    pimap[v]["lwing"] = 7  # left wing
    pimap[v]["rwing"] = 8  # right wing
    pimap[v]["lleg"] = 9  # left leg
    pimap[v]["lfoot"] = 10  # left foot
    pimap[v]["rleg"] = 11  # right leg
    pimap[v]["rfoot"] = 12  # right foot
    pimap[v]["tail"] = 13

    # [boat]
    # only has silhouette mask

    # [bottle]
    v = ANNOTATION_CLASS.bottle.value
    pimap[v] = {}
    pimap[v]["cap"] = 1
    pimap[v]["body"] = 2

    # [bus]
    v = ANNOTATION_CLASS.bus.value
    pimap[v] = {}
    pimap[v]["frontside"] = 1
    pimap[v]["leftside"] = 2
    pimap[v]["rightside"] = 3
    pimap[v]["backside"] = 4
    pimap[v]["roofside"] = 5
    pimap[v]["leftmirror"] = 6
    pimap[v]["rightmirror"] = 7
    pimap[v]["fliplate"] = 8  # front license plate
    pimap[v]["bliplate"] = 9  # back license plate
    for ii in range(1, 10 + 1):
        pimap[6][("door_%d" % ii)] = 10 + ii
    for ii in range(1, 10 + 1):
        pimap[6][("wheel_%d" % ii)] = 20 + ii
    for ii in range(1, 10 + 1):
        pimap[6][("headlight_%d" % ii)] = 30 + ii
    for ii in range(1, 20 + 1):
        pimap[6][("window_%d" % ii)] = 40 + ii

    # [car]
    pimap[ANNOTATION_CLASS.car.value] = pimap[
        ANNOTATION_CLASS.bus.value
    ].copy()  # car has the same set of parts with bus

    # [cat]
    v = ANNOTATION_CLASS.cat.value
    pimap[v] = {}
    pimap[v]["head"] = 1
    pimap[v]["leye"] = 2  # left eye
    pimap[v]["reye"] = 3  # right eye
    pimap[v]["lear"] = 4  # left ear
    pimap[v]["rear"] = 5  # right ear
    pimap[v]["nose"] = 6
    pimap[v]["torso"] = 7
    pimap[v]["neck"] = 8
    pimap[v]["lfleg"] = 9  # left front leg
    pimap[v]["lfpa"] = 10  # left front paw
    pimap[v]["rfleg"] = 11  # right front leg
    pimap[v]["rfpa"] = 12  # right front paw
    pimap[v]["lbleg"] = 13  # left back leg
    pimap[v]["lbpa"] = 14  # left back paw
    pimap[v]["rbleg"] = 15  # right back leg
    pimap[v]["rbpa"] = 16  # right back paw
    pimap[v]["tail"] = 17

    # [chair]
    # only has sihouette mask

    # [cow]
    v = ANNOTATION_CLASS.cow.value
    pimap[v] = {}
    pimap[v]["head"] = 1
    pimap[v]["leye"] = 2  # left eye
    pimap[v]["reye"] = 3  # right eye
    pimap[v]["lear"] = 4  # left ear
    pimap[v]["rear"] = 5  # right ear
    pimap[v]["muzzle"] = 6
    pimap[v]["lhorn"] = 7  # left horn
    pimap[v]["rhorn"] = 8  # right horn
    pimap[v]["torso"] = 9
    pimap[v]["neck"] = 10
    pimap[v]["lfuleg"] = 11  # left front upper leg
    pimap[v]["lflleg"] = 12  # left front lower leg
    pimap[v]["rfuleg"] = 13  # right front upper leg
    pimap[v]["rflleg"] = 14  # right front lower leg
    pimap[v]["lbuleg"] = 15  # left back upper leg
    pimap[v]["lblleg"] = 16  # left back lower leg
    pimap[v]["rbuleg"] = 17  # right back upper leg
    pimap[v]["rblleg"] = 18  # right back lower leg
    pimap[v]["tail"] = 19

    # [table]
    # only has silhouette mask

    # [dog]
    pimap[ANNOTATION_CLASS.dog.value] = pimap[
        ANNOTATION_CLASS.cat.value
    ].copy()  # dog has the same set of parts with cat,
    # except for the additional
    # muzzle
    pimap[ANNOTATION_CLASS.dog.value]["muzzle"] = 20

    # [horse]
    v = ANNOTATION_CLASS.horse.value
    pimap[v] = pimap[
        ANNOTATION_CLASS.cow.value
    ].copy()  # horse has the same set of parts with cow,
    # except it has hoof instead of horn
    del pimap[v]["lhorn"]
    del pimap[v]["rhorn"]
    pimap[v]["lfho"] = 30
    pimap[v]["rfho"] = 31
    pimap[v]["lbho"] = 32
    pimap[v]["rbho"] = 33

    # [motorbike]
    v = ANNOTATION_CLASS.motorbike.value
    pimap[v] = {}
    pimap[v]["fwheel"] = 1
    pimap[v]["bwheel"] = 2
    pimap[v]["handlebar"] = 3
    pimap[v]["saddle"] = 4
    for ii in range(1, 10 + 1):
        pimap[v][("headlight_%d" % ii)] = 10 + ii

    # [person]
    v = ANNOTATION_CLASS.person.value
    pimap[v] = {}
    pimap[v]["head"] = 1
    pimap[v]["leye"] = 2  # left eye
    pimap[v]["reye"] = 3  # right eye
    pimap[v]["lear"] = 4  # left ear
    pimap[v]["rear"] = 5  # right ear
    pimap[v]["lebrow"] = 6  # left eyebrow
    pimap[v]["rebrow"] = 7  # right eyebrow
    pimap[v]["nose"] = 8
    pimap[v]["mouth"] = 9
    pimap[v]["hair"] = 10

    pimap[v]["torso"] = 11
    pimap[v]["neck"] = 12
    pimap[v]["llarm"] = 13  # left lower arm
    pimap[v]["luarm"] = 14  # left upper arm
    pimap[v]["lhand"] = 15  # left hand
    pimap[v]["rlarm"] = 16  # right lower arm
    pimap[v]["ruarm"] = 17  # right upper arm
    pimap[v]["rhand"] = 18  # right hand

    pimap[v]["llleg"] = 19  # left lower leg
    pimap[v]["luleg"] = 20  # left upper leg
    pimap[v]["lfoot"] = 21  # left foot
    pimap[v]["rlleg"] = 22  # right lower leg
    pimap[v]["ruleg"] = 23  # right upper leg
    pimap[v]["rfoot"] = 24  # right foot

    # [pottedplant]
    v = ANNOTATION_CLASS.pottedplant.value
    pimap[v] = {}
    pimap[v]["pot"] = 1
    pimap[v]["plant"] = 2

    # [sheep]
    v = ANNOTATION_CLASS.sheep.value
    pimap[v] = pimap[
        ANNOTATION_CLASS.cow.value
    ].copy()  # sheep has the same set of parts with cow

    # [sofa]
    # only has sihouette mask

    # [train]
    v = ANNOTATION_CLASS.train.value
    pimap[v] = {}
    pimap[v]["head"] = 1
    pimap[v]["hfrontside"] = 2  # head front side
    pimap[v]["hleftside"] = 3  # head left side
    pimap[v]["hrightside"] = 4  # head right side
    pimap[v]["hbackside"] = 5  # head back side
    pimap[v]["hroofside"] = 6  # head roof side

    for ii in range(1, 10 + 1):
        pimap[v][("headlight_%d" % ii)] = 10 + ii

    for ii in range(1, 10 + 1):
        pimap[v][("coach_%d" % ii)] = 20 + ii

    for ii in range(1, 10 + 1):
        pimap[v][("cfrontside_%d" % ii)] = 30 + ii  # coach front side

    for ii in range(1, 10 + 1):
        pimap[v][("cleftside_%d" % ii)] = 40 + ii  # coach left side

    for ii in range(1, 10 + 1):
        pimap[v][("crightside_%d" % ii)] = 50 + ii  # coach right side

    for ii in range(1, 10 + 1):
        pimap[v][("cbackside_%d" % ii)] = 60 + ii  # coach back side

    for ii in range(1, 10 + 1):
        pimap[v][("croofside_%d" % ii)] = 70 + ii  # coach roof side

    # [tvmonitor]
    v = ANNOTATION_CLASS.tvmonitor.value
    pimap[v] = {}
    pimap[v]["screen"] = 1

    return pimap
