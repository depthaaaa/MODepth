# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)

import os
import shutil
def save_code(srcfile, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if os.path.isdir(srcfile):
        if not os.path.exists(os.path.join(log_path, os.path.split(srcfile)[-1])):
            os.makedirs(os.path.join(log_path, os.path.split(srcfile)[-1]))
        log_path = os.path.join(log_path, os.path.split(srcfile)[-1])
    if os.path.isdir(srcfile):
        for ipath in os.listdir(srcfile):
            fulldir = os.path.join(srcfile, ipath)  
            print(srcfile, "file")         
            if os.path.isfile(fulldir):  
                shutil.copy(fulldir, log_path)
            if os.path.isdir(fulldir):  
                save_code(fulldir, log_path)
    else:
        if not os.path.isfile(srcfile):
            print("%s not exist!"%(srcfile))
        else:
            fpath, fname = os.path.split(srcfile)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            shutil.copy(srcfile, os.path.join(log_path, fname))
            print("copy %s -> %s" % (srcfile, os.path.join(log_path, fname)))
        