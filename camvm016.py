# Copyright (c) 2020 PHYTEC Messtechnik GmbH
# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import numpy as np
import time
import subprocess
import fcntl
import ioctl_h
import ctypes

V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE = 9
VIDIOC_STREAMOFF = ioctl_h._IOW('V', 19, ctypes.c_int)


def get_camera():
    videodev = 'video0'
    buildinfo = cv2.getBuildInformation()

    if buildinfo.find('GStreamer') < 0:
        print('no GStreamer support in OpenCV')
        exit(0)

    path = os.path.join('/sys/bus/i2c/devices', '2-0010', 'driver')
    if not os.path.exists(path):
        return None

    failed = False
    # Make sure the camera is in a defined state
    cmd = 'media-ctl -V "31:0[fmt:SGRBG8_1X8/1280x800 (4,4)/1280x800]"'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('media-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'media-ctl -V "22:0[fmt:SGRBG8_1X8/1280x800]"'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('media-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -d0 -v width=1280,height=800,pixelformat=GRBG'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -d0 -c vertical_flip=1'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -c horizontal_blanking=2500'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -c digital_gain_red=1400'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True
    cmd = 'v4l2-ctl -c digital_gain_blue=1700'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        failed = True

    if failed:
        vd = os.open('/dev/video0', os.O_RDWR | os.O_NONBLOCK, 0)
        btype = ctypes.c_uint(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        fcntl.ioctl(vd, VIDIOC_STREAMOFF, btype)
        os.close(vd)

        cmd = 'media-ctl -V "31:0[fmt:SGRBG8_1X8/1280x800 (4,4)/1280x800]"'
        ret = subprocess.call(cmd, shell=True)
        cmd = 'media-ctl -V "22:0[fmt:SGRBG8_1X8/1280x800]"'
        ret = subprocess.call(cmd, shell=True)
        cmd = 'v4l2-ctl -d0 -v width=1280,height=800,pixelformat=GRBG'
        ret = subprocess.call(cmd, shell=True)
        cmd = 'v4l2-ctl -d0 -c vertical_flip=1'
        ret = subprocess.call(cmd, shell=True)
        cmd = 'v4l2-ctl -c horizontal_blanking=2500'
        ret = subprocess.call(cmd, shell=True)
        cmd = 'v4l2-ctl -c digital_gain_red=1400'
        ret = subprocess.call(cmd, shell=True)
        cmd = 'v4l2-ctl -c digital_gain_blue=1700'
        ret = subprocess.call(cmd, shell=True)

    pipeline = 'v4l2src device=/dev/{video} ! appsink'.format(video=videodev)
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def color_convert(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2RGB)
