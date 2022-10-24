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

    width = 1280
    height = 800

    size = f'{width}x{height}'
    cmd = f'setup-pipeline-csi1 -s {size} -c {size}'
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        subprocess.call(cmd, shell=True)

    controls = [
        '-c vertical_flip=1',
        '-c horizontal_blanking=2500',
        '-c digital_gain_red=1400',
        '-c digital_gain_blue=1700',
    ]
    cmd = f'v4l2-ctl -d /dev/cam-csi1 {" ".join(controls)}'
    print(cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print('v4l2-ctl failed: {}'.format(ret))
        subprocess.call(cmd, shell=True)

    fmt = f'video/x-bayer,format=grbg,width={width},height={height}'
    pipeline = f'v4l2src device=/dev/video-csi1 ! {fmt} ! appsink'
    return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


def color_convert(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2RGB)
