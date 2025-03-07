# Copyright (c) 2025 PHYTEC Messtechnik GmbH
# SPDX-License-Identifier: Apache-2.0
# Author: Martin Schwan <m.schwan@phytec.de>

import subprocess

import cv2 as cv

class Camera():
    def __init__(self):
        self.camera_device = None
        self.color_conversion_code = None
        self.api_preference = None
        self.video_capture = cv.VideoCapture()

    def open(self, filename):
        if self.api_preference is None:
            raise AttributeError('API preference must be set before opening '
                                 'video capture device!')
        self.video_capture.open(filename, self.api_preference)
        if not self.video_capture.isOpened():
            raise ValueError(f'Failed opening video capture device "{filename}"!')

    def convert_frame_color(self, frame):
        if self.color_conversion_code is None:
            raise AttributeError('Color conversion code must be set before '
                                 'converting frame colors!')
        return cv.cvtColor(frame, self.color_conversion_code)

class CameraUSB(Camera):
    def __init__(self):
        super().__init__()
        self.color_conversion_code = cv.COLOR_BGR2RGB
        self.api_preference = cv.CAP_ANY

    def open(self, filename=1):
        super().open(filename)

class CameraVM016(Camera):
    def __init__(self):
        super().__init__()
        self.color_conversion_code = cv.COLOR_BAYER_GB2RGB
        self.api_preference = cv.CAP_GSTREAMER

        if cv.getBuildInformation().find('GStreamer') < 0:
            raise ValueError('This version of OpenCV does not support GStreamer!')

    def open(self, filename='/dev/cam-csi1'):
        video_device = '/dev/video-isi-csi1'

        width = 1280
        height = 800

        size = f'{width}x{height}'
        cmd = f'setup-pipeline-csi1 -s {size} -c {size}'
        subprocess.run(cmd, shell=True, check=True)

        controls = [
            '-c vertical_flip=1',
            '-c horizontal_blanking=2500',
            '-c digital_gain_red=1400',
            '-c digital_gain_blue=1700',
        ]
        cmd = f'v4l2-ctl -d {filename} {" ".join(controls)}'
        subprocess.run(cmd, shell=True, check=True)

        fmt = f'video/x-bayer,format=grbg,width={width},height={height}'
        super().open(f'v4l2src device={video_device} ! {fmt} ! appsink')
