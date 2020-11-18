# Copyright (c) 2020 PHYTEC Messtechnik GmbH
# SPDX-License-Identifier: Apache-2.0

import cv2


def get_camera():
    return cv2.VideoCapture(0)


def color_convert(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)