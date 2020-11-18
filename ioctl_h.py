# Copyright (c) 2020 PHYTEC Messtechnik GmbH
# SPDX-License-Identifier: Apache-2.0

import ctypes


_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ  = 2


def _IOC(dir_, type_, nr, size):
    return (
        ctypes.c_int32(dir_ << _IOC_DIRSHIFT).value |
        ctypes.c_int32(ord(type_) << _IOC_TYPESHIFT).value |
        ctypes.c_int32(nr << _IOC_NRSHIFT).value |
        ctypes.c_int32(size << _IOC_SIZESHIFT).value)


def _IOC_TYPECHECK(t):
    return ctypes.sizeof(t)


def _IO(type_, nr):
    return _IOC(_IOC_NONE, type_, nr, 0)


def _IOW(type_, nr, size):
    return _IOC(_IOC_WRITE, type_, nr, _IOC_TYPECHECK(size))


def _IOR(type_, nr, size):
    return _IOC(_IOC_READ, type_, nr, _IOC_TYPECHECK(size))


def _IOWR(type_, nr, size):
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, _IOC_TYPECHECK(size))
