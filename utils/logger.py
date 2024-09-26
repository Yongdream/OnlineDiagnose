# -*- coding: UTF-8 -*-
"""
@Time     : 2021/12/12 20:32
@Author   : Caiming Liu
@Version  : V1
@File     : logger.py
@Software : PyCharm
"""

# Local Modules
import logging

# Third-party Modules


# Self-written Modules


#!/usr/bin/python
# -*- coding:utf-8 -*-

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

