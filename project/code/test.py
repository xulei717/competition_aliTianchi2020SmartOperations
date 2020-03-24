# -*- coding:utf-8 -*-
# @time   : 2020-03-24 12:48
# @author : xl
# @project: project

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
print(curPath)
print(os.path.split(curPath))
print(parentPath)
print(rootPath)
sys.path.append(parentPath)
