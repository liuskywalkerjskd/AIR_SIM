import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

import os
import shutil
import argparse
import multiprocessing as mp

import traceback
from discoverse.airbot_play import AirbotPlayIK_nopin #使用这个进行逆运动学解算无需URDF
from discoverse import DISCOVERSE_ROOT_DIR , DISCOVERSE_ASSERT_DIR #引入仿真器路径和模型路径

from discoverse.utils import get_body_tmat , step_func , SimpleStateMachine #获取旋转矩阵，步进，状态机

from discoverse.envs.hand_with_arm_base import HandWithArmCfg #引入手臂基础配置

class SimNode():
    def __init__(self):
        pass