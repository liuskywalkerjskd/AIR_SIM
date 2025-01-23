import mujoco
import numpy as np
import sys
import os

from discoverse.utils.base_config import BaseConfig
from discoverse.envs import SimulatorBase


class LeapHandCfg(BaseConfig):
    mjcf_file_path = "mjcf/leaphand_sensor_softness/leaphand_sensor_softness.xml"
    decimation     = 4
    timestep       = 0.001
    sync           = True
    headless       = False
    init_key       = "0"
    render_set     = {
        "fps"    : 24,
        "width"  : 1920,
        "height" : 1080,
    }
    obs_rgb_cam_id  = None
    # rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False

class LeapHandBase(SimulatorBase):
    def __init__(self, config: LeapHandCfg):
        self.nj = 16
        super().__init__(config)

        self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
        self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[:self.nj]

        self.resetState()
        print("key_shape:",self.mj_model.key(self.config.init_key))
        
        # key_shape: <_MjModelKeyframeViews
        # act: array([], dtype=float64)
        # ctrl: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # id: 0
        # mpos: array([], dtype=float64)
        # mquat: array([], dtype=float64)
        # name: '0'
        # qpos: array([-2.26135e-003,  8.37106e-007, -2.83312e-004, -3.43030e-130,
        #     -2.26135e-003,  8.37106e-007, -2.83312e-004, -8.67362e-020,
        #     -2.26135e-003,  8.37106e-007, -2.83312e-004, -8.67362e-020,
        #     -2.47518e-003, -1.96501e-130, -1.35525e-021, -6.77626e-022])
        # qvel: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # time: array([0.])
        # >
        

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        #if self.teleop:
        #    self.teleop.reset()

        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_ctrl.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        # print("mj_data.ctrl shape:", self.mj_data.ctrl.shape)

        for i in range(self.nj):
            self.mj_data.ctrl[i] = action[i]
            self.mj_data.ctrl[i] = np.clip(self.mj_data.ctrl[i], self.mj_model.actuator_ctrlrange[i][0], self.mj_model.actuator_ctrlrange[i][1])

    def step_func(self, current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq"  : self.mj_data.qpos[:self.nj].tolist(),
            "jv"  : self.mj_data.qvel[:self.nj].tolist(),
            "img" : self.img_rgb_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":
    cfg = LeapHandCfg()
    exec_node = LeapHandBase(cfg)

    obs = exec_node.reset()

    action = exec_node.init_joint_pose[:exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
