import mujoco
import numpy as np

from discoverse.utils.base_config import BaseConfig
from discoverse.envs import SimulatorBase


class InspireHandCfg(BaseConfig):
    mjcf_file_path = "mjcf/inspire_hand/inspire_right_hand.xml"
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
    obj_list       = []
    use_gaussian_renderer = False
    
class InspireHandBase(SimulatorBase):
    def __init__(self, config: InspireHandCfg):
        self.nj = 12 #inspire hand has 12 joints
        self.na = 6 #inspire hand has 6 actuators
        super().__init__(config)
        
        self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.nj]
        self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[:self.na]
        self.resetState()
        print("key_shape:",self.mj_model.key(self.config.init_key))
        
        # key_shape: <_MjModelKeyframeViews
        # act: array([], dtype=float64)
        # ctrl: array([0., 0., 0., 0., 0., 0.])
        # id: 10
        # mpos: array([], dtype=float64)
        # mquat: array([], dtype=float64)
        # name: '10'
        # qpos: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # qvel: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        # time: array([0.])
        #>
        
    def resetState(self):
        print("sim_reset")
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.na] = self.init_joint_ctrl.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)
        
    def updateControl(self, action):
        # print("mj_data.ctrl shape:", self.mj_data.ctrl.shape)

        for i in range(self.na):
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
    cfg = InspireHandCfg()
    exec_node = InspireHandBase(cfg)
    
    obs = exec_node.reset()
    
    action = exec_node.init_joint_ctrl[:exec_node.na]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)