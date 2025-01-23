import numpy as np
import mujoco
import matplotlib.pyplot as plt
from discoverse.envs.inspire_hand_base import InspireHandCfg, InspireHandBase

action = np.zeros(12) #inspire hand has 12 joints
obs_lst = []

class SimNode(InspireHandBase):
    key_id = 0
    #inspire hand has 12 joints
    target_action = np.zeros(12) 
    joint_move_ratio = np.zeros(12)
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def resetState(self):
        super().resetState()
        global action, obs_lst
        obs_lst.clear()
        self.key_id = 0
        self.update_control_from_keyframe("0")
        action[:] = self.target_action[:]

    def update_control_from_keyframe(self, key):
        self.target_action = self.mj_model.key(key).qpos[:self.nj].copy()
        global action
        dif = np.abs(action - self.target_action)
        self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
        
        print("target pos:",self.mj_model.key(key).qpos[:self.nj])
        print("joint_move_ratio:",self.joint_move_ratio)

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        
        if key != -1:
            print("key:", key)
            
        if key == ord(' '):
            self.key_id += 1
            if self.key_id > 12:
                self.key_id = 12
                print("key_id is out of range")
            self.update_control_from_keyframe(self.key_id)
        return ret
    
    def getObservation(self):
        self.obs = {
            "jq"    : self.mj_data.qpos[:self.nj].tolist(),
            "jv"    : self.mj_data.qvel[:self.nj].tolist(),
            "img"   : self.img_rgb_obs_s,
        }
        #print("mj_data:",self.mj_data.qpos[:self.nj].tolist())
        return self.obs


cfg = InspireHandCfg()
cfg.use_gaussian_renderer = False

cfg.timestep     = 0.001
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.init_key     = "10"
cfg.render_set   = {
    "fps"    : 30,
    "width"  : 1920, # 640,
    "height" : 1080  # 480
}
cfg.obs_rgb_cam_id   = -1

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    def step_func(current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

    sim_node = SimNode(cfg)
    sim_node.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False


    try:
        while sim_node.running:

            for i in range(sim_node.nj):
                action[i] = step_func(action[i], sim_node.target_action[i], 2. * sim_node.joint_move_ratio[i] * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

            obs, _, _, _, _ = sim_node.step(action)
            if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
                obs_lst.append(obs)

    except Exception as e:
        print(e)
    finally:
        if len(obs_lst):
            print("have data")

        else:
            print("no data")

