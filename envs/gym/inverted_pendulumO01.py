
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def _step(self, a):
        # reward = 1.0
        reward = self._get_reward()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone
        done = False
        ob += np.random.uniform(low=-0.1, high=0.1, size=ob.shape)
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reward(self):
        old_ob = self._get_obs()
        reward = -((old_ob[1]) ** 2)
        return reward

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

    def cost_np_vec(self, obs, acts, next_obs):
        return ((obs[:, 1]) ** 2)
