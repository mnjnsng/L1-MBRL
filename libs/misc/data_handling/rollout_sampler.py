from libs.misc.utils import get_inner_env
from libs.misc.visualization import turn_off_video_recording, turn_on_video_recording
from model.controllers import RandomController
from libs.misc.data_handling.path import Path
from gym.monitoring import VideoRecorder
import logger
import numpy as np
from libs.misc.l1_adaptive_controller import L1_adapt
import csv
from scipy.linalg import null_space

import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image



class RolloutSampler:
    def __init__(self, env, dynamics=None, controller=None):
        self.env = env
        self.inner_env = get_inner_env(self.env)
        self.dynamics = dynamics
        self.controller = controller
        self.random_controller = RandomController(self.env)
        self.u0=None
        self.epsilon=1
        self.wc = 350
        self.envname = 'IP-w'+ str(self.wc) +'-e' + str(self.epsilon)
        self.aff_count_total = 0
        self.path_count = 0

    def update_dynamics(self, new_dynamics):
        self.dynamics = new_dynamics

    def update_controller(self, new_controller):
        self.controller = new_controller

    def generate_random_rollouts(self, num_paths, horizon=1000):
        logger.info("Generating random rollouts.")
        random_paths = self.sample(
            num_paths=num_paths,
            horizon=horizon,
            use_random_controller=True
        )
        logger.info("Done generating random rollouts.")
        return random_paths

    def sample(self,
               num_paths=3,
               horizon=1000,
               visualize=False,
               visualize_path_no=None,
               use_random_controller=False,
               use_adaptive_controller=False):
        # Write a sampler function which takes in an environment, a controller
        # (either random or the MPC controller), and returns rollouts by running on
        # the env. Each path can have elements for observations, next_observations,
        # rewards, returns, actions, etc.
        paths = []
        total_timesteps = 0
        path_num = 0

        controller = self._get_controller(use_random_controller=use_random_controller)

        while True:
            turn_off_video_recording()
            if visualize and not isinstance(controller, RandomController):
                if (visualize_path_no is None) or (path_num == visualize_path_no):
                    turn_on_video_recording()

            self._reset_env_for_visualization()

            logger.info("Path {} | total_timesteps {}.".format(path_num, total_timesteps))

            # update randomness
            if controller.__class__.__name__ == "MPCcontroller" \
                and hasattr(controller.dyn_model, 'update_randomness'):
                controller.dyn_model.update_randomness()

            path, total_timesteps = self._rollout_single_path(horizon, controller,
                                                             total_timesteps,use_adaptive_controller)


            paths.append(path)
            path_num += 1
            if total_timesteps >= num_paths * horizon:
                break

        turn_off_video_recording()

        return paths

    # ---- Private methods ----

    def _reset_env_for_visualization(self):
        # A hack for resetting env while recording videos
        if hasattr(self.env.wrapped_env, "stats_recorder"):
            setattr(self.env.wrapped_env.stats_recorder, "done", None)

    def _get_controller(self, use_random_controller=False):
        if use_random_controller:
            return self.random_controller
        return self.controller

    def affine_f(self,f,x,u,u0,i=None):

        f_x_u0, f_jacobian_u0 = f(x,u0,i)

        g1 = f_x_u0 - f_jacobian_u0 @ np.expand_dims(u0,axis=-1)
        g2 = f_jacobian_u0

        return g1 + g2 @ np.expand_dims(u,axis=-1), g1 , g2

    def f(self,x,u,i=None):
        pred_obs, jacobian= self.dynamics.predict_with_jacobian(np.array(x).reshape(1,-1),np.array(u).reshape(1,-1))
        pred_obs=pred_obs.squeeze(1)
        
        if i is None:
            return np.mean(np.expand_dims(pred_obs-x,axis=-1),0),np.mean(jacobian,axis=0)[:,-u.shape[0]:]

        return np.expand_dims(pred_obs-x,axis=-1)[i], jacobian[i][:,-u.shape[0]:]

    def _rollout_single_path(self, horizon, controller, total_timesteps,use_adaptive_controller=False):
        path = Path()
        obs = self.env.reset()
        self.u0= None
        affinization_counter = 0

        obs_list=['observation']
        u_b=['u_b']
        u_l=['u_l']
        u_actual = ['u_actual']
        metadata = {}
        frames=[]
        for horizon_num in range(1, horizon + 1):
        
            action=np.clip(controller.get_action(obs),self.env.wrapped_env.action_space.low,
                self.env.wrapped_env.action_space.high)
            
            if use_adaptive_controller:
                
                if self.u0 is None:
                    self.u0=action

                    affine_output, g1, g2 =self.affine_f(self.f,obs, action ,self.u0)
                    adaptive_controller = L1_adapt(self.env, g1, g2, obs,self.wc)
                
                output,_=self.f(obs, action)
                affine_output, g1, g2  = self.affine_f(self.f,obs, action ,self.u0)

                norm=np.linalg.norm(output-affine_output)

                if norm > self.epsilon:
                    self.u0=action
                    affine_output, g1, g2  = self.affine_f(self.f,obs, action ,self.u0)
                    affinization_counter+=1
                    adaptive_controller = L1_adapt(self.env, g1, g2, obs, self.wc)

                else:
                    adaptive_controller.f=g1/self.env.wrapped_env.env.dt
                    adaptive_controller.g=g2/self.env.wrapped_env.env.dt
                    adaptive_controller.g_perp=null_space((g2/self.env.wrapped_env.env.dt).T)

                u_bl=action
                u, l1_metadata =adaptive_controller.get_control_input(obs,u_bl)
                next_obs, reward, done, info = self.env.step(u)
                

                for key,val in l1_metadata.items():
                    if key not in metadata:
                        metadata[key] = [np.around(val,2)]
                    else: 
                        metadata[key].append(np.around(val,2))

                obs_list.append(np.around(obs,2))
                u_b.append(np.around(u_bl,2))
                u_l.append(np.around(u-u_bl,2))
                u_actual.append(np.around(u,2))
                path.add_timestep(obs, action, next_obs, reward)

            else:
                next_obs, reward, done, _info = self.env.step(action)
                path.add_timestep(obs, action, next_obs, reward)

            obs = next_obs
            if done or horizon_num == horizon:
                total_timesteps += horizon_num
                break
        logger.info(f"number of affinization with epsilon = {self.epsilon} is {affinization_counter}")
        
        self.aff_count_total += affinization_counter
        self.path_count += 1
        logger.info(f"average number of affinization = {self.aff_count_total / self.path_count}")
        
        data = [obs_list,u_b,u_l,u_actual]
        for key, val in metadata.items():
            val.insert(0,key)
            data.append(val)
        
        if use_adaptive_controller: file_name = str(self.envname) + '.csv'
        else: file_name = './l1_off.csv'

        with open(file_name, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return path, total_timesteps
