import os
import tomli
import shutil
import numpy as np
import gymnasium as gym
from spacecraft import Spacecraft
from propagator import Propagator
from storage import Storage
from observation_space import ObservationSpaceModel
from action_space import ActionSpaceModel
from reward import RewardModel
from normalizator import Normalizator


# get config data
with open(os.path.join(os.path.dirname(__file__), "configs", "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class SpacecraftEnv(gym.Env):
    def __init__(self, norm_stats_path=None, training=True):

        self.spacecraft = Spacecraft()
        self.propagator = Propagator()
        self.storage = Storage()
        self.observation_space_model = ObservationSpaceModel()
        self.action_space_model = ActionSpaceModel()
        self.reward_model = RewardModel()

        self.n_skipped_frames = CFG['environment']['n_skipped_frames']
        self.use_random_seed = CFG['environment']['use_random_seed']
        self.random_seed = CFG['environment']['random_seed']
        self.normalize_obs = CFG['environment']['normalize_obs']
        self.normalize_reward = CFG['environment']['normalize_reward']

        # define observation space
        self.observation_space = self.observation_space_model.get_observation_space()

        # define action space
        self.action_space = self.action_space_model.get_action_space()

        # initialize normalizator object if needed
        if self.normalize_obs or self.normalize_reward:
            self.normalizator = Normalizator(
                obs_space_shape=self.observation_space.shape,
                norm_stats_path=norm_stats_path,
                norm_obs=self.normalize_obs,
                norm_reward=self.normalize_reward,
                training=training
            )
        else:
            self.normalizator = None


    def reset(self, *, seed=None, options=None):

        if self.use_random_seed:
            seed = self.random_seed

        # the following line is needed for custom environments
        super().reset(seed=seed)

        # reset spacecraft object
        self.spacecraft.reset()

        # reset propagator object
        self.propagator.reset()

        # reset normalizator object
        if self.normalize_obs or self.normalize_reward:
            self.normalizator.reset()

        # reset storage object
        self.storage.reset(
            self.spacecraft.attitude.current_quaternion,
            self.spacecraft.attitude.quaternion_error,
            self.spacecraft.attitude.angular_error,
            self.spacecraft.attitude.angular_velocity
        )

        observation = self._get_observation()

        # normalize observation
        if self.normalize_obs:
            observation = self.normalizator.get_normalized_obs(observation)

        info = {}

        return observation, info

    def step(self, action):

        # elaborate action 
        action = self.action_space_model.get_elaborated_action(action, self.storage)

        # propagate spacecraft states
        is_last_step = self._environment_step(action)

        # get environmnent observations
        observation = self._get_observation()

        # normalize observation
        if self.normalize_obs:
            observation = self.normalizator.get_normalized_obs(observation)

        # compute agent reward
        is_last_reward, rewards = self.reward_model.get_reward(self.storage)
        reward = sum(rewards)

        # normalize reward
        if self.normalize_reward:
            reward = self.normalizator.get_normalized_reward(reward)

        # check termination condition
        if is_last_reward or is_last_step:
            terminated = True
        else:
            terminated = False

        # set truncated parameter to False (unused)
        truncated = False

        # store rewards values so that can be accessed by TensorboardCallback object
        info = {
            'rewards': rewards
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def render_animation(self):

        self.storage.render_animation(
            self.spacecraft.attitude.target_quaternion,
            self.propagator.integration_step
        )

    def plot_results(self):

        self.storage.plot_results()

    def save_configs(self, path):

        shutil.copy(
            os.path.join(os.path.dirname(__file__), "configs", "config.toml"),
            os.path.join(path, "env_config.toml")
        )

        if self.normalize_obs or self.normalize_reward:
            self.normalizator.save_stats(path)


    def _get_observation(self):

        return self.storage.get_env_states()

    def _environment_step(self, action):

        for frame in range(self.n_skipped_frames + 1):

            if frame > 0:
                action = np.array([0, 0, 0])

            # get spacecraft states
            states = self.spacecraft.get_prop_states()

            # propagate spacecraft states
            is_last_step, ode_solution = self.propagator.propagate(
                states,
                action,
                self.spacecraft.inertia.matrix
            )

            # update spacecraft states
            self.spacecraft.update_states(ode_solution)

            # update records
            self.storage.update_records(
                self.propagator.current_time,
                is_last_step,
                self.spacecraft.attitude.current_quaternion,
                self.spacecraft.attitude.quaternion_error,
                self.spacecraft.attitude.angular_error,
                self.spacecraft.attitude.angular_velocity,
                action
            )

        return is_last_step


if __name__ == "__main__":

    import time

    env = SpacecraftEnv()
    env.reset()


    t1 = time.time()
    for i in range(1000):
        action = env.action_space.sample()
        env.step(action)
    t2 = time.time()

    print(t2-t1)

    env.plot_results()
    env.render_animation()