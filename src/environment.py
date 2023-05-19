import tomli
import numpy as np
from gymnasium import Env
from spacecraft import Spacecraft
from propagator import Propagator
from storage import Storage
from observation_space import ObservationSpaceModel
from action_space import ActionSpaceModel
from reward import RewardModel


# get config data
with open("configs/config.toml", "rb") as config_file:
    CFG = tomli.load(config_file)


class SpacecraftEnv(Env):
    def __init__(self):

        self.spacecraft = Spacecraft()
        self.propagator = Propagator()
        self.storage = Storage()
        self.observation_space_model = ObservationSpaceModel()
        self.action_space_model = ActionSpaceModel()
        self.reward_model = RewardModel()

        self.n_skipped_frames = CFG['environment']['n_skipped_frames']

        # define observation space
        self.observation_space = self.observation_space_model.get_observation_space()

        # define action space
        self.action_space = self.action_space_model.get_action_space()

    def reset(self):

        # reset spacecraft object
        self.spacecraft.reset()

        # reset propagator object
        self.propagator.reset()

        # reset storage object
        self.storage.reset(
            self.spacecraft.attitude.current_quaternion,
            self.spacecraft.attitude.quaternion_error,
            self.spacecraft.attitude.angular_error,
            self.spacecraft.attitude.angular_velocity
        )

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):

        # propagate spacecraft states
        is_last_step = self._environment_step(action)

        # get environmnent observations
        observation = self._get_observation()

        # compute agent reward
        is_last_reward, reward = self.reward_model.get_reward(self.storage)

        # check termination condition
        if is_last_reward or is_last_step:
            terminated = True
        else:
            terminated = False

        info = {}

        return observation, reward, terminated, False, info

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