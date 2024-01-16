import os
import tomli
import numpy as np
from common import standardization


# get config data
with open(os.path.join(os.path.dirname(__file__), "configs", "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class RewardModel():
    def __init__(self):

        # define mapping dictionary
        self.reward_model_map = {
            "model_1": self._model_1,
            "model_2": self._model_2,
            "model_3": self._model_3,
            "model_4": self._model_4,
            "model_5": self._model_5,
            "model_6": self._model_6,
            "model_7": self._model_7,
            "model_8": self._model_8,
            "model_9": self._model_9,
            "model_10": self._model_10,
        }

        self.model = CFG['environment']['reward_model']['model']

    def get_reward(self, storage):

        return self.reward_model_map[self.model](storage)

    def _model_1(self, storage):

        # get spacecraft angular velocity norm
        sc_w_norm = np.sqrt(np.sum(storage.angular_velocities[-1]**2))

        # if spacecraft angular velocity norm greater than threshold -> terminate episode
        is_last_reward = bool(sc_w_norm > 0.5)

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0

        if not is_last_reward:

            # check angular error
            if storage.angular_errors[-1] > 0.5:

                # if current angular error lower then previous -> discrete positive reward. Negative otherwise
                if storage.angular_errors[-1] < storage.angular_errors[-2]:
                    reward_1 = 0.1
                else:
                    reward_1 = -0.1

            else:

                # continuous reward, dependent on angular error
                reward_2 = np.dot(
                    np.array([1, -1, -1, -1]),
                    np.array(
                        [
                            storage.quaternion_errors[-1][0]**2,
                            storage.quaternion_errors[-1][1]**2,
                            storage.quaternion_errors[-1][2]**2,
                            storage.quaternion_errors[-1][3]**2
                        ]
                    )
                )

        if is_last_reward or storage.is_last_step:

            # compute a termination reward if this is the last episode
            if storage.angular_errors[-1] < 0.5:
                reward_3 = 10
            else:
                reward_3 = -10

        rewards = [reward_1, reward_2, reward_3]

        return is_last_reward, rewards

    def _model_2(self, storage):

        # get spacecraft angular velocity norm
        sc_w_norm = np.sqrt(np.sum(storage.angular_velocities[-1]**2))

        # if spacecraft angular velocity norm greater than threshold -> terminate episode
        is_last_reward = bool(sc_w_norm > 0.5)

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0

        if not is_last_reward:

            # check angular error
            if storage.angular_errors[-1] > 0.5:

                # if current angular error lower then previous -> discrete positive reward. Negative otherwise
                if storage.angular_errors[-1] < storage.angular_errors[-2]:
                    reward_1 = 0.1
                else:
                    reward_1 = -0.1

            else:

                # continuous reward, dependent on angular error
                reward_2 = np.dot(
                    np.array([1, -1, -1, -1]),
                    np.array(
                        [
                            storage.quaternion_errors[-1][0]**2,
                            storage.quaternion_errors[-1][1]**2,
                            storage.quaternion_errors[-1][2]**2,
                            storage.quaternion_errors[-1][3]**2
                        ]
                    )
                )

        if is_last_reward or storage.is_last_step:

            # compute a termination reward if this is the last episode
            if storage.angular_errors[-1] < 0.5:
                reward_3 = 100
            else:
                reward_3 = -100

        # include a reward aimed at smoothing the control effort 
        reward_4 = - np.sqrt(np.sum(storage.actions[-1]**2))

        rewards = [reward_1, reward_2, reward_3, reward_4]

        return is_last_reward, rewards

    def _model_3(self, storage):

        # get spacecraft angular velocity norm
        sc_w_norm = np.sqrt(np.sum(storage.angular_velocities[-1]**2))

        # if spacecraft angular velocity norm greater than threshold -> terminate episode
        is_last_reward = bool(sc_w_norm > 0.5)

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0
        reward_5 = 0

        if not is_last_reward:

            # check angular error
            if storage.angular_errors[-1] > 0.5:

                # if current angular error lower then previous -> discrete positive reward. Negative otherwise
                if storage.angular_errors[-1] < storage.angular_errors[-2]:
                    reward_1 = 0.1
                else:
                    reward_1 = -0.1

            else:

                # continuous reward, dependent on angular error
                reward_2 = np.dot(
                    np.array([1, -1, -1, -1]),
                    np.array(
                        [
                            storage.quaternion_errors[-1][0]**2,
                            storage.quaternion_errors[-1][1]**2,
                            storage.quaternion_errors[-1][2]**2,
                            storage.quaternion_errors[-1][3]**2
                        ]
                    )
                )

        if is_last_reward or storage.is_last_step:

            # compute a termination reward if this is the last episode
            if storage.angular_errors[-1] < 0.5:
                reward_3 = 100
            else:
                reward_3 = -100
        
        # Reward 4: Penalize high values of commanded torques (aimed at avoiding torque saturation)  
        reward_4 = - np.sqrt(np.sum(storage.actions[-1]**2))
        
        # Reward 5: Penalize high variation of torque in a defined interval (aimed at avoiding 'noisy' commanded torques) 
        interval = 6
        if len(storage.actions) >= interval:

            var = np.std(storage.actions[-interval:], 0)**2
            reward_5 = - np.linalg.norm(var)
        
        rewards = [reward_1, reward_2, reward_3, reward_4, reward_5]

        return is_last_reward, rewards

    # The idea was to apply standardization here. The best would be to normalize the components of the reward. Normalization methods should be implemented.  
    def _model_4(self, storage):
        
        pass

    def _model_5(self, storage):

        reward_1 = np.exp(-0.5 * (np.pi + np.deg2rad(storage.angular_errors[-1]) - np.deg2rad(storage.angular_errors[-2])))
        reward_2 = 14 / (1 + np.exp(2 * abs(np.deg2rad(storage.angular_errors[-1]))))

        is_last_reward = False
        rewards = [reward_1 * reward_2]

        return is_last_reward, rewards

    def _model_6(self, storage):

        n_skipped_frames = CFG['environment']['n_skipped_frames']

        # define maximum action value
        max_action = 0.5

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0
        reward_5 = 0


        # check angular error
        if storage.angular_errors[-1] > 0.5:

            # if current angular error lower then previous -> discrete positive reward. Negative otherwise
            if storage.angular_errors[-1] < storage.angular_errors[-2]:
                reward_1 = 0.1
            else:
                reward_1 = -0.1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_2 += - 0.01 * (abs(current_action - previous_action) / max_action)
            except:
                pass

        else:

            # continuous reward, dependent on angular error
            reward_3 = np.dot(
                np.array([1, -1, -1, -1]),
                np.array(
                    [
                        storage.quaternion_errors[-1][0]**2,
                        storage.quaternion_errors[-1][1]**2,
                        storage.quaternion_errors[-1][2]**2,
                        storage.quaternion_errors[-1][3]**2
                    ]
                )
            )

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_4 += - 0.1 * (abs(current_action - previous_action) / max_action)
            except:
                pass

        if storage.is_last_step:

            # compute a termination reward if this is the last episode
            if storage.angular_errors[-1] < 0.5:
                reward_5 = 10
            else:
                reward_5 = -10

        is_last_reward = False
        rewards = [reward_1, reward_2, reward_3, reward_4, reward_5]

        return is_last_reward, rewards


    def _model_7(self, storage):


        n_skipped_frames = CFG['environment']['n_skipped_frames']

        # define maximum action value
        max_action = 0.5

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0
        reward_5 = 0
        reward_6 = 0

        # check angular error
        if storage.angular_errors[-1] > 0.5:

            # if current angular error lower then previous -> discrete positive reward. Negative otherwise
            if storage.angular_errors[-1] < storage.angular_errors[-2]:
                reward_1 = 0.1
            else:
                reward_1 = -0.1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_2 += - 0.03 * (abs(current_action - previous_action) / max_action)
            except:
                pass

        else:

            # continuous reward, dependent on angular error
            reward_3 = 4 * (storage.angular_errors[-1] - 0.5) ** 2 + 1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_4 += - 0.2 * (abs(current_action - previous_action) / max_action)
            except:
                pass

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    reward_5 += - 0.1 * (abs(current_action) / max_action)
            except:
                pass

        if storage.is_last_step:

            # compute a termination reward if this is the last episode
            if storage.angular_errors[-1] < 0.5:
                reward_6 = 10
            else:
                reward_6 = -10

        is_last_reward = False
        rewards = [reward_1, reward_2, reward_3, reward_4, reward_5, reward_6]

        return is_last_reward, rewards

    def _model_8(self, storage):


        n_skipped_frames = CFG['environment']['n_skipped_frames']

        # define maximum action value
        max_action = 0.5

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0
        reward_5 = 0

        k_1 = 0.01
        k_2 = 0.2
        k_3 = 0.2

        # check angular error
        if storage.angular_errors[-1] > 0.5:

            # if current angular error lower then previous -> discrete positive reward. Negative otherwise
            if storage.angular_errors[-1] < storage.angular_errors[-2]:
                reward_1 = 0.1
            else:
                reward_1 = -0.1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_2 += - k_1 * (abs(current_action - previous_action) / max_action)
            except:
                pass

        else:

            # continuous reward, dependent on angular error
            reward_3 = 4 * (storage.angular_errors[-1] - 0.5) ** 2 + 1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_4 += - k_2 * (abs(current_action - previous_action) / max_action)
            except:
                pass

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    reward_5 += - k_3 * (abs(current_action) / max_action)
            except:
                pass

        is_last_reward = False
        rewards = [reward_1, reward_2, reward_3, reward_4, reward_5]

        return is_last_reward, rewards

    def _model_9(self, storage):

        n_skipped_frames = CFG['environment']['n_skipped_frames']

        # define maximum action value
        max_action = 0.5

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0
        reward_5 = 0
        reward_6 = 0

        k_1 = 0.01
        k_2 = 0.01
        k_3 = 0.2
        k_4 = 0.2

        # check angular error
        if storage.angular_errors[-1] > 0.5:

            # if current angular error lower then previous -> discrete positive reward. Negative otherwise
            if storage.angular_errors[-1] < storage.angular_errors[-2]:
                reward_1 = 0.1
            else:
                reward_1 = -0.1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_2 += - k_1 * (abs(current_action - previous_action) / max_action)
            except:
                pass

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    reward_3 += - k_2 * (abs(current_action) / max_action)
            except:
                pass

        else:

            # continuous reward, dependent on angular error
            reward_4 = 4 * (storage.angular_errors[-1] - 0.5) ** 2 + 1

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_5 += - k_3 * (abs(current_action - previous_action) / max_action)
            except:
                pass

            try: 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    reward_6 += - k_4 * (abs(current_action) / max_action)
            except:
                pass

        is_last_reward = False
        rewards = [reward_1, reward_2, reward_3, reward_4, reward_5, reward_6]

        return is_last_reward, rewards

    def _model_10(self, storage):

        n_skipped_frames = CFG['environment']['n_skipped_frames']

        # define maximum action value
        max_action = 0.5

        # initialize rewards to 0
        reward_1 = 0
        reward_2 = 0
        reward_3 = 0
        reward_4 = 0
        reward_5 = 0
        reward_6 = 0

        k_1 = 0.05
        k_2 = 0.05
        k_3 = 0.5
        k_4 = 0.5


        # if current angular error lower then positive reward. Negative otherwise
        if storage.angular_errors[-1] < storage.angular_errors[-2]:
            # positive reward 
            reward_1 = np.exp(-np.deg2rad(storage.angular_errors[-1]) / (0.14 * 2 * np.pi))

        else:
            # negative reward   
            reward_1 = np.exp(-np.deg2rad(storage.angular_errors[-1]) / (0.14 * 2 * np.pi)) - 1

        # check angular error
        if storage.angular_errors[-1] > 0.5:

            try:
                # negative reward 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_2 += - k_1 * (abs(current_action - previous_action) / max_action)
            except:
                pass

            try:
                # negative reward 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    reward_3 += - k_2 * (abs(current_action) / max_action)
            except:
                pass

        else:

            # positive reward
            reward_4 = reward_1 + 9

            try:
                # negative reward 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    previous_action = storage.actions[-2 * n_skipped_frames - 2][idx]
                    reward_5 += - k_3 * (abs(current_action - previous_action) / max_action)
            except:
                pass

            try:
                # negative reward 
                for idx in range(3):
                    current_action = storage.actions[-1 * n_skipped_frames - 1][idx]
                    reward_6 += - k_4 * (abs(current_action) / max_action)
            except:
                pass

        is_last_reward = False
        rewards = [reward_1, reward_2, reward_3, reward_4, reward_5, reward_6]

        return is_last_reward, rewards