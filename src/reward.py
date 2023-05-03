import tomli
import numpy as np


# get config data
with open("configs/config.toml", "rb") as config_file:
    CFG = tomli.load(config_file)


class RewardModel():
    def __init__(self):

        # define mapping dictionary
        self.reward_model_map = {
            "model_1": self._model_1,
            "model_2": self._model_2,
        }

        self.model = CFG['environment']['reward_model']['model']

    def get_reward(self, action, storage):

        return self.reward_model_map[self.model](
            action,
            storage
        )

    def _model_1(self, action, storage):

        # get spacecraft angular velocity norm
        sc_w_norm = np.sqrt(np.sum(storage.angular_velocities[-1]**2))

        # if spacecraft angular velocity norm greater than threshold -> terminate episode
        is_last_reward = bool(sc_w_norm > 0.5)

        if not is_last_reward:

            # check angular error
            if storage.angular_errors[-1] > 0.5:

                # if current angular error lower then previous -> discrete positive reward. Negative otherwise
                if storage.angular_errors[-1] < storage.angular_errors[-2]:
                    reward = 0.1
                else:
                    reward = -0.1

            else:

                # continuous reward, dependent on angular error
                reward = np.dot(
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

        else:

            # compute a termination reward if this is the last episode
            if storage.angular_errors[-1] < 0.5:
                reward = 10
            else:
                reward = -10

        return is_last_reward, reward

    def _model_2(self):
        pass


