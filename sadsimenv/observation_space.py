import os
import tomli
import numpy as np
from gymnasium import spaces


# get config data
with open(os.path.join(os.path.dirname(__file__), "configs", "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class ObservationSpaceModel():
    def __init__(self):

        # define mapping dictionary
        self.observation_model_map = {
            "model_1": self._model_1,
            "model_2": self._model_2,
        }

        self.model = CFG['environment']['observation_space']['model']

    def get_observation_space(self):

        return self.observation_model_map[self.model]()

    def _model_1(self):

        # define observation space limits
        observation_limit = np.array(
            [
                1,
                1,
                1,
                1,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max,
                np.finfo(np.float32).max
            ],
            dtype=np.float32,
        )

        # define observation space
        observation_space = spaces.Box(
            -observation_limit,
            observation_limit,
            dtype=np.float32
        )

        return observation_space

    def _model_2(self):

        # define observation space
        observation_space = spaces.Box(
            np.array(
                [
                    - 1,
                    - 1,
                    - 1,
                    - 1,
                    - np.finfo(np.float32).max,
                    - np.finfo(np.float32).max,
                    - np.finfo(np.float32).max,
                    - 0.5,
                    - 0.5,
                    - 0.5
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    1,
                    1,
                    1,
                    1,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    np.finfo(np.float32).max,
                    0.5,
                    0.5,
                    0.5
                ],
                dtype=np.float32,
            ),
            dtype=np.float32
        )

        return observation_space

