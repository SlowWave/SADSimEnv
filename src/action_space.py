import tomli
import numpy as np
from gymnasium import spaces


# get config data
with open("configs/config.toml", "rb") as config_file:
    CFG = tomli.load(config_file)


class ActionSpaceModel():
    def __init__(self):

        # define mapping dictionary
        self.action_model_map = {
            "model_1": self._model_1,
            "model_2": self._model_2,
        }

        self.model = CFG['environment']['action_space']['model']

    def get_action_space(self):

        return self.action_model_map[self.model]()
    
    def _model_1(self):

        # define action space limits
        action_limit = np.array(
            [
                0.5,
                0.5,
                0.5
            ],
            dtype=np.float32,
        )

        # define observation space
        action_space = spaces.Box(
            -action_limit,
            action_limit,
            dtype=np.float32
        )

        return action_space
    
    def _model_2(self):
        pass



if __name__ == "__main__":

    act_model = ActionSpaceModel()
    act_space = act_model.get_action_space()
    sample = act_space.sample()
    print(sample)