import os
import tomli
import numpy as np
from gymnasium import spaces


# get config data
with open(os.path.join(os.path.dirname(__file__), "configs", "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class ActionSpaceModel():
    def __init__(self):

        # define mapping dictionaries
        self.action_model_map = {
            "model_1": self._model_1,
            "model_2": self._model_2,
            "model_3": self._model_3,
            "model_5": self._model_5,
        }

        self.action_elaboration_map = {
            "model_1": self._elaborate_action_1,
            "model_2": self._elaborate_action_2,
            "model_3": self._elaborate_action_3,
            "model_5": self._elaborate_action_5,
        }

        self.model = CFG['environment']['action_space']['model']

    def get_action_space(self):

        return self.action_model_map[self.model]()
    
    def get_elaborated_action(self, action, storage):

        return self.action_elaboration_map[self.model](action, storage)

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

    def _elaborate_action_1(self, action, storage):

        return action

    def _model_2(self):

        # define action space limits
        action_limit = np.array(
            [
                0.2,
                0.2,
                0.2
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

    def _elaborate_action_2(self, action, storage):
        
        return action

    def _model_3(self):

        # define action space
        action_space = spaces.MultiDiscrete(
            nvec=np.array([21, 21, 21])
        )

        self.elaborate_action = True

        return action_space

    def _elaborate_action_3(self, action, storage):
        
        slope = 0.05
        y_intercept = - 0.5

        return action * slope + y_intercept

    def _model_5(self):

       # define action space
       action_space = spaces.MultiDiscrete(
        nvec=np.array([21, 21, 21])
       )

       return action_space

    def _elaborate_action_5(self, action, storage):

        # map raw action in the interval [-0.1, 0.1] 
        slope = 0.05
        y_intercept = - 0.5
        delta_torque = action * slope + y_intercept

        # apply delta_torque to past torque control input
        raw_torque = storage.actions[-1] + delta_torque

        # define ema window and alpha parameters
        ema_win = 5
        ema_alpha = 0.5   

        # aggregate actions to be filtered
        last_actions = storage.actions[- ema_win:]
        actions_tbf = last_actions + [raw_torque]

        # get first value of ema 
        ema_values = [actions_tbf[0]]
        
        # apply ema filter
        for i in range(1, len(actions_tbf)):
            ema = ema_alpha * actions_tbf[i] + (1 - ema_alpha) * ema_values[-1]
            ema_values.append(ema)

        action = ema_values[-1]

        # action saturation
        for i in range(len(action)):
            if action[i] > 0.5:
                action[i] = 0.5
            elif action[i] < - 0.5:
                action[i] = - 0.5 

        return action


if __name__ == "__main__":

    act_model = ActionSpaceModel()
    act_space = act_model.get_action_space()
    sample = act_space.sample()
    print(sample)