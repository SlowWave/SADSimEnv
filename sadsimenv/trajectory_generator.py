import tomli
import os
import numpy as np

# get config data
with open(os.path.join(os.path.join(os.path.dirname(__file__), "configs"), "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class TrajectoryGenerator():
    def __init__(self):
        pass

    def reset(self):
        pass

    def generate_trajectory(self):
        pass

    def get_next_quaternion(self):
        pass

