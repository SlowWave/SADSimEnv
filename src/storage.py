import numpy as np
import matplotlib.pyplot as plt
from animation import Animation

class Storage():
    def __init__(self):

        self.time_steps = None
        self.quaternions = None
        self.quaternion_errors = None
        self.angular_errors = None
        self.angular_velocities = None
        self.actions = None

        self.animation_utils = Animation()

    def reset(self,
        quaternion,
        quaternion_error,
        angular_error,
        angular_velocity
    ):

        self.time_steps = [0]
        self.quaternions = [quaternion]
        self.quaternion_errors = [quaternion_error]
        self.angular_errors = [angular_error]
        self.angular_velocities = [angular_velocity]
        self.actions = [np.array([0, 0, 0])]

    def update_records(self,
        time_step,
        quaternion,
        quaternion_error,
        angular_error,
        angular_velocity,
        action
    ):

        self.time_steps.append(time_step)
        self.quaternions.append(quaternion)
        self.quaternion_errors.append(quaternion_error)
        self.angular_errors.append(angular_error)
        self.angular_velocities.append(angular_velocity)
        self.actions.append(action)

    def get_env_states(self):

        # get current quaternion
        quaternion = self.quaternions[-1]

        # get current angular velocity
        angular_velocity = self.angular_velocities[-1]

        states = np.concatenate((quaternion, angular_velocity), axis=None)

        return states

    def render_animation(self, target_quaternion, time_step):

        self.animation_utils.animate(self.quaternion_errors, target_quaternion, time_step)


    def plot_results(self):

        # plot quaternions
        plt.figure()
        plt.plot(self.time_steps, self.quaternions)
        plt.xlabel('t [s]')
        plt.ylabel('quaternions')
        plt.grid()

        # plot quaternion errors
        plt.figure()
        plt.plot(self.time_steps, self.quaternion_errors)
        plt.xlabel('t [s]')
        plt.ylabel('quaternion errors')
        plt.grid()

        # plot angular errors
        plt.figure()
        plt.plot(self.time_steps, self.angular_errors)
        plt.xlabel('t [s]')
        plt.ylabel('angular errors')
        plt.grid()

        # plot angular velocities
        plt.figure()
        plt.plot(self.time_steps, self.angular_velocities)
        plt.xlabel('t [s]')
        plt.ylabel('angular velocoties')
        plt.grid()

        # plot actions
        plt.figure()
        plt.plot(self.time_steps, self.actions)
        plt.xlabel('t [s]')
        plt.ylabel('actions')
        plt.grid()

        plt.show()
