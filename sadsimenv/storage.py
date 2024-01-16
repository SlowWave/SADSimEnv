import os
import tomli
import numpy as np
import matplotlib.pyplot as plt
from animation import Animation

# get config data
with open(os.path.join(os.path.dirname(__file__), "configs", "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class Storage():
    def __init__(self):

        self.time_steps = None
        self.is_last_step = None
        self.quaternions = None
        self.quaternion_errors = None
        self.angular_errors = None
        self.angular_velocities = None
        self.actions = None

        self.animation_utils = Animation()

        self.obs_model = CFG['environment']['observation_space']['model']
        self.obs_model_map = {
            "model_1": self._get_obs_model_1,
            "model_2": self._get_obs_model_2,
        }

    def reset(self,
        quaternion,
        quaternion_error,
        angular_error,
        angular_velocity
    ):

        self.time_steps = [0]
        self.is_last_step = False
        self.quaternions = [quaternion]
        self.quaternion_errors = [quaternion_error]
        self.angular_errors = [angular_error]
        self.angular_velocities = [angular_velocity]
        self.actions = [np.array([0, 0, 0])]

    def update_records(self,
        time_step,
        is_last_step,
        quaternion,
        quaternion_error,
        angular_error,
        angular_velocity,
        action
    ):

        self.time_steps.append(time_step)
        self.is_last_step = is_last_step
        self.quaternions.append(quaternion)
        self.quaternion_errors.append(quaternion_error)
        self.angular_errors.append(angular_error)
        self.angular_velocities.append(angular_velocity)
        self.actions.append(action)

    def get_env_states(self):

        return self.obs_model_map[self.obs_model]()

    def render_animation(self, target_quaternion, time_step):

        self.animation_utils.animate(self.quaternion_errors, target_quaternion, time_step)


    def plot_results(self):

        # plot quaternions
        plt.figure()
        plt.plot(
            self.time_steps,
            self.quaternions,
            label=['q0', 'q1', 'q2', 'q3']
        )
        plt.xlabel('t [s]')
        plt.ylabel('quaternions')
        plt.legend()
        plt.grid()

        # plot quaternion errors
        plt.figure()
        plt.plot(
            self.time_steps,
            self.quaternion_errors,
            label=['q0_err', 'q1_err', 'q2_err', 'q3_err']
        )
        plt.xlabel('t [s]')
        plt.ylabel('quaternion errors')
        plt.legend()
        plt.grid()

        # plot angular errors
        plt.figure()
        plt.plot(self.time_steps, self.angular_errors)
        plt.xlabel('t [s]')
        plt.ylabel('angular error [deg]')
        plt.grid()

        # plot angular velocities
        plt.figure()
        plt.plot(
            self.time_steps,
            self.angular_velocities,
            label=['w1', 'w2', 'w3']
        )
        plt.xlabel('t [s]')
        plt.ylabel('angular velocoties [rad/s]')
        plt.legend()
        plt.grid()

        ##
        fig, axs = plt.subplots(3)
        
        # w1 = self.angular_velocities[:, 0]
        # w2 = self.angular_velocities[:, 1]
        # w3 = self.angular_velocities[:, 2]

        w1 = [item[0] for item in self.angular_velocities]
        w2 = [item[1] for item in self.angular_velocities]
        w3 = [item[2] for item in self.angular_velocities]

        axs[0].plot(
            self.time_steps,
            w1,
        )
                
        axs[0].set_ylabel('DC BUS Voltage [V]', fontsize=7)
        axs[0].grid()
        
        axs[1].plot(
            self.time_steps,
            w2,
        )
                
        axs[1].set_ylabel('DC BUS Current [A]', fontsize=7)
        axs[1].grid()
        
        axs[2].plot(
            self.time_steps,
            w3,
        )

        axs[2].set_ylabel('DC BUS Power [W]', fontsize=7)
        axs[2].grid()

        plt.xlabel('Time [s]', fontsize=7)
        fig.tight_layout()

        plt.show()

        ##


        # plot actions
        plt.figure()
        plt.plot(
            self.time_steps,
            self.actions,
            label=['a1', 'a2', 'a3']
        )
        plt.xlabel('t [s]')
        plt.ylabel('actions [Nm]')
        plt.legend()
        plt.grid()

        plt.show()


    def _get_obs_model_1(self):

        # get current quaternion
        quaternion = self.quaternions[-1]

        # get current angular velocity
        angular_velocity = self.angular_velocities[-1]

        # aggregate states
        states = np.concatenate((quaternion, angular_velocity), axis=None)

        return states


    def _get_obs_model_2(self):

        # get current quaternion
        quaternion = self.quaternions[-1]

        # get current angular velocity
        angular_velocity = self.angular_velocities[-1]

        # get last action
        try:
            action = self.actions[-1 * CFG['environment']['n_skipped_frames'] - 1]
        except:
            action = np.array([0, 0, 0])

        # aggregate states
        states = np.concatenate((quaternion, angular_velocity, action), axis=None)

        return states