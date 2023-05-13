import numpy as np
import tomli


# get config data
with open("configs/config.toml", "rb") as config_file:
    CFG = tomli.load(config_file)


class PerturbationsModel():
    def __init__(self):
        
        self.n_constant_perturbations = CFG['force_model']['perturbations']['n_constant_perturbations']
        self.constant_perturbations_amplitudes = CFG['force_model']['perturbations']['constant_perturbations_amplitudes']
        self.constant_perturbations_times = CFG['force_model']['perturbations']['constant_perturbations_times']
        self.constant_perturbation_frames = CFG['force_model']['perturbations']['constant_perturbation_frames']
        self.n_sinusoidal_perturbations = CFG['force_model']['perturbations']['n_sinusoidal_perturbations']
        self.sinusoidal_perturbations_amplitudes = CFG['force_model']['perturbations']['sinusoidal_perturbations_amplitudes']
        self.sinusoidal_perturbations_periods = CFG['force_model']['perturbations']['sinusoidal_perturbations_periods']
        self.sinusoidal_perturbations_frames = CFG['force_model']['perturbations']['sinusoidal_perturbations_frames']

        self.torques = self._get_torques()

    def _get_torques(self):

        torques = {
            'constant_perturbations_amplitudes': list(),
            'constant_perturbations_times': list(),
            'constant_perturbation_frames': list(),
            'sinusoidal_perturbations_amplitudes': list(),
            'sinusoidal_perturbations_frames': list()
        }

        # add constant perturbations
        for idx in range(self.n_constant_perturbations):

            torques['constant_perturbations_amplitudes'].append(
                self.constant_perturbations_amplitudes[idx]
            )

            torques['constant_perturbations_times'].append(
                self.constant_perturbations_times[idx]
            )

            torques['constant_perturbation_frames'].append(
                self.constant_perturbation_frames[idx]
            )

        # add sinusoidal perturbations
        for i in range(self.n_sinusoidal_perturbations):

            perturbation_lambdas = list()

            for j in range(3):
                if self.sinusoidal_perturbations_periods[i][j] > 0:
                    perturbation_lambdas.append(
                        lambda t, ii=i, jj=j: self.sinusoidal_perturbations_amplitudes[ii][jj] * np.cos(2 * np.pi * 1 / self.sinusoidal_perturbations_periods[ii][jj] * t)
                    )
                else:
                    perturbation_lambdas.append(
                        lambda t: 0
                    )

            torques['sinusoidal_perturbations_amplitudes'].append(perturbation_lambdas)

            torques['sinusoidal_perturbations_frames'].append(
                self.sinusoidal_perturbations_frames[i]
            )

        return torques

    def ode(self, t, x):

        torques = np.array([0.0, 0.0, 0.0])

        # build rotation matrix
        rotation_matrix = np.array(
            [
                [1 - 2 * x[2]**2 - 2 * x[3]**2, 2 * x[1] * x[2] - 2 * x[0] * x[3], 2 * x[1] * x[3] + 2 * x[0] * x[2]],
                [2 * x[1] * x[2] + 2 * x[0] * x[3], 1 - 2 * x[1]**2 - 2 * x[3]**2, 2 * x[2] * x[3] - 2 * x[0] * x[1]],
                [2 * x[1] * x[3] - 2 * x[0] * x[2], 2 * x[2] * x[3] + 2 * x[0] * x[1], 1 - 2 * x[1]**2 - 2 * x[2]**2]
            ],
            dtype=np.float32
        )

        # compute constant disturbances
        for idx, time_range in enumerate(self.torques['constant_perturbations_times']):

            if time_range[0] <= t and time_range[1] >= t:

                if self.torques['constant_perturbation_frames'][idx] == 'rotating':
                    torques += np.array(self.torques['constant_perturbations_amplitudes'][idx])

                elif self.torques['constant_perturbation_frames'][idx] == 'fixed':
                    torque = np.array(self.torques['constant_perturbations_amplitudes'][idx])
                    torques += np.dot(torque, rotation_matrix)

        # compute sinusoidal disturbances
        for idx, lambda_torque in enumerate(self.torques['sinusoidal_perturbations_amplitudes']):

            if self.torques['sinusoidal_perturbations_frames'][idx] == 'rotating':
                torques += np.array(
                    [
                        lambda_torque[0](t),
                        lambda_torque[1](t),
                        lambda_torque[2](t)
                    ]
                )

            elif self.torques['sinusoidal_perturbations_frames'][idx] == 'fixed':
                torque = np.array(
                    [
                        lambda_torque[0](t),
                        lambda_torque[1](t),
                        lambda_torque[2](t)
                    ]
                )
                torques += np.dot(torque, rotation_matrix)

        return torques


class AttitudeDynamicsModel():
    def __init__(self):

        ode_map = {
            "quaternion": self.quaternion_ode
        }

        self.ode = ode_map[CFG['force_model']['attitude_dynamics']['ode']]

    def quaternion_ode(self, x, u, d, inertia_matrix):

        # get spacecraft angular velocity components
        sc_w = np.array([x[4], x[5], x[6]])

        # get spacecraft quaternion matrix
        sc_q_matrix = np.matrix(
            [
                [-x[1], -x[2], -x[3]],
                [x[0], -x[3], x[2]],
                [x[3], x[0], -x[1]],
                [-x[2], x[1], x[0]]
            ]
        )

        # compute derivatives
        sc_qe_dot = np.dot(sc_q_matrix, sc_w) / 2
        sc_w_dot = - np.dot(inertia_matrix.I, np.cross(sc_w, np.dot(inertia_matrix.A, sc_w))) + \
            np.dot(inertia_matrix.I, u) + np.dot(inertia_matrix.I, d)

        x_dot = np.concatenate((sc_qe_dot.A, sc_w_dot.A), axis=None)

        return x_dot


class ForceModel():
    def __init__(self):

        self.attitude_dynamics = AttitudeDynamicsModel()
        self.perturbations = PerturbationsModel()
        self.attitude_ode = self.attitude_dynamics.ode
        self.perturbations_ode = self.perturbations.ode
        self.use_perturbations = CFG['force_model']['use_perturbations']

    def ode(self, t, x, u, inertia_matrix):

        if self.use_perturbations:

            # compute torque disturbances
            disturbances = self.perturbations_ode(t, x)

        else:
            disturbances = np.array([0, 0, 0])

        # compute spacecraft attitude
        x_dot = self.attitude_ode(x, u, disturbances, inertia_matrix)

        return x_dot



if __name__ == "__main__":

    pass