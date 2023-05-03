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
        self.n_sinusoidal_perturbations = CFG['force_model']['perturbations']['n_sinusoidal_perturbations']
        self.sinusoidal_perturbations_amplitudes = CFG['force_model']['perturbations']['sinusoidal_perturbations_amplitudes']
        self.sinusoidal_perturbations_periods = CFG['force_model']['perturbations']['sinusoidal_perturbations_periods']

    def get_torques(self, t):

        perturbation = np.array([0.0, 0.0, 0.0])

        # add constant perturbations
        for idx in range(self.n_constant_perturbations):

            perturbation = self._add_constant_perturbation(perturbation, t, idx)

        # add sinusoidal perturbations
        for idx in range(self.n_sinusoidal_perturbations):

            perturbation = self._add_sinusoidal_perturbation(perturbation, t, idx)

        return perturbation

    def _add_constant_perturbation(self, perturbation, t, idx):

        if self.constant_perturbations_times[idx] == 0:
            perturbation += np.array(self.constant_perturbations_amplitudes[idx])

        elif self.constant_perturbations_times[idx] == t:
            perturbation += np.array(self.constant_perturbations_amplitudes[idx])

        return perturbation

    def _add_sinusoidal_perturbation(self, perturbation, t, idx):

        perturbation += np.array(
            [
                self.sinusoidal_perturbations_amplitudes[idx][0] * np.cos(2 * np.pi * 1 / self.sinusoidal_perturbations_periods[idx][0] * t),
                self.sinusoidal_perturbations_amplitudes[idx][1] * np.cos(2 * np.pi * 1 / self.sinusoidal_perturbations_periods[idx][1] * t),
                self.sinusoidal_perturbations_amplitudes[idx][2] * np.cos(2 * np.pi * 1 / self.sinusoidal_perturbations_periods[idx][2] * t)

            ]
        )

        return perturbation


class AttitudeDynamicsModel():
    def __init__(self):

        ode_map = {
            "quaternion": self.quaternion_ode
        }

        self.ode = ode_map[CFG['force_model']['attitude_dynamics']['ode']]

    def quaternion_ode(self, t, x, u, d, inertia_matrix):

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

    def get_perturbations(self, t):

        perturbations = self.perturbations.get_torques(t)

        return perturbations
    

if __name__ == "__main__":

    pass