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

    def get_torques(self, t, rotation_matrix):

        perturbation = np.array([0.0, 0.0, 0.0])

        # add constant perturbations
        for idx in range(self.n_constant_perturbations):

            perturbation += self._get_constant_perturbation(t, idx, rotation_matrix)

        # add sinusoidal perturbations
        for idx in range(self.n_sinusoidal_perturbations):

            perturbation += self._get_sinusoidal_perturbation(t, idx, rotation_matrix)

        return perturbation

    def _get_constant_perturbation(self, t, idx, rotation_matrix):

        if self.constant_perturbations_times[idx] == 0:    
            perturbation = np.array(self.constant_perturbations_amplitudes[idx])

        elif self.constant_perturbations_times[idx] == t:
            perturbation = np.array(self.constant_perturbations_amplitudes[idx])

        if self.constant_perturbation_frames[idx] == "rotating":
            return perturbation

        elif self.constant_perturbation_frames[idx] == "fixed":
            perturbation = np.dot(perturbation, rotation_matrix)
            return perturbation
        else:
            return np.array([0.0, 0.0, 0.0])

    def _get_sinusoidal_perturbation(self, t, idx, rotation_matrix):

        perturbation_list = list()

        for i in range(3):
            if self.sinusoidal_perturbations_periods[idx][i] > 0:
                perturbation_list.append(
                    self.sinusoidal_perturbations_amplitudes[idx][i] * np.cos(2 * np.pi * 1 / self.sinusoidal_perturbations_periods[idx][i] * t)
                )
            else:
                perturbation_list.append(0)

        perturbation = np.array(perturbation_list)

        if self.sinusoidal_perturbations_frames[idx] == "rotating":
            return perturbation

        elif self.sinusoidal_perturbations_frames[idx] == "fixed":
            perturbation = np.dot(perturbation, rotation_matrix)
            return perturbation
        else:
            return np.array([0.0, 0.0, 0.0])


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

    def get_perturbations(self, t, quaternion):

        # build rotation matrix
        q0, q1, q2, q3 = quaternion
        rotation_matrix = np.array(
            [
                [1 - 2 * q2**2 - 2 * q3**2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2],
                [2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1**2 - 2 * q3**2, 2 * q2 * q3 - 2 * q0 * q1],
                [2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1**2 - 2 * q2**2]
            ],
            dtype=np.float32
        )

        # get perturbation torques
        perturbations = self.perturbations.get_torques(t, rotation_matrix)



        rotated_perturbations = np.dot(perturbations, rotation_matrix)

        return perturbations
    

if __name__ == "__main__":

    pass