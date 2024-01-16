import os
import tomli
import numpy as np

# get config data
with open(os.path.join(os.path.dirname(__file__), "configs", "config.toml"), "rb") as config_file:
    CFG = tomli.load(config_file)


class Inertia():
    def __init__(self):

        # define moments of inertia
        if CFG['spacecraft']['inertia']['use_random_moi']:

            self.moi = list()
            
            for idx in range(3):
                self.moi.append(np.random.uniform(
                    CFG['spacecraft']['inertia']['random_moi_min'][idx],
                    CFG['spacecraft']['inertia']['random_moi_max'][idx],
                ))

        else:
            self.moi = CFG['spacecraft']['inertia']['moi']

        # define products of inertia
        if CFG['spacecraft']['inertia']['use_random_poi']:        

            self.poi = list()
            
            for idx in range(3):
                self.poi.append(np.random.uniform(
                    CFG['spacecraft']['inertia']['random_poi_min'][idx],
                    CFG['spacecraft']['inertia']['random_poi_max'][idx],
                ))

        else:
            self.poi = CFG['spacecraft']['inertia']['poi']

        # define tensor of inertia
        self.matrix = np.matrix(
            [
                [self.moi[0], self.poi[0], self.poi[1]],
                [self.poi[0], self.moi[1], self.poi[2]],
                [self.poi[1], self.poi[2], self.moi[2]],
            ]
        )


class Attitude():
    def __init__(self):

        self.angular_velocity = None
        self.target_quaternion = None
        self.current_quaternion = None
        self.quaternion_error = None
        self.angular_error = None

    def init_states(self):

        # set random seed
        if CFG['environment']['use_random_seed']:
            np.random.seed(CFG['environment']['random_seed'])

        # get initial angular velocity
        self._init_angular_velocity()

        # get target quaternion
        self._init_target_quaternion()

        # get initial quaternion
        self._init_quaternion()

    def update_states(self, states):

        # update angular velocity
        self.angular_velocity = states[4:]

        # normalize and update quaternion
        self.current_quaternion = self._normalize_quaternion(states[0:4])

        # update quaternion error
        self.quaternion_error = self._get_quaternion_error(self.current_quaternion)

        # update angular error
        self.angular_error = self._get_angular_error(self.quaternion_error)

    def get_states(self):

        # get current quaternion
        quaternion = self.current_quaternion

        # get current angular velocity
        angular_velocity = self.angular_velocity

        states = np.concatenate((quaternion, angular_velocity), axis=None)

        return states

    def _init_angular_velocity(self):

        self.angular_velocity = np.array(CFG['spacecraft']['attitude']['initial_angular_velocity'])

    def _init_target_quaternion(self):
        
        self.target_quaternion = np.array(CFG['spacecraft']['attitude']['target_quaternion'])

    def _init_quaternion(self):

        # define maximum and minimum initial angular error boundaries
        max_error = CFG['spacecraft']['attitude']['initial_angular_error_max']
        min_error = CFG['spacecraft']['attitude']['initial_angular_error_min']

        while True:

            # generate a random initial quaternion
            raw_quaternion = 2 * np.random.random_sample((1, 4))[0] - 1

            # normalize the quaterion
            unit_quaternion = self._normalize_quaternion(raw_quaternion)

            # compute the quaternion error
            quaternion_error = self._get_quaternion_error(unit_quaternion)

            # get angular error [deg]
            angular_error = self._get_angular_error(quaternion_error)

            # verify that the angular error between the initial and the target quaternion is within the provided boundaries
            if angular_error >= min_error and angular_error <= max_error:
                break

        # store quaternions
        self.current_quaternion = unit_quaternion
        self.quaternion_error = quaternion_error
        self.angular_error = angular_error

    def _normalize_quaternion(self, quaternion):

        quaternion_norm = np.sqrt(np.sum(quaternion**2))
        unit_quaternion = quaternion / quaternion_norm

        return unit_quaternion

    def _get_quaternion_error(self, quaternion):

        # get quaternion conjugate
        q_conj = np.array(
            [
                quaternion[0],
                - quaternion[1],
                - quaternion[2],
                - quaternion[3]
            ]
        )

        # get Hamilton product
        q_error = np.array(
            [
                self.target_quaternion[0] * q_conj[0] - self.target_quaternion[1] * q_conj[1] - self.target_quaternion[2] * q_conj[2] - self.target_quaternion[3] * q_conj[3],
                self.target_quaternion[0] * q_conj[1] + self.target_quaternion[1] * q_conj[0] + self.target_quaternion[2] * q_conj[3] - self.target_quaternion[3] * q_conj[2],
                self.target_quaternion[0] * q_conj[2] - self.target_quaternion[1] * q_conj[3] + self.target_quaternion[2] * q_conj[0] + self.target_quaternion[3] * q_conj[1],
                self.target_quaternion[0] * q_conj[3] + self.target_quaternion[1] * q_conj[2] - self.target_quaternion[2] * q_conj[1] + self.target_quaternion[3] * q_conj[0]
            ]
        )

        return q_error


    def _get_angular_error(self, quaternion_error):

        return np.rad2deg(2 * np.arccos(quaternion_error[0]))

   

class Spacecraft():
    def __init__(self):

        # set random seed
        if CFG['environment']['use_random_seed']:
            np.random.seed(CFG['environment']['random_seed'])

        # initialize spacecraft components
        self.inertia = Inertia()
        self.attitude = Attitude()


    def reset(self):

        self.attitude.init_states()

    def get_prop_states(self):

        return self.attitude.get_states()
    
    def update_states(self, ode_solution):

        # get states from ode solution object
        states = np.take(ode_solution.y, -1, -1)

        # update attitude states
        self.attitude.update_states(states)



if __name__ == "__main__":

    # i = Inertia()
    # print(i.poi)

    s = Spacecraft()
    s.reset()
    st = s.get_prop_states()
    print(st)