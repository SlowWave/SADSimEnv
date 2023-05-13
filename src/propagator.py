import tomli
import numpy as np
from force_model import ForceModel
from scipy.integrate import solve_ivp

# get config data
with open("configs/config.toml", "rb") as config_file:
    CFG = tomli.load(config_file)

class Propagator():
    def __init__(self):

        self.integration_step = CFG['propagator']['integration_step']
        self.time_horizon = CFG['propagator']['time_horizon']
        self.integration_method = CFG['propagator']['integration_method']
        self.force_model = ForceModel()
        self.current_time = None

    def reset(self):
        
        self.current_time = 0

    def propagate(self, states, action, inertia_matrix):

        # integrate ode
        ode_solution = self._integrate_ode(states, action, inertia_matrix)

        # update current time
        self.current_time = ode_solution.t[-1]

        # check termination condition
        if self.current_time >= self.time_horizon:
            is_last_step = True
        else:
            is_last_step = False

        return is_last_step, ode_solution

    def _integrate_ode(self, states, action, inertia_matrix):

        ode_solution = solve_ivp(
            fun=self.force_model.ode,
            t_span=[self.current_time,self.current_time + self.integration_step],
            y0=states,
            method=self.integration_method,
            dense_output=False,
            args=(action, inertia_matrix)
        )

        return ode_solution

if __name__ == "__main__":

    states = np.array([1, 0, 0, 0, 0, 0, 0])
    action = np.array([0, 0, 0.1])
    inertia_matrix = np.matrix(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    prop = Propagator()

    prop.reset()
    sol, b = prop.propagate(states, action, inertia_matrix)

    x = sol.y[0:-1]
    x1 = np.take(sol.y, -1, -1)

    print(x1)
    q = x1[0:4]
    w = x1[4:]

    print(q, w)

