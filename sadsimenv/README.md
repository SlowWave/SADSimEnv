
# SADSimEnv

https://github.com/SlowWave/SADSimEnv/assets/95315431/7ba0de66-8c63-4300-ae1a-cc170400280e

## Project Description

Welcome to the SADSimEnv (Spacecraft Attitude Dynamics Simulation Environment) project! This custom gymnasium environment allows you to simulate the dynamics of a spacecraft for reinforcement learning and control experiments.


## Installation

To use the environment, please follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/SlowWave/SADSimEnv.git
    ```

2. Install the required dependencies by navigating to the project directory and running:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```python
    from environment import SpacecraftEnv

    # initialize and reset the environment
    env = SpacecraftEnv()
    observation, info = env.reset()
    
    # set the number of simulation steps
    n_steps = 1000
    
    for _ in range(n_steps):
    
        # randomly sample an action from the action space
        action = env.action_space.sample()
        
        # perform an environment step
        observation, reward, terminated, truncated, info = env.step(action)
    
    # plot data acquired during simulation and render the animation
    env.plot_results()
    env.render_animation()
```

## SpacecraftEnv Object

The `SpacecraftEnv` object is divided into several classes, each one dedicated to performing a specific function:

- `ObservationSpaceModel` object: used to define the environment observation space
- `ActionSpaceModel` object: used to define the environment action space
- `Spacecraft` object: used to store the spacecraft properties and the current spacecraft state $s_t$
- `Propagator` object: used to propagate the spacecraft state from $s_t$ to $s_{t+1}$ according to the spacecraft dynamic equations and the integration properties
- `Storage` object: used to store the list of spacecraft states and agent actions
- `RewardModel` object: used to compute the agent reward $r_t$ starting from data contained inside `Storage` object


![1](https://github.com/SlowWave/spacecraft_env/assets/95315431/1bd32c64-0edf-4249-95d2-3b846a4d36fe)


![2](https://github.com/SlowWave/spacecraft_env/assets/95315431/650cbde8-c5a3-4a14-a982-18757b381d95)


## Customization

This simple enviroment has been developed with a focus on modularity, allowing users to easily extend and customize its features to meet specific requirements.
There are two ways to customize the environment:
- Modify the `config.toml` file
- Modify the methods exposed by the classes used inside the environment

### Config File

`config.toml` is a configuration file written according to the TOML file format. It allows to configure several environment parameters in order to customize the simulations.

#### environment configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| n_skipped_frames | int | positive integers starting from 0 | defines the number of steps performed by the environment considering a null agent action. Refer to Frame-Skipping technique for more info |
| use_random_seed | bool | true, false | **currently not used** |
| random_seed | int | positive integers | **currently not used** |
| normalize_obs | bool | true, false | if true, environment observations are normalized |
| normalize_reward | bool | true, false | if true, environment rewards are normalized |

#### observation space configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| model | string | "model_1", "model_2", ..., "model_n" | defines the observation space model used inside the environment |

#### action space configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| model | string | "model_1", "model_2", ..., "model_n" | defines the action space model used inside the environment |

#### reward model configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| model | string | "model_1", "model_2", ..., "model_n" | defines the reward model used to compute the agent reward |

#### spacecraft inertia configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| moi | list of float | $$[I_{xx}, I_{yy}, I_{zz}]$$ where $I_{xx}, I_{yy}, I_{zz}$ are positive numbers | defines the spacecraft moment of inertia components that are used to build the inertia tensor |
| poi | list of float | $$[I_{xy}, I_{xz}, I_{yz}]$$ where $I_{xy}, I_{xz}, I_{yz}$ are positive numbers | defines the spacecraft product of inertia components that are used to build the inertia tensor |
| use_random_moi | bool | true, false | if true, a random moi is generated |
| use_random_poi | bool | true, false | if true, a random poi is generated |
| random_moi_max | list of float | $$[I_{xx}, I_{yy}, I_{zz}]$$ where $I_{xx}, I_{yy}, I_{zz}$ are positive numbers | defines the upper boudaries for random moi generation |
| random_moi_min | list of float | $$[I_{xx}, I_{yy}, I_{zz}]$$ where $I_{xx}, I_{yy}, I_{zz}$ are positive numbers | defines the lower boudaries for random moi generation |
| random_poi_max | list of float | $$[I_{xy}, I_{xz}, I_{yz}]$$ where $I_{xy}, I_{xz}, I_{yz}$ are positive numbers | defines the upper boudaries for random poi generation |
| random_poi_min | list of float | $$[I_{xy}, I_{xz}, I_{yz}]$$ where $I_{xy}, I_{xz}, I_{yz}$ are positive numbers | defines the lower boudaries for random poi generation |

#### spacecraft attitude configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| representation | string | "quaternions" | **currently not used** |
| initial_angular_error_max | float | $$(0, 360]$$ | defines the upper boundary for the initial angular error of the spacecraft, i.e. the error between the initial attitude quaternion and the target one |
| initial_angular_error_min | float | $$[0, 360]$$ | defines the lower boundary for the initial angular error of the spacecraft, i.e. the error between the initial attitude quaternion and the target one |
| target_quaternion | list of float | $$q_t = [q_0, q_1, q_2, q_3]$$ must be a unit quaternion | defines the target spacecraft attitude quaternion |
| initial_angular_velocity | list of float | $$[\omega_x, \omega_y, \omega_z]$$ where each component can be any float number | defines the initial spacecraft angular velocity in [rad/s] | 


#### force model configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| use_perturbations | bool | true, false | if true, torque perturbations are considered in the spacraft attitude dynamic equations |

#### attitude dynamics configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| ode | string | "quaternion" | defines the attitude dynamics model to be used in order to update the spacecraft state |

#### dynamics perturbations configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| n_constant_perturbations | int | positive integers starting from 0 | defines the number of constant perturbations to be included in the perturbations model |
| constant_perturbations_amplitudes | list of lists of floats | $$[a_{cst,1}, a_{cst,2}, \ldots, a_{cst,n}]$$ where $$a_{cst,n} = [a_{n,x}, a_{n,y}, a_{n,z}]$$ defines the amplitude of the $\text{n}^{th}$ constant perturbation. Each component of $a_{cst,n}$ can be any float number | defines the amplitudes of the constant torque perturbations |
| constant_perturbations_times | list of lists of floats | $$[t_{cst,1}, t_{cst,2}, \ldots, t_{cst,n}]$$ where $$t_{cst,n} = [t_{n,start}, t_{n,end}]$$ defines the duration of the $\text{n}^{th}$ constant perturbation in [sec]. Each component of $t_{cst,n}$ can be any float number | defines the time boundaries of the constant torque perturbations |
| constant_perturbations_frames | list of strings | $$[f_{cst,1}, f_{cst,2}, \ldots, f_{cst,n}]$$ where each component can be defined as "fixed" or "rotating" | defines the reference frames associated with each component of the constant torque perturbations list. If "fixed", the perturbation acts according to the *external* reference frame associated with the quaternion $[1, 0, 0, 0]$. If "rotating", the perturbation acts according to the *spacecraft* reference frame defined by the current attitude quaternion at time $t$: $q_t$ |
| n_sinusoidal_perturbations | int | positive integers starting from 0 | defines the number of sinusoidal perturbations to be included in the perturbations model |
| sinusoidal_perturbations_amplitudes | list of lists of floats | $$[a_{sin,1}, a_{sin,2}, \ldots, a_{sin,n}]$$ where $$a_{sin,n} = [a_{n,x}, a_{n,y}, a_{n,z}]$$ defines the amplitude of the $\text{n}^{th}$ sinusoidal perturbation. Each component of $a_{sin,n}$ can be any float number | defines the amplitudes of the sinusoidal torque perturbations |
| sinusoidal_perturbations_periods | list of lists of floats | $$[p_{sin,1}, p_{sin,2}, \ldots, p_{sin,n}]$$ where $$p_{sin,n} = [p_{n,x}, p_{n,y}, p_{n,z}]$$ defines the periods of the $\text{n}^{th}$ sinusoidal perturbation in [sec]. Each component of $p_{sin,n}$ can be any float number | defines the periods of the sinusoidal torque perturbations |
| sinusoidal_perturbations_frames | list of strings | $$[f_{sin,1}, f_{sin,2}, \ldots, f_{sin,n}]$$ where each component can be defined as "fixed" or "rotating" | defines the reference frames associated with each component of the sinusoidal torque perturbations list. If "fixed", the perturbation acts according to the *external* reference frame associated with the quaternion $[1, 0, 0, 0]$. If "rotating", the perturbation acts according to the *spacecraft* reference frame defined by the current attitude quaternion at time $t$: $q_t$ |

#### propagator configs

| Parameter Name | Format | Allowed Values | Description |
| -- | -- | -- | -- |
| integration_step | float | any float number greater than 0 | defines the time interval between $s_t$ and $s_{t+1}$ in [sec] |
| time_horizon | float | any float number greater than 0 | defines the duration of each episode in [sec] |
| integration_method | string | "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA" | defines the integration method used by the propagator. Refer to the `scipy.integrate.solve_ivp` method for more info |


## Future Updates

- Create a Reference Generator object capable of updating/modifying the target quaternion at each time step according to some user-defined settings.

## License

This project is licensed under the MIT license - see the [LICENSE](LICENSE) file for details.
