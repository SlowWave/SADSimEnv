from environment import SpacecraftEnv

# initialize and reset the environment
env = SpacecraftEnv()
observation, info = env.reset()

# set the number of simulation steps
n_steps = 1000

for _ in range(n_steps):

    # randomly sample an action from the action space
    action = env.action_space.sample()
    # action = [0, 0, 0]
    
    # perform an environment step
    observation, reward, terminated, truncated, info = env.step(action)

# plot data acquired during simulation and render the animation
env.plot_results()
env.render_animation()