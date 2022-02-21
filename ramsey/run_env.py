from stable_baselines3.common.env_checker import check_env

from ramsey_env import RamseyGame

environment = RamseyGame(n_nodes = 6, k_clique = 3)

check_env(environment)
episodes = 1

for episode in range(episodes):

    done = False
    obs = environment.reset()
    while not done:
        random_action = environment.action_space.sample()
        print('action', random_action)
        obs, reward, done, info = environment.step(random_action)
        print('reward', reward)
        environment.render()