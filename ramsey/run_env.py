from ramsey_env import RamseyGame

environment = RamseyGame(n_nodes = 6, k_clique = 3)
episodes = 4

for episode in range(episodes):

    done = False
    obs = environment.reset()
    while not done:
        random_action = environment.action_space.sample()
        print('action', random_action)
        obs, reward, done, info = environment.step(random_action)
        print('reward', reward)
        environment.render()