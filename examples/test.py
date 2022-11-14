import random
from gobigger.envs import create_env

env = create_env('st_t2p2')
obs = env.reset()
for i in range(1000):
    actions = {0: [random.uniform(-1, 1), random.uniform(-1, 1), -1],
               1: [random.uniform(-1, 1), random.uniform(-1, 1), -1],
               2: [random.uniform(-1, 1), random.uniform(-1, 1), -1],
               3: [random.uniform(-1, 1), random.uniform(-1, 1), -1]}
    obs, rew, done, info = env.step(actions)
    print('[{}] leaderboard={}'.format(i, obs[0]['leaderboard']))
    if done:
        print('finish game!')
        break
env.close()
