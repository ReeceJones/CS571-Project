import random
from gobigger.envs import create_env_custom
import matplotlib.pyplot as plt
from learning import LearningBase
from nnlearner import NNLearner
import numpy as np
NUM_GAMES = 2
NUM_PLAYERS = 4
FRAME_LIMIT = 5 * 1200 # 5 minutes
#MAP_WIDTH = 100
#MAP_HEIGHT = 100
MAP_WIDTH = 75
MAP_HEIGHT = 75
SAVE_VIDEO = False
SAVE_FRAME = False
LEARNING_WINDOW = 5

TRAIN_MODE = 0
TEST_MODE = 1
MODE = TEST_MODE

NUM_ENVS = 100

GEN_GRAPH = False
RANDOM_PLAYERS = [2,3]

def run_game(learning_impl: LearningBase):
    # create an environment
    cfg=dict(
        team_num=NUM_PLAYERS,
        player_num_per_team=1,
        frame_limit=FRAME_LIMIT,
        map_width=MAP_WIDTH,
        map_height=MAP_HEIGHT,
        playback_settings=dict(
            playback_type=('by_video' if SAVE_VIDEO else ('by_frame' if SAVE_FRAME else 'none')),
            by_video = dict(
                save_video=(True if SAVE_VIDEO else False),
                save_fps=10,
                save_resolution=480,
                save_all=True,
                save_partial=False,
                #save_dir='/tmp',
                save_name_prefix='gamevod'
            ),
            by_frame = dict(
                save_frame=(True if SAVE_FRAME else False),
                save_dir='/tmp',
                save_name_prefix='gameframes'
            )
        ),
        manager_settings=dict(
            player_manager=dict(
                ball_settings=dict(
                    score_decay_rate_per_frame=0.00005,
                    score_decay_min=2600
                )
            )
        )
    )
    envs = []
    for _ in range(NUM_ENVS):
        envs.append(create_env_custom(type='st', cfg=cfg, step_mul=30))
        cfg['playback_settings']['by_video']['save_video'] = False

    alpha = 0.3
    decay = 0.95

    if MODE == TRAIN_MODE:
        learning_impl.train()
    else:
        alpha = 0.0
        learning_impl.load()
        learning_impl.eval()

    accumulated_reward = [[0.0] * NUM_PLAYERS for e in envs]

    reward_discount = 0.95
    current_reward_discount = 1.0
    dist_from_food = {}
    dist_from_enemy = {}
    leaderboards = {}
    #for e in range(NUM_GAMES):
    while True:
        obs = [e.reset() for e in envs]

        f = 0
        actions = [{k[0]: [random.uniform(-1,1), random.uniform(-1,1), -1] for k in e.get_team_infos()} for e in envs]
        starting_state = [None] * NUM_ENVS
        starting_actions = [None] * NUM_ENVS
        while True:
            done = False

            batched_player_state = []

            if MODE == TRAIN_MODE:
                current_reward_discount *= reward_discount

            for (env_i, e) in enumerate(envs):
                obs, rew, done, info = e.step(actions[env_i])
                leaderboards[env_i] = obs[0]['leaderboard']
                print('[{}] leaderboard={}'.format(f, obs[0]['leaderboard']))

                if MODE == TRAIN_MODE:
                    for (i, r) in enumerate(rew):
                        accumulated_reward[env_i][i] += current_reward_discount * r
                
                _global_state, player_state = obs
                batched_player_state.append(player_state)

            if MODE == TRAIN_MODE:
                if starting_state[0] is not None and f % LEARNING_WINDOW == 0:
                    ### do learning here
                    learning_impl.step(starting_state, starting_actions, accumulated_reward)
                    for i in range(len(accumulated_reward)):
                        accumulated_reward = [[0.0] * NUM_PLAYERS for e in envs]

            for (env_i, e) in enumerate(envs):
                ### apply learned model
                action, performance_metric = learning_impl.apply(batched_player_state[env_i], alpha, RANDOM_PLAYERS, MODE == TEST_MODE)
                actions[env_i].update(action)
                for pid in performance_metric[0].keys():
                    if pid not in dist_from_food.keys():
                        dist_from_food[pid] = []
                    dist_from_food[pid].append(performance_metric[0][pid])
                    if pid not in dist_from_enemy.keys():
                        dist_from_enemy[pid] = []
                    dist_from_enemy[pid].append(performance_metric[1][pid])
                print(actions[env_i])
                ###

                if MODE == TRAIN_MODE:
                    if f % LEARNING_WINDOW == 0:
                        starting_state[env_i] = batched_player_state[env_i]
                        starting_actions[env_i] = actions[env_i].copy()
                        current_reward_discount = 1.0

            f = f + 1

            if done:
                print('finish game!')
                if GEN_GRAPH:
                    y = []
                    counter = 0
                    for pid in dist_from_food.keys():
                        new_list = []
                        new_list2 = []
                        for i in range(len(dist_from_food[pid])-10):
                            new_list.append(np.mean(dist_from_food[pid][i:i+10]))
                            new_list2.append(np.mean(dist_from_enemy[pid][i:i+10]))
                        dist_from_food[pid] = new_list
                        dist_from_enemy[pid] = new_list2
                        if(len(y) == 0):
                            y = [i for i in range(len(dist_from_food[pid]))]
                        
                        plt.plot(y, dist_from_food[pid], label=str(pid))
                        plt.plot(y, dist_from_enemy[pid])
                        break
                    plt.show()

                if RANDOM_PLAYERS is not None:
                    total_rank_learner = list()
                    total_rank_random = list()
                    for i, leaderboard in leaderboards.items():
                        for p, s in leaderboard.items():
                            if p in RANDOM_PLAYERS:
                                total_rank_random.append(s)
                            else:
                                total_rank_learner.append(s)
                    print(f'Average learner score ({NUM_ENVS=}):\t{np.mean(total_rank_learner)}, {np.std(total_rank_learner)}, {np.median(total_rank_learner)}')
                    print(f'Average baseline score ({NUM_ENVS=}):\t{np.mean(total_rank_random)}, {np.std(total_rank_random)}, {np.median(total_rank_random)}')

                if MODE == TEST_MODE:
                    for e in envs:
                        e.close()
                    return
                elif MODE == TRAIN_MODE:
                    learning_impl.save()
                    alpha *= decay
                break
    env.close()

if __name__=='__main__':
    run_game(NNLearner(4))
