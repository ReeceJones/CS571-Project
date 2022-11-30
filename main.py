import random
from gobigger.envs import create_env_custom

from learning import LearningBase
from nnlearner import NNLearner

NUM_GAMES = 50
NUM_PLAYERS = 4
FRAME_LIMIT = 5 * 1200 # 5 minutes
#MAP_WIDTH = 100
#MAP_HEIGHT = 100
MAP_WIDTH = 75
MAP_HEIGHT = 75
SAVE_VIDEO = False
SAVE_FRAME = True
LEARNING_WINDOW = 5

TRAIN_MODE = 0
TEST_MODE = 1
MODE = TEST_MODE

NUM_ENVS = 1

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
                save_dir='/tmp',
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
                    score_decay_rate_per_frame=0.0
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
                actions[env_i].update(learning_impl.apply(batched_player_state[env_i], alpha, False, MODE == TEST_MODE))
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
