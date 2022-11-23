import random
from gobigger.envs import create_env_custom

from learning import LearningBase
from nnlearner import NNLearner

NUM_GAMES = 1
NUM_PLAYERS = 4
FRAME_LIMIT = 2400 # 2 minutes
MAP_WIDTH = 100
MAP_HEIGHT = 100
SAVE_VIDEO = True
SAVE_FRAME = False

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
                save_dir='.',
                save_name_prefix='gamevod'
            ),
            by_frame = dict(
                save_frame=(True if SAVE_FRAME else False),
                save_dir='.',
                save_name_prefix='gameframes'
            )
        )
    )
    env = create_env_custom(type='st', cfg=cfg)

    alpha = 0.95
    decay = 0.95

    for e in range(NUM_GAMES):
        obs = env.reset()

        f = 0
        team_info = env.get_team_infos()
        actions = {k[0]: [random.uniform(-1,1), random.uniform(-1,1), -1] for k in team_info}
        prev_state = None
        while True:
            f = f + 1
            obs, rew, done, info = env.step(actions)
            
            _global_state, player_state = obs
            print(rew)

            #print('[{}] leaderboard={}'.format(f, obs[0]['leaderboard']))
            if prev_state is not None:
                ### do learning here
                learning_impl.step(prev_state, actions, player_state, rew)
                ###

            ### apply learned model
            actions.update(learning_impl.apply(player_state, alpha))
            print(actions)
            ###

            if done:
                print('finish game!')
                break
            prev_state = player_state
        alpha *= decay
    env.close()

if __name__=='__main__':
    run_game(NNLearner(4))
