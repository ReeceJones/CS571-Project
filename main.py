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
SAVE_VIDEO = True
SAVE_FRAME = False
LEARNING_WINDOW = 10

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
        )#,
        #manager_settings=dict(
        #    player_manager=dict(
        #        ball_settings=dict(
        #            score_decay_rate_per_frame=0.0
        #        )
        #    )
        #)
    )
    env = create_env_custom(type='st', cfg=cfg, step_mul=30)

    learning_impl.train()

    alpha = 0.3
    decay = 0.95

    accumulated_reward = [0.0] * NUM_PLAYERS

    reward_discount = 0.95
    current_reward_discount = 1.0

    #for e in range(NUM_GAMES):
    while True:
        obs = env.reset()

        f = 0
        team_info = env.get_team_infos()
        actions = {k[0]: [random.uniform(-1,1), random.uniform(-1,1), -1] for k in team_info}
        starting_state = None
        starting_actions = None
        while True:
            obs, rew, done, info = env.step(actions)

            current_reward_discount *= reward_discount

            for (i, r) in enumerate(rew):
                accumulated_reward[i] += current_reward_discount * r
            
            _global_state, player_state = obs
            print(rew)

            #print('[{}] leaderboard={}'.format(f, obs[0]['leaderboard']))
            if starting_state is not None and f % LEARNING_WINDOW == 0:
                ### do learning here
                learning_impl.step(starting_state, starting_actions, accumulated_reward)
                for i in range(len(accumulated_reward)):
                    accumulated_reward[i] = 0.0

            ### apply learned model
            force_random = f % LEARNING_WINDOW == 0
            actions.update(learning_impl.apply(player_state, alpha, force_random))
            print(actions)
            ###

            if done:
                print('finish game!')
                learning_impl.save()
                alpha *= decay
                break

            if f % LEARNING_WINDOW == 0:
                starting_state = player_state
                starting_actions = actions.copy()
                current_reward_discount = 1.0

            f = f + 1
    env.close()

if __name__=='__main__':
    run_game(NNLearner(4))
