import torch
import torch.optim as optim
import torch.nn as nn
import math
import random

from itertools import product

from learning import LearningBase

CLONE_X = 0
CLONE_Y = 1
CLONE_RADIUS = 2
CLONE_PID = 8

class NNLearner(LearningBase):
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.im_layer1 = nn.Linear(158, 128)
            self.nonlin1 = nn.ReLU()
            self.im_layer2 = nn.Linear(128, 64)
            self.nonlin2 = nn.ReLU()
            self.im_layer3 = nn.Linear(64, 32)
            self.nonlin3 = nn.ReLU()
            self.im_layer4 = nn.Linear(32, 16)
            self.nonlin4 = nn.ReLU()
            self.out_layer = nn.Linear(16, 1)

        def forward(self,X):
            Y = self.im_layer1(X)
            Y = self.nonlin1(Y)
            Y = self.im_layer2(Y)
            Y = self.nonlin2(Y)
            Y = self.im_layer3(Y)
            Y = self.nonlin3(Y)
            Y = self.im_layer4(Y)
            Y = self.nonlin4(Y)
            Y = self.out_layer(Y)
            return Y

    def __init__(self, num_players):
        self.num_players = num_players
        self.model = [self.SimpleNN() for i in range(num_players)]
        self.optimizer = optim.Adam(list(self.model[0].parameters()))
        self.criterion = nn.MSELoss()

    def step(self, prev_states, actions, new_states, rew):
        """
        Apply outcomes of the previous actions to the learning model.
        """

        # Only take a step if at least one non-zero reward was seen
        # Most steps are just zero reward, and we don't want to overfit the
        # model to always output zero
        no_reward = all([r == 0.0 for r in rew])
        if no_reward:
            return

        for player_id in new_states.keys():
            prev_state = prev_states[player_id]
            action = actions[player_id]
            state = new_states[player_id]
            reward = [rew[player_id]]
            encoded, target = self.batch_encode(player_id, prev_state, action, reward)
            score = self.model[player_id](encoded.float())
            loss = self.criterion(score, target.float())
            self.optimizer.zero_grad()
            self.optimizer.step()
        """

        encoded, target = self.batch_encode(prev_states, actions, rew)

        # update gradients
        scores = self.model(encoded.float())
        loss = self.criterion(scores, target.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        """


    def apply(self, player_states, alpha):
        """
        Apply learning model to determine the best actions to take for the player states.
        """
        with torch.no_grad():
            actions = dict()
            vector_opts = [-1, -0.5, 0, 0.5, 1]
            action_opts = list(range(3))
            all_opts = list(product(vector_opts, vector_opts, action_opts))
            for pid in player_states.keys():
                best_action = ((random.uniform(-1,1), random.uniform(-1,1), random.randint(0,2)), -99999999)
                if random.uniform(0,1) > alpha[pid]:
                    for opt in all_opts:
                        reward = self.model[pid](self.encode(pid, player_states[pid], opt)).item()
                        best_action = max((opt, reward), best_action, key=lambda x:x[1])
                actions[pid] = best_action[0]
            return actions

    def batch_encode(self, player_id, states, actions, rew):
        l = list()
        s = list()
        e = list(self.encode(player_id, states, actions))
        l.append(e)
        s.append(rew)
        """
        for pid in actions.keys():
            e = list(self.encode(pid, states[pid], actions[pid]))
            l.append(e)
            s.append(rew[pid])
        """
        return torch.tensor(l), torch.tensor(s)

    def encode(self, pid, state, action):
        """
        Convert gobigger state data to pytorch tensor.
        """
        # Compute the window center (x, y)
        window_center = ((state['rectangle'][0] + state['rectangle'][2]) / 2.0, (state['rectangle'][1] + state['rectangle'][3]) / 2.0)

        # Determine which clones belong to the player that we are encoding the state for
        players_clones = [clone for clone in state['overlap']['clone'] if clone[8] == pid]
        # Sort the players_clones in decreasing radius. Here we assume that the biggest clones
        # are the most important
        max_player_clones = 10
        most_important_clones = sorted(players_clones, key=lambda clone: -clone[CLONE_RADIUS])[:max_player_clones]
        # Construct the information vector for the player
        player_info = [999] * 3 * max_player_clones

        i = 0
        for clone in most_important_clones:
            player_info[i] = clone[CLONE_X] - window_center[0] # x
            player_info[i+1] = clone[CLONE_Y] - window_center[1] # y
            player_info[i+2] = clone[CLONE_RADIUS] # radius
            i += 3

        # Determine enemy clones
        enemy_clones = [clone for clone in state['overlap']['clone'] if clone[8] != pid]

        return torch.tensor(
            list(action[:2]) # Encode the direction that we want to go
            + [(1 if x == action[2] else 0) for x in range(3)] # Encode the action we are taking as a one hot vector
            + player_info # Encode the position and radius of our blobs
            + self.get_candidates(players_clones, enemy_clones, window_center, 10) # Encode the position and radius of all enemy blobs
            + self.get_candidates(players_clones, state['overlap']['food'], window_center, 10)
            + self.get_candidates(players_clones, state['overlap']['thorns'], window_center, 10)
            + self.get_candidates(players_clones, state['overlap']['spore'], window_center, 10)
            + [int(state['can_split']), int(state['can_eject']), int(state['score'])]
        )

    def get_candidates(self, ref, tgt, window_center, candidate_length, ignore_size=False):
        """
        Returns candidate_length number of positions that are closest to ref
        """
        z = [999] * 3 * candidate_length
        i = 0
        cl = list()
        for r in ref:
            for t in tgt:
                dst = math.sqrt((r[0] - t[0]) ** 2 + (r[1] - t[1]) ** 2)
                cl.append((t, dst))

        # Sort by increasing distance
        cl.sort(key=lambda x: x[1])
        for x in cl:
            if i >= len(z):
                break
            z[i] = x[0][CLONE_X] - window_center[0] # x
            z[i+1] = x[0][CLONE_Y] - window_center[1] # y
            z[i+2] = x[0][CLONE_RADIUS] # radius
            i = i + 3
        return z
