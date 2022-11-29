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

ACTION_MOVE = 0
ACTION_EJECT = 1
ACTION_SPLIT = 2

allowed_actions = [ACTION_MOVE, ACTION_SPLIT]

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
            self.dropout = nn.Dropout(0.25)

        def forward(self,X):
            Y = self.im_layer1(X)
            Y = self.nonlin1(Y)
            Y = self.dropout(Y)
            Y = self.im_layer2(Y)
            Y = self.nonlin2(Y)
            Y = self.dropout(Y)
            Y = self.im_layer3(Y)
            Y = self.nonlin3(Y)
            Y = self.dropout(Y)
            Y = self.im_layer4(Y)
            Y = self.nonlin4(Y)
            Y = self.dropout(Y)
            Y = self.out_layer(Y)
            return Y

    def __init__(self, num_players):
        self.num_players = num_players
        self.model = [self.SimpleNN() for i in range(num_players)]
        self.optimizers = [optim.Adam(m.parameters(), lr=0.05) for m in self.model]
        self.criterion = nn.MSELoss()

    def train(self):
        for m in self.model:
            m.train()

    def eval(self):
        for m in self.model:
            m.train()

    def save(self):
        for (i, model) in enumerate(self.model):
            torch.save(model.state_dict(), "model" + str(i) + ".pt")

    def load(self):
        for (i, model) in enumerate(self.model):
            model.load_state_dict(torch.load("model" + str(i) + ".pt"))

    def step(self, starting_states, actions, accum_rew):
        """
        Apply outcomes of the previous actions to the learning model.
        """
        for player_id in starting_states.keys():
            starting_state = starting_states[player_id]
            action = actions[player_id]
            actual_reward = torch.tensor(accum_rew[player_id])
            encoded = self.encode(player_id, starting_state, action)
            score = self.model[player_id](encoded.float()).squeeze()
            loss = self.criterion(score, actual_reward)
            self.optimizers[player_id].zero_grad()
            self.optimizers[player_id].step()

    def apply(self, player_states, alpha, force_random):
        """
        Apply learning model to determine the best actions to take for the player states.
        """
        with torch.no_grad():
            actions = dict()
            vector_opts = [-1, -0.5, 0, 0.5, 1]
            action_opts = allowed_actions
            all_opts = list(product(vector_opts, vector_opts, action_opts))
            for pid in player_states.keys():
                if force_random:
                    actions[pid] = random.choice(all_opts)
                else:
                    best_action = ((random.uniform(-1,1), random.uniform(-1,1), random.choice(action_opts)), float('-inf'))
                    if random.uniform(0,1) >= alpha:
                        for opt in all_opts:
                            reward = self.model[pid](self.encode(pid, player_states[pid], opt)).item()
                            best_action = max((opt, reward), best_action, key=lambda x:x[1])
                    actions[pid] = best_action[0]
            return actions

    def encode(self, pid, state, action):
        """
        Convert gobigger state,action data to pytorch tensor.
        """

        # Determine which clones belong to the player that we are encoding the state for
        players_clones = [clone for clone in state['overlap']['clone'] if clone[CLONE_PID] == pid]

        # Sort the players_clones in decreasing radius. Here we assume that the biggest clones
        # are the most important
        max_player_clones = 10
        most_important_clones = sorted(players_clones, key=lambda clone: -clone[CLONE_RADIUS])[:max_player_clones]
        # Construct the information vector for the player
        player_info = [-1.0] * 3 * max_player_clones

        i = 0
        for clone in most_important_clones:
            player_info[i] = clone[CLONE_X]
            player_info[i+1] = clone[CLONE_Y]
            player_info[i+2] = clone[CLONE_RADIUS] # radius
            i += 3

        # Determine enemy clones
        enemy_clones = [clone for clone in state['overlap']['clone'] if clone[CLONE_PID] != pid]

        return torch.tensor(
            list(action[:2]) # Encode the direction that we want to go
            + [(1 if x == action[2] else 0) for x in range(3)] # Encode the action we are taking as a one hot vector
            + player_info # Encode the position and radius of our blobs
            + self.get_candidates(players_clones, enemy_clones, 10) # Encode the position and radius of all enemy blobs
            + self.get_candidates(players_clones, state['overlap']['food'], 10)
            + self.get_candidates(players_clones, state['overlap']['thorns'], 10)
            + self.get_candidates(players_clones, state['overlap']['spore'], 10)
            + [int(state['can_split']), int(state['can_eject']), int(state['score'])]
        )

    def get_candidates(self, ref, tgt, candidate_length):
        """
        Returns candidate_length number of positions that are closest to ref
        """
        z = [-1.0] * 3 * candidate_length
        i = 0
        cl = list()
        for t in tgt:
            min_dst = None
            for r in ref:
                dst = math.sqrt((r[0] - t[0]) ** 2 + (r[1] - t[1]) ** 2)
                if min_dst is None or dst < min_dst:
                    min_dist = dst
            cl.append((t, min_dist))

        # Sort by increasing distance
        cl.sort(key=lambda x: x[1])
        for (t, dist) in cl:
            if i >= len(z):
                break
            z[i] = t[CLONE_X] # x
            z[i+1] = t[CLONE_Y] # y
            z[i+2] = t[CLONE_RADIUS] # radius
            i = i + 3
        return z
