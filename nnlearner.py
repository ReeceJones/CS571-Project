import torch
import torch.optim as optim
import torch.nn as nn
import math
import random

from itertools import product

from learning import LearningBase

class NNLearner(LearningBase):
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.im_layer1 = nn.Linear(68, 68)
            self.im_layer2 = nn.Linear(68, 34)
            self.out_layer = nn.Linear(34, 1)

        def forward(self,X):
            Y = self.im_layer1(X)
            Y = self.im_layer2(Y)
            Y = self.out_layer(Y)
            return Y

    def __init__(self):
        self.model = self.SimpleNN()
        self.optimizer = optim.Adam(list(self.model.parameters()))
        self.criterion = nn.MSELoss()

    def step(self, prev_states, actions, new_states):
        """
        Apply outcomes of the previous actions to the learning model.
        """
        actual_scores = dict()
        for player_id in new_states.keys():
            action = actions[player_id]
            state = new_states[player_id]
            actual_scores[player_id] = state['score']

        encoded, target = self.batch_encode(prev_states, actions, actual_scores)

        # update gradients
        scores = self.model(encoded.float())
        loss = self.criterion(scores, target.float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def apply(self, player_states):
        """
        Apply learning model to determine the best actions to take for the player states.
        """
        with torch.no_grad():
            alpha = 0.5
            actions = dict()
            vector_opts = [-1, -0.5, 0, 0.5, 1]
            action_opts = list(range(3))
            all_opts = list(product(vector_opts, vector_opts, action_opts))
            for pid in player_states.keys():
                best_action = ((random.uniform(-1,1), random.uniform(-1,1), random.randint(0,2)), -99999999)
                if random.uniform(0,1) > alpha:
                    for opt in all_opts:
                        score = self.model(self.encode(player_states[pid], opt)).item()
                        best_action = max((opt, score), best_action, key=lambda x:x[1])
                actions[pid] = best_action[0]
            return actions

    def batch_encode(self, states, actions, target_dict):
        l = list()
        s = list()
        for pid in actions.keys():
            e = list(self.encode(states[pid], actions[pid]))
            l.append(e)
            s.append(target_dict[pid])
        return torch.tensor(l), torch.tensor(s)


    def encode(self, state, action):
        """
        Convert gobigger state data to pytorch tensor.
        """
        return torch.tensor(
            list(action[:2])
            + [(1 if x == action[2] else 0) for x in range(3)]
            + self.get_candidates(state['overlap']['clone'], state['overlap']['clone'], 5)
            + self.get_candidates(state['overlap']['clone'], state['overlap']['food'], 5)
            + self.get_candidates(state['overlap']['clone'], state['overlap']['thorns'], 5)
            + self.get_candidates(state['overlap']['clone'], state['overlap']['spore'], 5)
            + [int(state['can_split']), int(state['can_eject']), int(state['score'])]
        )

    def get_candidates(self, ref, tgt, candidate_length, ignore_size=False):
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

        cl.sort(key=lambda x: x[1])
        for x in cl:
            if i >= len(z):
                break
            z[i] = x[0][0] # x
            z[i+1] = x[0][1] # y
            z[i+2] = x[0][2] # radius
            i = i + 3
        return z
