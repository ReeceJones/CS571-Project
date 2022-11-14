

class LearningBase:
    def __init__(self):
        pass

    def step(self, prev_states, actions, player_states):
        """
        Apply outcomes of the previous actions to the learning model.
        """
        pass

    def apply(self, player_states):
        """
        Apply learning model to determine the best actions to take for the player states.
        """
        pass
