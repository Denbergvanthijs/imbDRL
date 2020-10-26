import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts


class ClassifyEnv(PyEnvironment):
    """
    Custom `PyEnvironment` environment for imbalanced classification.
    Based on https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    """

    def __init__(self, X_train, y_train, imb_rate):
        """Initialization of environment with X_train and y_train."""
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name="action")
        self._observation_spec = ArraySpec(shape=X_train.shape[1:], dtype=np.float32, name="observation")
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train
        self.imb_rate = imb_rate  # Imbalance rate: 0 < x < 1

        self.X_len = self.X_train.shape[0]
        self.id = np.arange(self.X_len)  # List of IDs to connect X and y data

        self.episode_step = 0  # Episode step, resets every episode
        self._state = self.X_train[self.id[self.episode_step]]

    def action_spec(self):
        """Definition of the actions."""
        return self._action_spec

    def observation_spec(self):
        """Definition of the observations."""
        return self._observation_spec

    def _reset(self):
        """Shuffles data and returns the first state to begin training on new episode."""
        np.random.shuffle(self.id)
        self.episode_step = 0  # Reset episode step counter at the end of every episode
        self._state = self.X_train[self.id[self.episode_step]]
        self._episode_ended = False

        return ts.restart(self._state)

    def _step(self, action):
        """Take one step in the environment.
        If the action is correct, the environment will either return 1 or `imb_rate` depending on the current class.
        If the action is incorrect, the environment will either return -1 or -`imb_rate` depending on the current class.
        """
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode
            return self.reset()

        env_action = self.y_train[self.id[self.episode_step]]
        self.episode_step += 1

        if action == env_action:  # Correct action
            if env_action:  # Minority
                reward = 1  # True Positive
            else:  # Majority
                reward = self.imb_rate  # True Negative

        else:  # Incorrect action
            if env_action:  # Minority
                reward = -1  # False Negative
                self._episode_ended = True  # Stop episode when minority class is misclassified
            else:  # Majority
                reward = -self.imb_rate  # False Positive

        if self.episode_step == self.X_len - 1:  # If last step in data
            self._episode_ended = True

        self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
