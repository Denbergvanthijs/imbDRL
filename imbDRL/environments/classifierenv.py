import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts


class ClassifierEnv(PyEnvironment):
    """
    Custom `PyEnvironment` environment for imbalanced classification.
    Based on https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    """

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, imb_ratio: float):
        """Initialization of environment with X_train and y_train.

        :param X_train: Features shaped: [samples, ..., ]
        :type  X_train: np.ndarray
        :param y_train: Labels shaped: [samples]
        :type  y_train: np.ndarray
        :param imb_ratio: Imbalance ratio of the data
        :type  imb_ratio: float

        :returns: None
        :rtype: NoneType

        """
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name="action")
        self._observation_spec = ArraySpec(shape=X_train.shape[1:], dtype=X_train.dtype, name="observation")
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train
        self.imb_ratio = imb_ratio  # Imbalance ratio: 0 < imb_ratio < 1
        self.id = np.arange(self.X_train.shape[0])  # List of IDs to connect X and y data

        self.episode_step = 0  # Episode step, resets every episode
        self._state = self.X_train[self.id[self.episode_step]]

    def action_spec(self):
        """
        Definition of the discrete actionspace.
        1 for the positive/minority class, 0 for the negative/majority class.
        """
        return self._action_spec

    def observation_spec(self):
        """Definition of the continous statespace e.g. the observations in typical RL environments."""
        return self._observation_spec

    def _reset(self):
        """Shuffles data and returns the first state of the shuffled data to begin training on new episode."""
        np.random.shuffle(self.id)  # Shuffle the X and y data
        self.episode_step = 0  # Reset episode step counter at the end of every episode
        self._state = self.X_train[self.id[self.episode_step]]
        self._episode_ended = False  # Reset terminal condition

        return ts.restart(self._state)

    def _step(self, action: int):
        """
        Take one step in the environment.
        If the action is correct, the environment will either return 1 or `imb_ratio` depending on the current class.
        If the action is incorrect, the environment will either return -1 or -`imb_ratio` depending on the current class.
        """
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode
            return self.reset()

        env_action = self.y_train[self.id[self.episode_step]]  # The label of the current state
        self.episode_step += 1

        if action == env_action:  # Correct action
            if env_action:  # Minority
                reward = 1  # True Positive
            else:  # Majority
                reward = self.imb_ratio  # True Negative

        else:  # Incorrect action
            if env_action:  # Minority
                reward = -1  # False Negative
                self._episode_ended = True  # Stop episode when minority class is misclassified
            else:  # Majority
                reward = -self.imb_ratio  # False Positive

        if self.episode_step == self.X_train.shape[0] - 1:  # If last step in data
            self._episode_ended = True

        self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)
