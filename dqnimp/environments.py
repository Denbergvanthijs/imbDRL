import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts


class ClassifyEnv(PyEnvironment):
    def __init__(self, X_train, y_train, imb_rate):
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = ArraySpec(shape=X_train.shape[1:], dtype=np.float32, name='observation')
        self._episode_ended = False

        self.X_train = X_train
        self.y_train = y_train
        self.imb_rate = imb_rate  # Imbalance rate: 0 < x < 1

        self.X_len = self.X_train.shape[0]
        self.id = np.arange(self.X_len)  # List of IDs to connect X and y data

        self.episode_step = 0  # Episode step, resets every episode
        self._state = self.X_train[self.id[self.episode_step]]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        np.random.shuffle(self.id)
        self.episode_step = 0  # Reset episode step counter at the end of every episode
        self._state = self.X_train[self.id[self.episode_step]]
        self._episode_ended = False

        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode
            return self.reset()

        curr_y_true = self.y_train[self.id[self.episode_step]]
        self.episode_step += 1

        if action == curr_y_true:  # Correct action
            if curr_y_true:  # Minority
                reward = 1
            else:  # Majority
                reward = self.imb_rate

        else:  # Incorrect action
            if curr_y_true:  # Minority
                reward = -1
                self._episode_ended = True  # Stop episode when minority class is misclassified
            else:  # Majority
                reward = -self.imb_rate

        if self.episode_step == self.X_len - 1:
            self._episode_ended = True

        self._state = self.X_train[self.id[self.episode_step]]  # Update state with new datapoint

        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)


if __name__ == "__main__":
    from tf_agents.environments.utils import validate_py_environment

    from dqnimp.data import load_data

    imb_rate = 0.00173  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected

    # Remove classes ∉ {min_class, maj_class}, imbalance the dataset
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

    environment = ClassifyEnv(X_train, y_train, imb_rate)
    validate_py_environment(environment, episodes=5)
