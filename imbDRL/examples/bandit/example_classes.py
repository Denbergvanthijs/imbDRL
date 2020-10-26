import pickle

import tensorflow as tf
from imbDRL.metrics import classification_metrics, network_predictions
from imbDRL.train.bandit import TrainBandit


class TrainCustomBandit(TrainBandit):
    """Class for the bandit training environment."""

    def collect_metrics(self, X_val, y_val):
        """Collects metrics using the trained Q-network."""
        y_pred = network_predictions(self.agent._reward_network, X_val)
        stats = classification_metrics(y_val, y_pred)

        with self.writer.as_default():
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test):
        """Final evaluation of trained Q-network with X_test and y_test."""
        y_pred = network_predictions(self.agent._reward_network, X_test)
        return classification_metrics(y_test, y_pred)

    def save_model(self):
        """Saves Q-network as pickle to `model_dir`."""
        with open(self.model_dir + ".pkl", "wb") as f:  # Save Q-network as pickle
            pickle.dump(self.agent._reward_network, f)

    @staticmethod
    def load_model(fp: str):
        """Static method to load Q-network pickle from given filepath."""
        with open(fp + ".pkl", "rb") as f:  # Load the Q-network
            network = pickle.load(f)
        return network
