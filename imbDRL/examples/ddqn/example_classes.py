import pickle

import numpy as np
import tensorflow as tf
from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_pr_curve, plot_roc_curve)
from imbDRL.train.ddqn import TrainDDQN
from tf_agents.trajectories import time_step


class TrainCustomDDQN(TrainDDQN):
    """Class for the custom training environment."""

    def collect_metrics(self, X_val, y_val, save_best: str = None):
        """Collects metrics using the trained Q-network."""
        y_pred = network_predictions(self.agent._target_q_network, X_val)
        stats = classification_metrics(y_val, y_pred)

        avgQ = np.mean(np.max(self.agent._target_q_network(X_val, step_type=tf.constant(
            [time_step.StepType.FIRST] * X_val.shape[0]), training=False)[0].numpy(), axis=1))  # Max action for each x in X

        if save_best is not None:
            if not hasattr(self, 'best_score'):  # If no best model yet
                self.best_score = 0.0

            if stats.get(save_best) >= self.best_score:  # Overwrite best model
                self.save_model()  # Saving directly to avoid shallow copy without trained weights
                self.best_score = stats.get(save_best)

        with self.writer.as_default():
            tf.summary.scalar("AverageQ", avgQ, step=self.global_episode)  # Average Q-value for this epoch
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """
        Final evaluation of trained Q-network with X_test and y_test.
        Optional PR and ROC curve comparison to X_train, y_train to ensure no overfitting is taking place.
        """
        if hasattr(self, 'best_score'):
            print(f"\033[92mBest score: {self.best_score:6f}!\033[0m")
            model = self.load_model(self.model_dir)  # Load best saved model
        else:
            model = self.agent._target_q_network  # Load latest target model

        if (X_train is not None) and (y_train is not None):
            plot_pr_curve(model, X_test, y_test, X_train, y_train)
            plot_roc_curve(model, X_test, y_test, X_train, y_train)

        y_pred = network_predictions(model, X_test)
        return classification_metrics(y_test, y_pred)

    def save_model(self):
        """Saves Q-network as pickle to `model_dir`."""
        with open(self.model_dir + ".pkl", "wb") as f:  # Save Q-network as pickle
            pickle.dump(self.agent._target_q_network, f)

    @staticmethod
    def load_model(fp: str):
        """Static method to load Q-network pickle from given filepath."""
        with open(fp + ".pkl", "rb") as f:  # Load the Q-network
            network = pickle.load(f)
        return network
