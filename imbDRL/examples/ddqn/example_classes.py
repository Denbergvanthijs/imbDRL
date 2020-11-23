import pickle

import numpy as np
import tensorflow as tf
from imbDRL.metrics import (classification_metrics, network_predictions,
                            plot_pr_curve, plot_roc_curve)
from imbDRL.train.ddqn import TrainDDQN
from tf_agents.policies.policy_saver import PolicySaver


class TrainCustomDDQN(TrainDDQN):
    """Class for the custom training environment."""

    def collect_metrics(self, X_val, y_val):
        """Collects metrics using the trained Q-network."""
        y_pred = network_predictions(self.agent._target_q_network, X_val)
        stats = classification_metrics(y_val, y_pred)
        avgQ = np.mean(np.max(self.agent._target_q_network(X_val)[0].numpy(), axis=1))  # Max action for each x in X

        if not hasattr(self, 'best_f1'):  # If no best model yet
            self.best_f1 = 0.0

        if stats.get("F1") >= self.best_f1:  # Overwrite best model
            self.save_model()  # Saving directly to avoid shallow copy without trained weights
            self.best_f1 = stats.get("F1")

        with self.writer.as_default():
            tf.summary.scalar("AverageQ", avgQ, step=self.global_episode)

            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """
        Final evaluation of trained Q-network with X_test and y_test.
        Optional PR and ROC curve comparison to X_train, y_train to ensure no overfitting is taking place.
        """
        best_model = self.load_model(self.model_dir)

        if (X_train is not None) and (y_train is not None):
            plot_pr_curve(best_model, X_test, y_test, X_train, y_train)
            plot_roc_curve(best_model, X_test, y_test, X_train, y_train)

        y_pred = network_predictions(best_model, X_test)
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


class TrainCartPole(TrainDDQN):
    """Class for the CartPole environment."""

    def collect_metrics(self, val_env, val_episodes: int):
        """Calculates the average return for `val_episodes` using the trained policy."""
        total_return = 0.0
        for _ in range(val_episodes):
            time_step = val_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = val_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return // val_episodes

        with self.writer.as_default():
            tf.summary.scalar("avg_return", avg_return.numpy()[0], step=self.global_episode)

    def evaluate(self, test_env, test_episodes: int):
        """Final evaluation of policy."""
        return self.collect_metrics(test_env, test_episodes)

    def save_model(self):
        """Saves the policy to `model_dir`."""
        saver = PolicySaver(self.agent.policy)
        saver.save(self.model_dir)

    @staticmethod
    def load_model(fp: str):
        """Loads a saved policy from given filepath."""
        return tf.saved_model.load(fp)
