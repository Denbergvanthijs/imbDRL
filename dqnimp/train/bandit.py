import pickle
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from dqnimp.metrics import metrics_by_network
from tensorflow.keras.optimizers import Adam
from tf_agents.bandits.agents.examples.v2.trainer import get_training_loop_fn
from tf_agents.bandits.agents.neural_epsilon_greedy_agent import \
    NeuralEpsilonGreedyAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tqdm import tqdm


class TrainBandit(ABC):
    """Wrapper for DDQN training, validation, saving etc."""

    def __init__(self, episodes: int, lr: float, min_epsilon: float, decay_episodes: int, model_dir: str, log_dir: str,
                 batch_size: int = 64, steps_per_loop: int = 64, log_every: int = 10, val_every: int = 20):
        """
        Wrapper to make training easier.
        Code is partly based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
        """
        self.episodes = episodes  # Total episodes
        self.batch_size = batch_size  # Batch size of Replay Memory
        self.steps_per_loop = steps_per_loop

        self.log_every = log_every  # Report loss every `LOG_EVERY` episodes
        self.val_every = val_every  # Validate the policy every `VAL_EVERY` episodes

        self.lr = lr  # Learning Rate
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.global_episode = tf.Variable(0, name="global_episode", dtype=np.int64, trainable=False)  # Global train episode counter

        # Custom epsilon decay: https://github.com/tensorflow/agents/issues/339
        self.epsilon_decay = tf.compat.v1.train.polynomial_decay(
            1.0, self.global_episode, self.decay_episodes, end_learning_rate=self.min_epsilon)
        self.optimizer = Adam(learning_rate=self.lr)

    def compile_model(self, train_env, conv_layers, dense_layers, dropout_layers):
        """Initializes the Q-network, agent, collect policy and replay buffer."""
        self.train_env = train_env
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.dropout_layers = dropout_layers

        self.q_net = QNetwork(self.train_env.observation_spec(),
                              self.train_env.action_spec(),
                              conv_layer_params=self.conv_layers,
                              fc_layer_params=self.dense_layers,
                              dropout_layer_params=self.dropout_layers)

        self.agent = NeuralEpsilonGreedyAgent(train_env.time_step_spec(),
                                              train_env.action_spec(),
                                              reward_network=self.q_net,
                                              optimizer=Adam(learning_rate=self.lr),
                                              epsilon=self.epsilon_decay,
                                              train_step_counter=self.global_episode,
                                              error_loss_fn=tf.compat.v1.losses.sigmoid_cross_entropy)

        self.replay_buffer = TFUniformReplayBuffer(self.agent.policy.trajectory_spec,
                                                   self.train_env.batch_size,
                                                   self.steps_per_loop)

        self.driver = DynamicStepDriver(env=self.train_env,
                                        policy=self.agent.collect_policy,
                                        num_steps=self.steps_per_loop * self.train_env.batch_size,
                                        observers=[self.replay_buffer.add_batch])

        self.training_loop = get_training_loop_fn(self.driver, self.replay_buffer, self.agent, self.steps_per_loop)

    def train(self, *args):
        """Starts the training of the model. Includes warmup period, metrics collection and model saving."""

        self.collect_metrics(*args)  # Initial collection for step 0
        for _ in tqdm(range(self.episodes)):
            loss_info = self.training_loop()

            if not self.global_episode % self.log_every:
                with self.writer.as_default():
                    tf.summary.scalar("train_loss", loss_info.loss, step=self.global_episode)

            if not self.global_episode % self.val_every:
                self.collect_metrics(*args)

        self.save_model()

    @abstractmethod
    def save_model(self):
        """Abstract method for saving the model/network/policy to disk."""
        pass

    @abstractmethod
    def load_model(fp: str):
        """Abstract method for loading the model/network/policy of disk."""
        pass

    @abstractmethod
    def collect_metrics(self):
        """*args given in train() will be passed to this function."""
        pass

    @abstractmethod
    def evaluate(self):
        """Evaluation function to run after training with seperate train-dataset."""
        pass


class TrainCustomBandit(TrainBandit):
    """Class for the bandit training environment."""

    def collect_metrics(self, X_val, y_val):
        """Collects metrics using the trained Q-network."""
        stats = metrics_by_network(self.agent._reward_network, X_val, y_val)

        with self.writer.as_default():
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test):
        """Final evaluation of trained Q-network with X_test and y_test."""
        return metrics_by_network(self.agent._reward_network, X_test, y_test)

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
