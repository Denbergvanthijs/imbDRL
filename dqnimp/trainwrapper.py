import pickle
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent, DqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common
from tqdm import tqdm

from dqnimp.metrics import metrics_by_network
from dqnimp.data import collect_data


class TrainWrapper(ABC):
    """Wrapper for (D)DQN training, validation, saving etc."""

    def __init__(self, episodes: int, warmup_episodes: int, lr: float, gamma: float, min_epsilon: float, decay_episodes: int,
                 model_dir: str, log_dir: str, batch_size: int = 64, memory_length: int = 100_000, collect_steps_per_episode: int = 1,
                 log_every: int = 200, val_every: int = 1_000, val_episodes: int = 10, target_model_update: int = 1,
                 target_update_tau: float = 1.0, ddqn: bool = True):
        """
        Wrapper to make training easier.
        Code is partly based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
        """
        self.episodes = episodes  # Total episodes
        self.warmup_episodes = warmup_episodes  # Amount of warmup steps before training
        self.batch_size = batch_size  # Batch size of Replay Memory
        self.memory_length = memory_length  # Max Replay Memory length
        self.collect_steps_per_episode = collect_steps_per_episode  # Amount of steps to collect data each episode

        self.log_every = log_every  # Print step and loss every `LOG_EVERY` episodes
        self.val_every = val_every  # Validate the policy every `VAL_EVERY` episodes
        self.val_episodes = val_episodes  # Number of episodes to use to calculate metrics during training

        self.lr = lr  # Learning Rate
        self.gamma = gamma  # Discount factor
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`
        self.target_model_update = target_model_update  # Period for soft updates
        self.target_update_tau = target_update_tau
        self.ddqn = ddqn  # Use Double DQN?

        self.model_dir = model_dir
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.global_episode = tf.Variable(0, name="global_episode", dtype=np.int64, trainable=False)  # Global train episode counter

        # Custom epsilon decay: https://github.com/tensorflow/agents/issues/339
        self.epsilon_decay = tf.compat.v1.train.polynomial_decay(
            1.0, self.global_episode, self.decay_episodes, end_learning_rate=self.min_epsilon)
        self.optimizer = Adam(learning_rate=self.lr)

    def compile_model(self, train_env, val_env, conv_layers, dense_layers, dropout_layers):
        """Initializes the Q-network, agent, collect policy and replay buffer."""
        self.train_env = train_env
        self.val_env = val_env
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.dropout_layers = dropout_layers

        if self.ddqn:
            agent = DdqnAgent
        else:
            agent = DqnAgent

        self.q_net = QNetwork(self.train_env.observation_spec(),
                              self.train_env.action_spec(),
                              conv_layer_params=self.conv_layers,
                              fc_layer_params=self.dense_layers,
                              dropout_layer_params=self.dropout_layers)

        self.agent = agent(self.train_env.time_step_spec(),
                           self.train_env.action_spec(),
                           q_network=self.q_net,
                           optimizer=self.optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=self.global_episode,
                           target_update_period=self.target_model_update,
                           target_update_tau=self.target_update_tau,
                           gamma=self.gamma,
                           epsilon_greedy=self.epsilon_decay)
        self.agent.initialize()

        self.random_policy = RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        self.replay_buffer = TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec,
                                                   batch_size=self.train_env.batch_size,
                                                   max_length=self.memory_length)

    def train(self, *args):
        """Starts the training of the model. Includes warmup period, metrics collection and model saving."""
        # Warmup period, fill memory with random actions
        collect_data(self.train_env, self.random_policy, self.replay_buffer, self.warmup_episodes, logging=True)
        self.dataset = self.replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=self.batch_size, num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)
        self.agent.train = common.function(self.agent.train)  # Optimalization

        self.collect_metrics(*args)  # Initial collection for step 0
        for _ in tqdm(range(self.episodes)):
            # Collect a few steps using collect_policy and save to `replay_buffer`
            collect_data(self.train_env, self.agent.collect_policy, self.replay_buffer, self.collect_steps_per_episode)

            # Sample a batch of data from `replay_buffer` and update the agent's network
            experiences, _ = next(self.iterator)
            train_loss = self.agent.train(experiences).loss

            if not self.global_episode % self.log_every:
                with self.writer.as_default():
                    tf.summary.scalar("train_loss", train_loss, step=self.global_episode)

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


class TrainCustom(TrainWrapper):
    """Class for the custom training environment."""

    def collect_metrics(self, X_val, y_val):
        """Collects metrics using the trained Q-network."""
        stats = metrics_by_network(self.agent._target_q_network, X_val, y_val)

        with self.writer.as_default():
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test):
        """Final evaluation of trained Q-network with X_test and y_test."""
        return metrics_by_network(self.agent._target_q_network, X_test, y_test)

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


class TrainCartPole(TrainWrapper):
    """Class for the CartPole environment."""

    def collect_metrics(self):
        """Calculates the average return for `val_episodes` using the trained policy."""
        total_return = 0.0
        for _ in range(self.val_episodes):
            time_step = self.val_env.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.val_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return // self.val_episodes
        with self.writer.as_default():
            tf.summary.scalar("avg_return", avg_return.numpy()[0], step=self.global_episode)

    def evaluate(self):
        """Final evaluation of policy."""
        return self.collect_metrics()

    def save_model(self):
        """Saves the policy to `model_dir`."""
        saver = PolicySaver(self.agent.policy)
        saver.save(self.model_dir)

    @staticmethod
    def load_model(fp: str):
        """Loads a saved policy from given filepath."""
        return tf.saved_model.load(fp)
