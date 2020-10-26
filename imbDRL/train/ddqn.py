from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import tensorflow as tf
from imbDRL.data import collect_data
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common
from tqdm import tqdm


class TrainDDQN(ABC):
    """Wrapper for DDQN training, validation, saving etc."""

    def __init__(self, episodes: int, warmup_episodes: int, lr: float, gamma: float, min_epsilon: float, decay_episodes: int,
                 model_dir: str = None, log_dir: str = None, batch_size: int = 64, memory_length: int = 100_000,
                 collect_steps_per_episode: int = 1, log_every: int = 200, val_every: int = 1_000,
                 target_model_update: int = 1, target_update_tau: float = 1.0):
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

        self.lr = lr  # Learning Rate
        self.gamma = gamma  # Discount factor
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`
        self.target_model_update = target_model_update  # Period for soft updates
        self.target_update_tau = target_update_tau

        NOW = datetime.now().strftime('%Y%m%d_%H%M%S')
        if model_dir is None:
            self.model_dir = "./models/" + NOW
        else:
            self.model_dir = model_dir

        if log_dir is None:
            self.log_dir = "./logs/" + NOW
        else:
            self.log_dir = log_dir

        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.global_episode = tf.Variable(0, name="global_episode", dtype=np.int64, trainable=False)  # Global train episode counter

        # Custom epsilon decay: https://github.com/tensorflow/agents/issues/339
        self.epsilon_decay = tf.compat.v1.train.polynomial_decay(
            1.0, self.global_episode, self.decay_episodes, end_learning_rate=self.min_epsilon)
        self.optimizer = Adam(learning_rate=self.lr)
        self.compiled = False

    def compile_model(self, train_env, conv_layers, dense_layers, dropout_layers, loss_fn=common.element_wise_squared_loss):
        """Initializes the Q-network, agent, collect policy and replay buffer."""
        for layer in (conv_layers, dense_layers, dropout_layers):
            if not isinstance(layer, (tuple, list, type(None))):
                raise TypeError(f"Layer {layer=} must be tuple or None, not {type(layer)}.")

        self.train_env = train_env
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.dropout_layers = dropout_layers
        self.loss_fn = loss_fn

        self.q_net = QNetwork(self.train_env.observation_spec(),
                              self.train_env.action_spec(),
                              conv_layer_params=self.conv_layers,
                              fc_layer_params=self.dense_layers,
                              dropout_layer_params=self.dropout_layers)

        self.agent = DdqnAgent(self.train_env.time_step_spec(),
                               self.train_env.action_spec(),
                               q_network=self.q_net,
                               optimizer=self.optimizer,
                               td_errors_loss_fn=loss_fn,
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
        self.compiled = True

    def train(self, *args):
        """Starts the training of the model. Includes warmup period, metrics collection and model saving."""
        assert self.compiled, "Model must be compiled with model.compile_model() before training."

        # Warmup period, fill memory with random actions
        collect_data(self.train_env, self.random_policy, self.replay_buffer, self.warmup_episodes, logging=True)
        self.dataset = self.replay_buffer.as_dataset(sample_batch_size=self.batch_size, num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)
        self.agent.train = common.function(self.agent.train)  # Optimalization

        self.collect_metrics(*args)  # Initial collection for step 0
        for _ in tqdm(range(self.episodes)):
            # Collect a few steps using collect_policy and save to `replay_buffer`
            # TODO: determine which policy to use: collect_policy or policy
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_model(fp: str):
        """Abstract method for loading the model/network/policy of disk."""
        raise NotImplementedError

    @abstractmethod
    def collect_metrics(self):
        """*args given in train() will be passed to this function."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """Evaluation function to run after training with seperate train-dataset."""
        raise NotImplementedError
