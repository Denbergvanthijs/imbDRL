from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import data
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common
from tqdm import tqdm


class TrainDDQN(ABC):
    """Wrapper for DDQN training, validation, saving etc."""

    def __init__(self, episodes: int, warmup_episodes: int, lr: float, gamma: float, min_epsilon: float, decay_episodes: int,
                 model_dir: str = None, log_dir: str = None, batch_size: int = 64, memory_length: int = None,
                 collect_steps_per_episode: int = 1, val_every: int = None, target_model_update: int = 1,
                 target_update_tau: float = 1.0, progressbar: bool = True) -> None:
        """
        Wrapper to make training easier.
        Code is partly based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

        :param episodes: Number of training episodes
        :type  episodes: int
        :param warmup_episodes: Number of episodes to fill Replay Buffer with random state-action pairs before training starts
        :type  warmup_episodes: int
        :param lr: Learning Rate for the Adam Optimizer
        :type  lr: float
        :param gamma: Discount factor for the Q-values
        :type  gamma: float
        :param min_epsilon: Lowest and final value for epsilon
        :type  min_epsilon: float
        :param decay_episodes: Amount of episodes to decay from 1 to `min_epsilon`
        :type  decay_episodes: int
        :param model_dir: Location to save the trained models
        :type  model_dir: str
        :param log_dir: Location to save the logs, usefull for TensorBoard
        :type  log_dir: str
        :param batch_size: Number of samples in minibatch to train on each step
        :type  batch_size: int
        :param memory_length: Maximum size of the Replay Buffer
        :type  memory_length: int
        :param collect_steps_per_episode: Amount of data to collect for Replay Buffer each episiode
        :type  collect_steps_per_episode: int
        :param val_every: Validate the model every X episodes using the `collect_metrics()` function
        :type  val_every: int
        :param target_model_update: Update the target Q-network every X episodes
        :type  target_model_update: int
        :param target_update_tau: Parameter for softening the `target_model_update`
        :type  target_update_tau: float
        :param progressbar: Enable or disable the progressbar for collecting data and training
        :type  progressbar: bool

        :return: None
        :rtype: NoneType
        """
        self.episodes = episodes  # Total episodes
        self.warmup_episodes = warmup_episodes  # Amount of warmup steps before training
        self.batch_size = batch_size  # Batch size of Replay Memory
        self.collect_steps_per_episode = collect_steps_per_episode  # Amount of steps to collect data each episode

        if memory_length is not None:
            self.memory_length = memory_length  # Max Replay Memory length
        else:
            self.memory_length = warmup_episodes

        if val_every is not None:
            self.val_every = val_every  # Validate the policy every `VAL_EVERY` episodes
        else:
            self.val_every = episodes // min(50, self.episodes)  # Can't validate the model 50 times if self.episodes < 50

        self.lr = lr  # Learning Rate
        self.gamma = gamma  # Discount factor
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`
        self.target_model_update = target_model_update  # Period for soft updates
        self.target_update_tau = target_update_tau
        self.progressbar = progressbar  # Enable or disable the progressbar for collecting data and training

        NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    def compile_model(self, train_env, conv_layers: tuple, dense_layers: tuple, dropout_layers: tuple,
                      loss_fn=common.element_wise_squared_loss) -> None:
        """Initializes the Q-network, agent, collect policy and replay buffer.

        :param train_env: The training environment used by the agent
        :type  train_env: tf_agents.environments.tf_py_environment.TFPyEnvironment
        :param conv_layers: Tuple of architecture of the convolutional layers.
            From tf_agents.networks.q_network:
                (...) where each item is a length-three tuple indicating (filters, kernel_size, stride).
        :type  conv_layers: tuple
        :param dense_layers: Tuple of dense layer architecture. Each number in the tuple is the number of neurons of that layer.
        :type  dense_layers: tuple
        :param dropout_layers: Tuple of percentage of dropout per each dense layer
        :type  dropout_layers: tuple
        :param loss_fn: Callable loss function
        :type  loss_fn: tf.compat.v1.losses

        :return: None
        :rtype: NoneType
        """
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

    def train(self, *args) -> None:
        """Starts the training of the model. Includes warmup period, metrics collection and model saving.

        :param *args: All arguments will be passed to `collect_metrics()`.
            This can be usefull to pass callables, testing environments or validation data.
        :type  *args: Any

        :return: None
        :rtype: NoneType, last step is saving the model as a side-effect
        """
        assert self.compiled, "Model must be compiled with model.compile_model() before training."

        # Warmup period, fill memory with random actions
        if self.progressbar:
            print(f"\033[92mCollecting data for {self.warmup_episodes} episodes... This might take a few minutes...\033[0m")
        self.collect_data(self.random_policy, self.warmup_episodes)

        self.dataset = self.replay_buffer.as_dataset(sample_batch_size=self.batch_size, num_steps=2,
                                                     num_parallel_calls=data.experimental.AUTOTUNE).prefetch(data.experimental.AUTOTUNE)
        self.iterator = iter(self.dataset)
        self.agent.train = common.function(self.agent.train)  # Optimalization

        self.collect_metrics(*args)  # Initial collection for step 0
        for _ in tqdm(range(self.episodes), disable=(not self.progressbar)):
            # Collect a few steps using collect_policy and save to `replay_buffer`
            # TODO: determine which policy to use: collect_policy or policy
            # TODO: determine if collected data is saved in the buffer and then passed to self.dataset
            self.collect_data(self.agent.collect_policy, self.collect_steps_per_episode)

            # Sample a batch of data from `replay_buffer` and update the agent's network
            experiences, _ = next(self.iterator)
            train_loss = self.agent.train(experiences).loss

            if not self.global_episode % self.val_every:
                with self.writer.as_default():
                    tf.summary.scalar("train_loss", train_loss, step=self.global_episode)

                self.collect_metrics(*args)

    def collect_data(self, policy, steps: int) -> None:
        """Collect data for a number of steps. Mainly used for warmup period."""
        _ = DynamicStepDriver(self.train_env,
                              policy,
                              observers=[self.replay_buffer.add_batch],
                              num_steps=steps).run()

    @abstractmethod
    def collect_metrics(self):
        """*args given in train() will be passed to this function."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """Evaluation function to run after training with seperate test-dataset."""
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        """Abstract method for saving the model/network/policy to disk."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_model(fp: str):
        """Abstract method for loading the model/network/policy of disk."""
        raise NotImplementedError
