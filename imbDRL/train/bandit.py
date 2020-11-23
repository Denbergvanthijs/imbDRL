from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import tensorflow as tf
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

    def __init__(self, episodes: int, lr: float, min_epsilon: float, decay_episodes: int, model_dir: str = None, log_dir: str = None,
                 batch_size: int = 64, steps_per_loop: int = 64, val_every: int = None) -> None:
        """
        Wrapper to make training easier.
        Code is partly based of https://www.tensorflow.org/agents/tutorials/bandits_tutorial

        :param episodes: Number of training episodes
        :type  episodes: int
        :param lr: Learning Rate for the Adam Optimizer
        :type  lr: float
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
        :param steps_per_loop: Number of steps to take each episode
        :type  steps_per_loop: int
        :param val_every: Validate the model every X episodes using the `collect_metrics()` function
        :type  val_every: int

        :return: None
        :rtype: NoneType
        """
        self.episodes = episodes  # Total episodes
        self.batch_size = batch_size  # Batch size of Replay Memory
        self.steps_per_loop = steps_per_loop

        if val_every is not None:
            self.val_every = val_every  # Validate the policy every `VAL_EVERY` episodes
        else:
            self.val_every = episodes // min(50, self.episodes)  # Can't validate the model 50 times if self.episodes < 50

        self.lr = lr  # Learning Rate
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`

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
                      loss_fn=tf.compat.v1.losses.mean_squared_error) -> None:
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

        self.agent = NeuralEpsilonGreedyAgent(train_env.time_step_spec(),
                                              train_env.action_spec(),
                                              reward_network=self.q_net,
                                              optimizer=Adam(learning_rate=self.lr),
                                              epsilon=self.epsilon_decay,
                                              train_step_counter=self.global_episode,
                                              error_loss_fn=loss_fn)

        self.replay_buffer = TFUniformReplayBuffer(self.agent.policy.trajectory_spec,
                                                   self.train_env.batch_size,
                                                   self.steps_per_loop)

        self.driver = DynamicStepDriver(env=self.train_env,
                                        policy=self.agent.collect_policy,
                                        num_steps=self.steps_per_loop * self.train_env.batch_size,
                                        observers=[self.replay_buffer.add_batch])

        self.training_loop = get_training_loop_fn(self.driver, self.replay_buffer, self.agent, self.steps_per_loop)
        self.compiled = True

    def train(self, *args) -> None:
        """Starts the training of the model. Includes metrics collection and model saving.

        :param *args: All arguments will be passed to `collect_metrics()`.
            This can be usefull to pass callables, testing environments or validation data.
        :type  *args: Any

        :return: None
        :rtype: NoneType, last step is saving the model as a side-effect
        """
        assert self.compiled, "Model must be compiled with model.compile_model() before training."

        self.collect_metrics(*args)  # Initial collection for step 0
        for _ in tqdm(range(self.episodes)):
            loss_info = self.training_loop()

            if not self.global_episode % self.val_every:
                self.collect_metrics(*args)

                with self.writer.as_default():
                    tf.summary.scalar("train_loss", loss_info.loss, step=self.global_episode)

        self.save_model()

    @abstractmethod
    def collect_metrics(self):
        """*args given in train() will be passed to this function."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """Evaluation function to run after training with seperate train-dataset."""
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
