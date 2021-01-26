import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from imbDRL.environments.classifierenv import ClassifierEnv
from imbDRL.metrics import (classification_metrics, decision_function,
                            network_predictions, plot_pr_curve, plot_roc_curve)
from imbDRL.utils import imbalance_ratio
from tensorflow import data
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.sequential import Sequential
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common
from tqdm import tqdm


class TrainDDQN():
    """Wrapper for DDQN training, validation, saving etc."""

    def __init__(self, episodes: int, warmup_steps: int, learning_rate: float, gamma: float, min_epsilon: float, decay_episodes: int,
                 model_path: str = None, log_dir: str = None, batch_size: int = 64, memory_length: int = None,
                 collect_steps_per_episode: int = 1, val_every: int = None, target_update_period: int = 1, target_update_tau: float = 1.0,
                 progressbar: bool = True, n_step_update: int = 1, gradient_clipping: float = 1.0, collect_every: int = 1) -> None:
        """
        Wrapper to make training easier.
        Code is partly based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

        :param episodes: Number of training episodes
        :type  episodes: int
        :param warmup_steps: Number of episodes to fill Replay Buffer with random state-action pairs before training starts
        :type  warmup_steps: int
        :param learning_rate: Learning Rate for the Adam Optimizer
        :type  learning_rate: float
        :param gamma: Discount factor for the Q-values
        :type  gamma: float
        :param min_epsilon: Lowest and final value for epsilon
        :type  min_epsilon: float
        :param decay_episodes: Amount of episodes to decay from 1 to `min_epsilon`
        :type  decay_episodes: int
        :param model_path: Location to save the trained model
        :type  model_path: str
        :param log_dir: Location to save the logs, usefull for TensorBoard
        :type  log_dir: str
        :param batch_size: Number of samples in minibatch to train on each step
        :type  batch_size: int
        :param memory_length: Maximum size of the Replay Buffer
        :type  memory_length: int
        :param collect_steps_per_episode: Amount of data to collect for Replay Buffer each episiode
        :type  collect_steps_per_episode: int
        :param collect_every: Step interval to collect data during training
        :type  collect_every: int
        :param val_every: Validate the model every X episodes using the `collect_metrics()` function
        :type  val_every: int
        :param target_update_period: Update the target Q-network every X episodes
        :type  target_update_period: int
        :param target_update_tau: Parameter for softening the `target_update_period`
        :type  target_update_tau: float
        :param progressbar: Enable or disable the progressbar for collecting data and training
        :type  progressbar: bool

        :return: None
        :rtype: NoneType
        """
        self.episodes = episodes  # Total episodes
        self.warmup_steps = warmup_steps  # Amount of warmup steps before training
        self.batch_size = batch_size  # Batch size of Replay Memory
        self.collect_steps_per_episode = collect_steps_per_episode  # Amount of steps to collect data each episode
        self.collect_every = collect_every  # Step interval to collect data during training
        self.learning_rate = learning_rate  # Learning Rate
        self.gamma = gamma  # Discount factor
        self.min_epsilon = min_epsilon  # Minimal chance of choosing random action
        self.decay_episodes = decay_episodes  # Number of episodes to decay from 1.0 to `EPSILON`
        self.target_update_period = target_update_period  # Period for soft updates
        self.target_update_tau = target_update_tau
        self.progressbar = progressbar  # Enable or disable the progressbar for collecting data and training
        self.n_step_update = n_step_update
        self.gradient_clipping = gradient_clipping  # Clip the loss
        self.compiled = False
        NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

        if memory_length is not None:
            self.memory_length = memory_length  # Max Replay Memory length
        else:
            self.memory_length = warmup_steps

        if val_every is not None:
            self.val_every = val_every  # Validate the policy every `val_every` episodes
        else:
            self.val_every = self.episodes // min(50, self.episodes)  # Can't validate the model 50 times if self.episodes < 50

        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = "./models/" + NOW + ".pkl"

        if log_dir is None:
            log_dir = "./logs/" + NOW
        self.writer = tf.summary.create_file_writer(log_dir)

    def compile_model(self, X_train, y_train, layers: list = [], imb_ratio: float = None, loss_fn=common.element_wise_squared_loss) -> None:
        """Initializes the neural networks, DDQN-agent, collect policies and replay buffer.

        :param X_train: Training data for the model.
        :type  X_train: np.ndarray
        :param y_train: Labels corresponding to `X_train`.  1 for the positive class, 0 for the negative class.
        :param y_train: np.ndarray
        :param layers: List of layers to feed into the TF-agents custom Sequential(!) layer.
        :type  layers: list
        :param imb_ratio: The imbalance ratio of the data.
        :type  imb_ratio: float
        :param loss_fn: Callable loss function
        :type  loss_fn: tf.compat.v1.losses

        :return: None
        :rtype: NoneType
        """
        if imb_ratio is None:
            imb_ratio = imbalance_ratio(y_train)

        self.train_env = TFPyEnvironment(ClassifierEnv(X_train, y_train, imb_ratio))
        self.global_episode = tf.Variable(0, name="global_episode", dtype=np.int64, trainable=False)  # Global train episode counter

        # Custom epsilon decay: https://github.com/tensorflow/agents/issues/339
        epsilon_decay = tf.compat.v1.train.polynomial_decay(
            1.0, self.global_episode, self.decay_episodes, end_learning_rate=self.min_epsilon)

        self.q_net = Sequential(layers, self.train_env.observation_spec())

        self.agent = DdqnAgent(self.train_env.time_step_spec(),
                               self.train_env.action_spec(),
                               q_network=self.q_net,
                               optimizer=Adam(learning_rate=self.learning_rate),
                               td_errors_loss_fn=loss_fn,
                               train_step_counter=self.global_episode,
                               target_update_period=self.target_update_period,
                               target_update_tau=self.target_update_tau,
                               gamma=self.gamma,
                               epsilon_greedy=epsilon_decay,
                               n_step_update=self.n_step_update,
                               gradient_clipping=self.gradient_clipping)
        self.agent.initialize()

        self.random_policy = RandomTFPolicy(self.train_env.time_step_spec(), self.train_env.action_spec())
        self.replay_buffer = TFUniformReplayBuffer(data_spec=self.agent.collect_data_spec,
                                                   batch_size=self.train_env.batch_size,
                                                   max_length=self.memory_length)

        self.warmup_driver = DynamicStepDriver(self.train_env,
                                               self.random_policy,
                                               observers=[self.replay_buffer.add_batch],
                                               num_steps=self.warmup_steps)  # Uses a random policy

        self.collect_driver = DynamicStepDriver(self.train_env,
                                                self.agent.collect_policy,
                                                observers=[self.replay_buffer.add_batch],
                                                num_steps=self.collect_steps_per_episode)  # Uses the epsilon-greedy policy of the agent

        self.agent.train = common.function(self.agent.train)  # Optimalization
        self.warmup_driver.run = common.function(self.warmup_driver.run)
        self.collect_driver.run = common.function(self.collect_driver.run)

        self.compiled = True

    def train(self, *args) -> None:
        """Starts the training of the model. Includes warmup period, metrics collection and model saving.

        :param *args: All arguments will be passed to `collect_metrics()`.
            This can be usefull to pass callables, testing environments or validation data.
            Overwrite the TrainDDQN.collect_metrics() function to use your own *args.
        :type  *args: Any

        :return: None
        :rtype: NoneType, last step is saving the model as a side-effect
        """
        assert self.compiled, "Model must be compiled with model.compile_model(X_train, y_train, layers) before training."

        # Warmup period, fill memory with random actions
        if self.progressbar:
            print(f"\033[92mCollecting data for {self.warmup_steps:_} steps... This might take a few minutes...\033[0m")

        self.warmup_driver.run(time_step=None, policy_state=self.random_policy.get_initial_state(self.train_env.batch_size))

        if self.progressbar:
            print(f"\033[92m{self.replay_buffer.num_frames():_} frames collected!\033[0m")

        dataset = self.replay_buffer.as_dataset(sample_batch_size=self.batch_size, num_steps=self.n_step_update + 1,
                                                num_parallel_calls=data.experimental.AUTOTUNE).prefetch(data.experimental.AUTOTUNE)
        iterator = iter(dataset)

        def _train():
            experiences, _ = next(iterator)
            return self.agent.train(experiences).loss
        _train = common.function(_train)  # Optimalization

        ts = None
        policy_state = self.agent.collect_policy.get_initial_state(self.train_env.batch_size)
        self.collect_metrics(*args)  # Initial collection for step 0
        pbar = tqdm(total=self.episodes, disable=(not self.progressbar), desc="Training the DDQN")  # TQDM progressbar
        for _ in range(self.episodes):
            if not self.global_episode % self.collect_every:
                # Collect a few steps using collect_policy and save to `replay_buffer`
                if self.collect_steps_per_episode != 0:
                    ts, policy_state = self.collect_driver.run(time_step=ts, policy_state=policy_state)
                pbar.update(self.collect_every)  # More stable TQDM updates, collecting could take some time

            # Sample a batch of data from `replay_buffer` and update the agent's network
            train_loss = _train()

            if not self.global_episode % self.val_every:
                with self.writer.as_default():
                    tf.summary.scalar("train_loss", train_loss, step=self.global_episode)

                self.collect_metrics(*args)
        pbar.close()

    def collect_metrics(self, X_val: np.ndarray, y_val: np.ndarray, save_best: str = None):
        """Collects metrics using the trained Q-network.

        :param X_val: Features of validation data, same shape as X_train
        :type  X_val: np.ndarray
        :param y_val: Labels of validation data, same shape as y_train
        :type  y_val: np.ndarray
        :param save_best: Saving the best model of all validation runs based on given metric:
            Choose one of: {Gmean, F1, Precision, Recall, TP, TN, FP, FN}
            This improves stability since the model at the last episode is not guaranteed to be the best model.
        :type  save_best: str
        """
        y_pred = network_predictions(self.agent._target_q_network, X_val)
        stats = classification_metrics(y_val, y_pred)
        avgQ = np.mean(decision_function(self.agent._target_q_network, X_val))  # Max action for each x in X

        if save_best is not None:
            if not hasattr(self, "best_score"):  # If no best model yet
                self.best_score = 0.0

            if stats.get(save_best) >= self.best_score:  # Overwrite best model
                self.save_network()  # Saving directly to avoid shallow copy without trained weights
                self.best_score = stats.get(save_best)

        with self.writer.as_default():
            tf.summary.scalar("AverageQ", avgQ, step=self.global_episode)  # Average Q-value for this epoch
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=self.global_episode)

    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """
        Final evaluation of trained Q-network with X_test and y_test.
        Optional PR and ROC curve comparison to X_train, y_train to ensure no overfitting is taking place.

        :param X_test: Features of test data, same shape as X_train
        :type  X_test: np.ndarray
        :param y_test: Labels of test data, same shape as y_train
        :type  y_test: np.ndarray
        :param X_train: Features of train data
        :type  X_train: np.ndarray
        :param y_train: Labels of train data
        :type  y_train: np.ndarray
        """
        if hasattr(self, "best_score"):
            print(f"\033[92mBest score: {self.best_score:6f}!\033[0m")
            network = self.load_network(self.model_path)  # Load best saved model
        else:
            network = self.agent._target_q_network  # Load latest target model

        if (X_train is not None) and (y_train is not None):
            plot_pr_curve(network, X_test, y_test, X_train, y_train)
            plot_roc_curve(network, X_test, y_test, X_train, y_train)

        y_pred = network_predictions(network, X_test)
        return classification_metrics(y_test, y_pred)

    def save_network(self):
        """Saves Q-network as pickle to `model_path`."""
        with open(self.model_path, "wb") as f:  # Save Q-network as pickle
            pickle.dump(self.agent._target_q_network, f)

    @staticmethod
    def load_network(fp: str):
        """Static method to load Q-network pickle from given filepath.

        :param fp: Filepath to the saved pickle of the network
        :type  fp: str

        :returns: The network-object loaded from a pickle file.
        :rtype: tensorflow.keras.models.Model
        """
        with open(fp, "rb") as f:  # Load the Q-network
            network = pickle.load(f)
        return network
