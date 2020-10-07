from __future__ import absolute_import, division, print_function

import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tf_agents.agents.dqn.dqn_agent import DdqnAgent
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.eval import metric_utils
from tf_agents.metrics.tf_metrics import (AverageEpisodeLengthMetric,
                                          AverageReturnMetric,
                                          NumberOfEpisodes)
from tf_agents.networks.q_network import QNetwork
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import \
    TFUniformReplayBuffer
from tf_agents.utils import common

from environment import ClassifyEnv
from get_data import load_data
from utils import collect_data, compute_metrics

# Code is based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

EPISODES = 2_00  # Total episodes
BATCH_SIZE = 64  # Batch size of Replay Memory
WARMUP_STEPS = 1_000  # Amount of warmup steps before training
MEMORY_LENGTH = 100_000  # Max Replay Memory length
COLLECT_STEPS_PER_EPISODE = 1  # Amount of steps to collect data each episode

LOG_EVERY = 200  # Print step and loss every `LOG_EVERY` episodes
VAL_EVERY = 1_000  # Validate the policy every `EVAL_EVERY` episodes
VAL_EPISODES = 10  # Number of episodes to use to calculate metrics during training
TEST_EPISODES = 50  # Number of episodes to use to test the final policy after training
LAYER_PARAMS = (100, )  # Q network architecture

LR = 0.001  # Learning Rate
GAMMA = 1.0  # Discount factor
EPSILON = 0.1  # Minimal chance of choosing random action
DECAY_STEPS = 1_000  # Number of episodes to decay from 1.0 to `EPSILON`
TARGET_MODEL_UPDATE = 1  # Period for soft updates

MODEL_DIR = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
WRITER = tf.summary.create_file_writer("./logs/" + NOW)
METRICS = [AverageReturnMetric(buffer_size=VAL_EPISODES), AverageEpisodeLengthMetric(buffer_size=VAL_EPISODES), NumberOfEpisodes()]
global_step = tf.Variable(0, name="global_step", dtype=np.int64, trainable=False)  # Global train episode counter

if True:  # Easy switch between custom and CartPole environments
    imb_rate = 0.00173  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data

    train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))  # Change Python environment to TF environment
    val_env = TFPyEnvironment(ClassifyEnv(X_val, y_val, imb_rate))
    test_env = TFPyEnvironment(ClassifyEnv(X_test, y_test, imb_rate))
else:
    train_env = TFPyEnvironment(suite_gym.load('CartPole-v0'))  # Change Python environment to TF environment
    val_env = TFPyEnvironment(suite_gym.load('CartPole-v0'))
    test_env = TFPyEnvironment(suite_gym.load('CartPole-v0'))

# Custom epsilon decay: https://github.com/tensorflow/agents/issues/339
epsilon_decay = tf.compat.v1.train.polynomial_decay(1.0, global_step, DECAY_STEPS, end_learning_rate=EPSILON)
q_net = QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=LAYER_PARAMS)
optimizer = Adam(learning_rate=LR)

agent = DdqnAgent(train_env.time_step_spec(),
                  train_env.action_spec(),
                  q_network=q_net,
                  optimizer=optimizer,
                  td_errors_loss_fn=common.element_wise_squared_loss,
                  train_step_counter=global_step,
                  target_update_period=TARGET_MODEL_UPDATE,
                  gamma=GAMMA,
                  epsilon_greedy=epsilon_decay)
agent.initialize()

random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())  # Choose random action from `action_spec`
replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec, batch_size=train_env.batch_size, max_length=MEMORY_LENGTH)
collect_data(train_env, random_policy, replay_buffer, WARMUP_STEPS)  # Warmup period, fill memory with random actions

dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=BATCH_SIZE, num_steps=2).prefetch(3)
iterator = iter(dataset)
agent.train = common.function(agent.train)  # Optimalization

# Calculte metrics once at `global_step` 0
metric_utils.eager_compute(METRICS, val_env, agent.policy, num_episodes=VAL_EPISODES,
                           train_step=global_step, summary_writer=WRITER, use_function=False)

for _ in range(EPISODES):
    # Collect a few steps using collect_policy and save to `replay_buffer`
    collect_data(train_env, agent.collect_policy, replay_buffer, COLLECT_STEPS_PER_EPISODE)
    # TODO: print(replay_buffer.num_frames())

    # Sample a batch of data from `replay_buffer` and update the agent's network
    experience, unused_info = next(iterator)  # TODO: print(len(experience))
    train_loss = agent.train(experience).loss

    if not global_step % LOG_EVERY:
        print(f"step={global_step.numpy()}; {train_loss=:.6f}")

    if not global_step % VAL_EVERY:
        metric_utils.eager_compute(METRICS, val_env, agent.policy, num_episodes=VAL_EPISODES,
                                   train_step=global_step, summary_writer=WRITER, use_function=False)

        stats = compute_metrics(agent._target_q_network, X_val, y_val)
        with WRITER.as_default():
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=global_step)

# policy_saver = PolicySaver(agent.policy, train_step=global_step)
# policy_saver.save(POLICY_DIR)  # Save policy to `POLICY_DIR`
# saved_policy = tf.saved_model.load(POLICY_DIR)  # Load the saved policy, optional

with open(MODEL_DIR + ".pkl", "wb") as f:  # Save Q-network as pickle
    pickle.dump(agent._target_q_network, f)

with open(MODEL_DIR + ".pkl", "rb") as f:  # Load the Q-network
    network = pickle.load(f)

test_results = compute_metrics(network, X_test, y_test)
print(*[(k, round(v, 6)) for k, v in test_results.items()])
