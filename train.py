from __future__ import absolute_import, division, print_function

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
from utils import collect_data

# Code is based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

num_iterations = 20_000  # Total episodes
initial_collect_steps = 1_000  # Warmup period
collect_steps_per_iteration = 1
replay_buffer_max_length = 100_000  # Replay Memory length
batch_size = 64  # Batch size of Replay Memory
learning_rate = 0.001  # Learning Rate
log_interval = 200  # Print step and loss every N steps
num_eval_episodes = 10  # Number of episodes to use to calculate the average return
num_test_episodes = 50  # Number of episodes to use to test the final policy
eval_interval = 1_000  # Evaluate the policy every N steps
fc_layer_params = (100, )  # Q network architecture
TARGET_MODEL_UPDATE = 1  # Period for soft updates
GAMMA = 1.0  # Discount factor
EPSILON = 0.1  # Minimal chance of choosing random action
DECAY_STEPS = 1_000  # Number of episodes to decay from 1.0 to `EPSILON`
policy_dir = "./models/" + datetime.now().strftime('%Y%m%d_%H%M%S')
logs_dir = "./logs/" + datetime.now().strftime('%Y%m%d_%H%M%S')

if False:  # Easy switch between custom and CartPole environments
    imb_rate = 0.00173  # Imbalance rate
    min_class = [1]  # Minority classes, must be same as trained model
    maj_class = [0]  # Majority classes, must be same as trained model
    datasource = "credit"  # The dataset to be selected
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)  # Load all data
    train_py_env = ClassifyEnv(X_train, y_train, imb_rate)
    eval_py_env = ClassifyEnv(X_val, y_val, imb_rate)
    test_py_env = ClassifyEnv(X_val, y_val, imb_rate)
else:
    train_py_env = suite_gym.load('CartPole-v0')
    eval_py_env = suite_gym.load('CartPole-v0')
    test_py_env = suite_gym.load('CartPole-v0')

train_env = TFPyEnvironment(train_py_env)  # Change Python environment to TF environment
eval_env = TFPyEnvironment(eval_py_env)  # Numpy arrays will be converted to TF Tensors
test_env = TFPyEnvironment(test_py_env)
del train_py_env, eval_py_env, test_py_env

q_net = QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params)
optimizer = Adam(learning_rate=learning_rate)

global_step = tf.Variable(0, name="global_step", dtype=np.int64, trainable=False)  # Counter for global train episode counter
# Custom epsilon decay: https://github.com/tensorflow/agents/issues/339
epsilon_policy = tf.compat.v1.train.polynomial_decay(1.0, global_step, DECAY_STEPS, end_learning_rate=EPSILON)
agent = DdqnAgent(train_env.time_step_spec(),
                  train_env.action_spec(),
                  q_network=q_net,
                  optimizer=optimizer,
                  td_errors_loss_fn=common.element_wise_squared_loss,
                  train_step_counter=global_step,
                  target_update_period=TARGET_MODEL_UPDATE,
                  gamma=GAMMA,
                  epsilon_greedy=epsilon_policy)
agent.initialize()

random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())  # Choose random action
replay_buffer = TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                      batch_size=train_env.batch_size,
                                      max_length=replay_buffer_max_length)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)  # Warmup period
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)
agent.train = common.function(agent.train)

# Evaluate the agent's policy once before training.
eval_summary_writer = tf.summary.create_file_writer(logs_dir)
eval_metrics = [AverageReturnMetric(buffer_size=num_eval_episodes),
                AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
                NumberOfEpisodes()]

# Calculte metrics once for step 0
metric_utils.eager_compute(eval_metrics, eval_env, agent.policy, num_episodes=num_eval_episodes,
                           train_step=global_step, summary_writer=eval_summary_writer, use_function=False)

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)
    # TODO: print(replay_buffer.num_frames())

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    # TODO: print(len(experience))
    train_loss = agent.train(experience).loss

    if not global_step % log_interval:
        print(f"step={global_step.numpy()}; {train_loss=:.6f}")

    # https://github.com/tensorflow/agents/issues/385
    if not global_step % eval_interval:
        metric_utils.eager_compute(eval_metrics, eval_env, agent.policy, num_episodes=num_eval_episodes,
                                   train_step=global_step, summary_writer=eval_summary_writer, use_function=False)

tf_policy_saver = PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)  # Save policy
saved_policy = tf.saved_model.load(policy_dir)

test_results = metric_utils.eager_compute(eval_metrics, test_env, saved_policy, num_episodes=num_test_episodes, use_function=False)
print(f"{[(k, v.numpy()) for k, v in test_results.items()]}")
