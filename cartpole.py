from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from utils import collect_data, collect_step, compute_avg_return

# Code is based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

num_iterations = 20_000  # Total episodes
initial_collect_steps = 100  # Warmup period
collect_steps_per_iteration = 1
replay_buffer_max_length = 100_000  # Replay Memory length
batch_size = 64  # Batch size of Replay Memory
learning_rate = 0.001  # Learning Rate
log_interval = 200  # Print step and loss every N steps
num_eval_episodes = 10  # Number of episodes to use to calculate the average return
eval_interval = 1_000  # Evaluate the policy every N steps
fc_layer_params = (100,)  # Q network architecture
env_name = 'CartPole-v0'
tf.compat.v1.enable_v2_behavior()

env = suite_gym.load(env_name)
train_py_env = suite_gym.load(env_name)  # Make individual train and evaluation environments
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)  # Change Python environment to TF environment
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)  # Numpy arrays will be converted to TF Tensors
del train_py_env, eval_py_env

q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                           train_env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=train_step_counter)
agent.initialize()

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())  # Baseline, choose random action
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=train_env.batch_size,
                                                               max_length=replay_buffer_max_length)


collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)  # Warmup period
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)  # Reset the train step

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()

    if not step % log_interval:
        print(f"{step=}; {train_loss=:.6f}")

    if not step % eval_interval:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print(f"{step=}; Average Return={avg_return:.2f}")
        returns.append(avg_return)


iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel("Average Return")
plt.xlabel("Iterations")
plt.ylim(top=250)
plt.show()
