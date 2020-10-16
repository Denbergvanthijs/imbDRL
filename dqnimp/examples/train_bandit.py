from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dqnimp.data import load_data
from dqnimp.metrics import metrics_by_network
from tensorflow.keras.optimizers import Adam
from tf_agents.bandits.agents.examples.v2.trainer import (get_replay_buffer,
                                                          get_training_loop_fn)
from tf_agents.bandits.agents.neural_epsilon_greedy_agent import \
    NeuralEpsilonGreedyAgent
from tf_agents.bandits.environments.classification_environment import \
    ClassificationBanditEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.metrics import tf_metrics
from tf_agents.networks.q_network import QNetwork
from tqdm import tqdm

tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.
batch_size = 64
training_loops = 1_000
steps_per_loop = 64

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )
dense_layers = (256, )
dropout_layers = None

lr = 0.001  # Learning Rate
min_epsilon = 0.01  # Minimal chance of choosing random action
decay_episodes = 250  # Number of episodes to decay from 1.0 to `min_epsilon`, divided by 4
global_step = tf.Variable(0, name="global_step", dtype=np.int64, trainable=False)  # Global train episode counter
eps_decay = tf.compat.v1.train.polynomial_decay(1.0, global_step, decay_episodes, end_learning_rate=min_epsilon)

log_dir = "./logs/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))

imb_rate = 0.01  # Imbalance rate
min_class = [2]  # Minority classes, same setup as in original paper
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes
datasource = "mnist"  # The dataset to be selected
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class, normalization=True)

with tf.device('/GPU:0'):  # due to b/128333994
    distr = tfp.distributions.Bernoulli(probs=[[imb_rate, -imb_rate], [-1, 1]], dtype=tf.float32)
    reward_distr = (tfp.bijectors.Shift([[imb_rate, -imb_rate], [-1, 1]])
                    (tfp.bijectors.Scale([[1, 1], [1, 1]])
                     (distr)))
    distr = tfp.distributions.Independent(reward_distr, reinterpreted_batch_ndims=2)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    environment = ClassificationBanditEnvironment(dataset, distr, batch_size)

    network = QNetwork(input_tensor_spec=environment.observation_spec(),
                       action_spec=environment.action_spec(),
                       conv_layer_params=conv_layers,
                       fc_layer_params=dense_layers)

    agent = NeuralEpsilonGreedyAgent(time_step_spec=environment.time_step_spec(),
                                     action_spec=environment.action_spec(),
                                     reward_network=network,
                                     optimizer=Adam(learning_rate=lr),
                                     epsilon=eps_decay,
                                     train_step_counter=global_step)

    replay_buffer = get_replay_buffer(agent.policy.trajectory_spec, environment.batch_size, steps_per_loop)

    driver = DynamicStepDriver(env=environment,
                               policy=agent.collect_policy,
                               num_steps=steps_per_loop * environment.batch_size,
                               observers=[replay_buffer.add_batch])

    training_loop = get_training_loop_fn(driver, replay_buffer, agent, steps_per_loop)
    writer = tf.summary.create_file_writer(log_dir)

    for i in tqdm(range(training_loops + 1)):
        training_loop()

        if not i % 20:
            stats = metrics_by_network(agent._reward_network, X_val, y_val)
            with writer.as_default():
                for k, v in stats.items():
                    tf.summary.scalar(k, v, step=i)

stats = metrics_by_network(agent._reward_network, X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
