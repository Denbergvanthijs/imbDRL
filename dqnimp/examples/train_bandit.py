import functools
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from dqnimp.data import load_data
from dqnimp.utils import metrics_by_network
from tensorflow.keras.optimizers import Adam
from tf_agents.bandits.agents import neural_epsilon_greedy_agent
from tf_agents.bandits.agents.examples.v2.trainer import (get_replay_buffer,
                                                          get_training_loop_fn)
from tf_agents.bandits.environments import classification_environment as ce
from tf_agents.bandits.environments import environment_utilities as env_util
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tqdm import tqdm

tf.compat.v1.enable_v2_behavior()  # The trainer only runs with V2 enabled.
BATCH_SIZE = 100
TRAINING_LOOPS = 1_000
STEPS_PER_LOOP = 4
log_dir = "./logs/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))

imb_rate = 0.05  # Imbalance rate
min_class = [3]  # Minority classes, must be same as trained model
maj_class = [8]  # Majority classes, must be same as trained model
datasource = "mnist"  # The dataset to be selected
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class, normalization=True)

with tf.device('/CPU:0'):  # due to b/128333994
    distr = tfp.distributions.Bernoulli(probs=[[imb_rate, -imb_rate], [-1, 1]], dtype=tf.float32)
    reward_distr = (tfp.bijectors.Shift([[imb_rate, -imb_rate], [-1, 1]])
                    (tfp.bijectors.Scale([[1, 1], [1, 1]])
                     (distr)))
    distr = tfp.distributions.Independent(reward_distr, reinterpreted_batch_ndims=2)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    environment = ce.ClassificationBanditEnvironment(dataset, distr, BATCH_SIZE)
    optimal_reward_fn = functools.partial(env_util.compute_optimal_reward_with_classification_environment, environment=environment)
    optimal_action_fn = functools.partial(env_util.compute_optimal_action_with_classification_environment, environment=environment)

    network = q_network.QNetwork(input_tensor_spec=environment.observation_spec(),
                                 action_spec=environment.action_spec(),
                                 conv_layer_params=((32, (5, 5), 2), (32, (5, 5), 2), ),
                                 fc_layer_params=(256, 128))

    agent = neural_epsilon_greedy_agent.NeuralEpsilonGreedyAgent(time_step_spec=environment.time_step_spec(),
                                                                 action_spec=environment.action_spec(),
                                                                 reward_network=network,
                                                                 optimizer=Adam(learning_rate=0.0005),
                                                                 epsilon=0.01)
    data_spec = agent.policy.trajectory_spec
    replay_buffer = get_replay_buffer(data_spec, environment.batch_size, STEPS_PER_LOOP)

    step_metric = tf_metrics.EnvironmentSteps()
    add_batch_fn = replay_buffer.add_batch

    observers = [add_batch_fn, step_metric]

    driver = dynamic_step_driver.DynamicStepDriver(env=environment,
                                                   policy=agent.collect_policy,
                                                   num_steps=STEPS_PER_LOOP * environment.batch_size,
                                                   observers=observers)

    training_loop = get_training_loop_fn(driver, replay_buffer, agent, STEPS_PER_LOOP)
    writer = tf.summary.create_file_writer(log_dir)

    for i in tqdm(range(TRAINING_LOOPS + 1)):
        training_loop()

        if not i % 20:
            stats = metrics_by_network(agent._reward_network, X_val, y_val)
            with writer.as_default():
                for k, v in stats.items():
                    tf.summary.scalar(k, v, step=i)

stats = metrics_by_network(agent._reward_network, X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
