from datetime import datetime

from dqnimp.data import load_data
from dqnimp.environments import ClassifyEnv
from dqnimp.train import TrainCustom
from tf_agents.environments.tf_py_environment import TFPyEnvironment

# Code is based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

episodes = 1_000  # Total episodes
warmup_episodes = 1_000  # Amount of warmup steps before training
memory_length = 100_000  # Max Replay Memory length
collect_steps_per_episode = 4
target_model_update = 10

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )
dense_layers = (256, 128, )
dropout_layers = None

lr = 0.00025  # Learning Rate
gamma = 0.0  # Discount factor
min_epsilon = 0.01  # Minimal chance of choosing random action
decay_episodes = 5_000  # Number of episodes to decay from 1.0 to `EPSILON`
normalization = True
model_dir = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
log_dir = "./logs/" + NOW

imb_rate = 0.05  # Imbalance rate
min_class = [3]  # Minority classes, must be same as trained model
maj_class = [8]  # Majority classes, must be same as trained model
datasource = "mnist"  # The dataset to be selected
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class, normalization=normalization)

train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))  # Change Python environment to TF environment
val_env = TFPyEnvironment(ClassifyEnv(X_val, y_val, imb_rate))

model = TrainCustom(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, model_dir, log_dir,
                    collect_steps_per_episode=collect_steps_per_episode, target_model_update=target_model_update)

model.compile(train_env, val_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])

q_net = TrainCustom.load_model(model_dir)
