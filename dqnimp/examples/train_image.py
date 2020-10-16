from datetime import datetime

from dqnimp.data import load_data
from dqnimp.environments import ClassifyEnv
from dqnimp.trainwrapper import TrainCustom
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 50_000  # Total episodes, 120_000 in original paper, the original code only trains every 4 steps
warmup_episodes = 50_000  # Amount of warmup steps before training
memory_length = 50_000  # Max Replay Memory length
collect_steps_per_episode = 1  # Since train_interval=4 in original code and `episodes` is divided by 4
target_model_update = 2_500  # Since train_interval=4 in original code and `episodes` is divided by 4, update target model 4x faster
target_update_tau = 0.2  # Soften the target model update

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )  # Convolutional layers
dense_layers = (256, )  # Dense layers
dropout_layers = None  # Dropout layers

lr = 0.001  # Learning Rate
gamma = 0.0  # Discount factor
min_epsilon = 0.01  # Minimal chance of choosing random action
decay_episodes = 25_000  # Number of episodes to decay from 1.0 to `min_epsilon`, divided by 4

model_dir = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
log_dir = "./logs/" + NOW

imb_rate = 0.01  # Imbalance rate
min_class = [2]  # Minority classes, same setup as in original paper
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes
datasource = "mnist"  # The dataset to be selected
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class)

train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))  # Change Python environment to TF environment
val_env = TFPyEnvironment(ClassifyEnv(X_val, y_val, imb_rate))

model = TrainCustom(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, model_dir, log_dir,
                    collect_steps_per_episode=collect_steps_per_episode, target_model_update=target_model_update, target_update_tau=target_update_tau)

model.compile(train_env, val_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
