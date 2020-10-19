from datetime import datetime

from dqnimp.data import get_train_test_val, load_creditcard
from dqnimp.environments import ClassifyEnv
from dqnimp.trainwrapper import TrainCustom
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 50_000  # Total number of episodes
warmup_episodes = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 50_000  # Max length of the Replay Memory

target_model_update = 2_500  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 0.2  # Soften the target model update

conv_layers = None  # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.2, 0.2, )  # Dropout layers

lr = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_episodes = 25_000  # Number of episodes to decay from 1.0 to `min_epsilon`

model_dir = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
log_dir = "./logs/" + NOW

imb_rate = 0.00173  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

# Change Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))
val_env = TFPyEnvironment(ClassifyEnv(X_val, y_val, imb_rate))

model = TrainCustom(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, model_dir,
                    log_dir, target_model_update=target_model_update, target_update_tau=target_update_tau)

model.compile_model(train_env, val_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
# ('Gmean', 0.903222) ('Fdot5', 0.711744) ('F1', 0.747664) ('F2', 0.787402) ('TP', 80) ('TN', 56828) ('FP', 36) ('FN', 18)
