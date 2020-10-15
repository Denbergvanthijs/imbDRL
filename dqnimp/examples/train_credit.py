from datetime import datetime

from dqnimp.data import load_data
from dqnimp.environments import ClassifyEnv
from dqnimp.trainwrapper import TrainCustom
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 100_000  # Total number of episodes
warmup_episodes = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory

conv_layers = None
dense_layers = (256, 256, )
dropout_layers = None  # (0.1, 0.1, )

lr = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.05  # Minimal and final chance of choosing random action
decay_episodes = 5_000  # Number of episodes to decay from 1.0 to `min_epsilon`
target_model_update = 2_500  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 0.1

model_dir = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
log_dir = "./logs/" + NOW

imb_rate = 0.00173  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
datasource = "credit"  # The dataset to be selected
normalization = True
X_train, y_train, X_test, y_test, X_val, y_val = load_data(datasource, imb_rate, min_class, maj_class, normalization=normalization)

# Change OpenAI Gym environment to Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))
val_env = TFPyEnvironment(ClassifyEnv(X_val, y_val, imb_rate))

model = TrainCustom(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, model_dir,
                    log_dir, target_model_update=target_model_update, target_update_tau=target_update_tau)
model.compile(train_env, val_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
