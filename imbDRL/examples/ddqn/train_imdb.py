import os

from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val, load_imdb
from imbDRL.utils import rounded_dict
from tensorflow.keras.layers import Dense

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU

episodes = 150_000  # Total number of episodes
warmup_steps = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 1000
collect_every = 1000

target_update_period = 10_000
target_update_tau = 1
n_step_update = 4

layers = [Dense(250, activation="relu"),
          Dense(2, activation=None)]

learning_rate = 0.00025  # Learning rate
gamma = 0.1  # Discount factor
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_episodes = 100_000  # Number of episodes to decay from 1.0 to `min_epsilon`

imb_ratio = 0.1  # Imbalance rate
min_class = [0]  # Minority classes
maj_class = [1]  # Majority classes
X_train, y_train, X_test, y_test = load_imdb(config=(5_000, 500))
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test,
                                                                    min_class, maj_class, val_frac=0.1, imb_ratio=imb_ratio, imb_test=False)

model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

model.compile_model(X_train, y_train, layers, imb_ratio=imb_ratio)
model.train(X_test, y_test, "Gmean")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.286451, 'F1': 0.152846, 'Precision': 0.4967, 'Recall': 0.09032, 'TP': 1129, 'TN': 11356, 'FP': 1144, 'FN': 11371}
