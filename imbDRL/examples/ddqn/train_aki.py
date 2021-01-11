import os

from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val, load_csv
from imbDRL.utils import rounded_dict
from tensorflow.keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

episodes = 25_000  # Total number of episodes
warmup_steps = 32_000  # Amount of warmup steps to collect data with random policy
memory_length = warmup_steps  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 2000
collect_every = episodes // 100

target_update_period = episodes // 30  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update
n_step_update = 4

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]  # No activation, pure Q-values

learning_rate = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.5  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon`

min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test = load_csv("./data/aki0.csv", "./data/aki1.csv", "aki", ["hadm_id"], normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test,
                                                                    min_class, maj_class, val_frac=0.2)

# X_train = X_train.reshape(X_train.shape[:-1] + (18, 7))  # RNN reshaping
# X_test = X_test.reshape(X_test.shape[:-1] + (18, 7))
# X_val = X_val.reshape(X_val.shape[:-1] + (18, 7))

model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

model.compile_model(X_train, y_train, layers)
model.train(X_val, y_val, "F1")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.736675, 'F1': 0.514218, 'Precision': 0.396468, 'Recall': 0.731461, 'TP': 651, 'TN': 2849, 'FP': 991, 'FN': 239}
