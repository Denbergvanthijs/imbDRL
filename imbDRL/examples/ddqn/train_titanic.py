import os

import numpy as np
import tensorflow_datasets as tfds
from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val
from imbDRL.utils import rounded_dict
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU on structured data

episodes = 16_000  # Total number of episodes
warmup_steps = 16_000  # Amount of warmup steps to collect data with random policy
memory_length = warmup_steps  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 500
collect_every = 500

target_update_period = 400  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update
n_step_update = 1

layers = [Dense(256, activation="relu"), Dropout(0.2),
          Dense(256, activation="relu"), Dropout(0.2),
          Dense(2, activation=None)]  # No activation, pure Q-values

learning_rate = 0.00025  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.5  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon``

min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes

df = tfds.as_dataframe(*tfds.load("titanic", split='train', with_info=True))
y = df.survived.values
df = df.drop(columns=["survived", "features/boat", "features/cabin", "features/home.dest", "features/name", "features/ticket"])
df = df.astype(np.float64)
df = (df - df.min()) / (df.max() - df.min())  # Normalization should happen after splitting train and test sets

X_train, X_test, y_train, y_test = train_test_split(df.to_numpy(), y, stratify=y, test_size=0.2)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test,
                                                                    min_class, maj_class, val_frac=0.2)

model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

model.compile_model(X_train, y_train, layers)
model.q_net.summary()
model.train(X_val, y_val, "F1")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.824172, 'F1': 0.781395, 'Precision': 0.730435, 'Recall': 0.84, 'TP': 84, 'TN': 131, 'FP': 31, 'FN': 16}
