from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val, load_image
from imbDRL.utils import rounded_dict
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tf_agents.utils import common

episodes = 120_000  # Total number of episodes
warmup_steps = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 1000
collect_every = 1000

target_update_period = 10_000
target_update_tau = 1
n_step_update = 4

layers = [Conv2D(32, (5, 5), padding="Same", activation="relu"),
          MaxPooling2D(pool_size=(2, 2)),
          Conv2D(32, (5, 5), padding="Same", activation="relu"),
          MaxPooling2D(pool_size=(2, 2)),
          Flatten(),
          Dense(256, activation="relu"),
          Dense(2, activation=None)]

learning_rate = 0.00025  # Learning rate
gamma = 0.1  # Discount factor
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_episodes = 100_000  # Number of episodes to decay from 1.0 to `min_epsilon`

loss_fn = common.element_wise_huber_loss

imb_ratio = 0.04  # Imbalance rate
min_class = [4, 5, 6]  # Minority classes
maj_class = [7, 8, 9]  # Majority classes
X_train, y_train, X_test, y_test = load_image("famnist")
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class,
                                                                    imb_ratio=imb_ratio, imb_test=False, val_frac=0.1)

model = TrainDDQN(episodes, warmup_steps, learning_rate, gamma, min_epsilon, decay_episodes, target_update_period=target_update_period,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

model.compile_model(X_train, y_train, layers, imb_ratio=imb_ratio, loss_fn=loss_fn)
model.train(X_val, y_val, "Gmean")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.964648, 'F1': 0.964877, 'Precision': 0.959157, 'Recall': 0.970667, 'TP': 2912, 'TN': 2876, 'FP': 124, 'FN': 88}
