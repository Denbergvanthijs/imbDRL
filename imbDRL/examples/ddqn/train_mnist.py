import os

from imbDRL.agents.ddqn import TrainDDQN
from imbDRL.data import get_train_test_val, load_image
from imbDRL.utils import rounded_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU is faster than GPU

episodes = 120_000  # Total number of episodes
warmup_episodes = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory
batch_size = 32
collect_steps_per_episode = 1000
collect_every = 1000

target_model_update = 10_000
target_update_tau = 1
n_step_update = 4

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )  # Convolutional layers
dense_layers = (256, )  # Dense layers
dropout_layers = None  # Dropout layers

lr = 0.00025  # Learning rate
gamma = 0.1  # Discount factor
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_episodes = 100_000  # Number of episodes to decay from 1.0 to `min_epsilon`

imb_rate = 0.01  # Imbalance rate
min_class = [2]  # Minority classes
maj_class = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # Majority classes
X_train, y_train, X_test, y_test, = load_image("mnist")
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test,
                                                                    min_class, maj_class, val_frac=0.1, imb_rate=imb_rate, imb_test=False)

model = TrainDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, target_model_update=target_model_update,
                  target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode,
                  memory_length=memory_length, collect_every=collect_every, n_step_update=n_step_update)

model.compile_model(X_train, y_train, imb_rate, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val, "Gmean")

stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.987032, 'F1': 0.955702, 'Precision': 0.930275, 'Recall': 0.982558, 'TP': 1014, 'TN': 8892, 'FP': 76, 'FN': 18}
