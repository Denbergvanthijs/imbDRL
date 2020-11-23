from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.environments import ClassifyEnv
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.utils import rounded_dict
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 25_000  # Total number of episodes
warmup_episodes = 60_000  # Amount of warmup steps to collect data with random policy
memory_length = warmup_episodes  # Max length of the Replay Memory
batch_size = 128
collect_steps_per_episode = 0

target_model_update = episodes // min(50, episodes)  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 0.9  # Soften the target model update

conv_layers = None  # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.2, 0.2, )  # Dropout layers

lr = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.05  # Minimal and final chance of choosing random action
decay_episodes = episodes // 10  # Number of episodes to decay from 1.0 to `min_epsilon``

imb_rate = 0.001729  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class, val_frac=0.2)

# Change Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))

model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, target_model_update=target_model_update,
                        target_update_tau=target_update_tau, batch_size=batch_size, collect_steps_per_episode=collect_steps_per_episode)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.886257, 'F1': 0.793814, 'Precision': 0.802083, 'Recall': 0.785714, 'TP': 77, 'TN': 56845, 'FP': 19, 'FN': 21}
