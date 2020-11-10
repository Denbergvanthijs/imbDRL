from imbDRL.data import get_train_test_val, load_sepsis
from imbDRL.environments import ClassifyEnv
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.utils import rounded_dict
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 50_000  # Total number of episodes
warmup_episodes = 30_000  # Amount of warmup steps to collect data with random policy
memory_length = 30_000  # Max length of the Replay Memory

target_model_update = episodes // 100  # Period to overwrite the target Q-network with the default Q-network  # 500
target_update_tau = 0.1  # Soften the target model update  # 0.01

conv_layers = None  # Convolutional layers
dense_layers = (40, 40, )  # Dense layers
dropout_layers = (0.5, 0.1, )  # Dropout layers

lr = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.1  # Minimal and final chance of choosing random action
decay_episodes = 25_000  # Number of episodes to decay from 1.0 to `min_epsilon`

imb_rate = 0.1097  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_sepsis(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

# Change Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))

model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes,
                        target_model_update=target_model_update, target_update_tau=target_update_tau)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test, X_train, y_train, plot=True)
print(rounded_dict(stats))
# {'Gmean': 0.669208, 'Fdot5': 0.217198, 'F1': 0.289522, 'F2': 0.434059, 'TP': 391, 'TN': 3775, 'FP': 1709, 'FN': 210}
