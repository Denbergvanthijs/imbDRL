from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.environments import ClassifyEnv
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.utils import rounded_dict
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 50_000  # Total number of episodes
warmup_episodes = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 50_000  # Max length of the Replay Memory

target_model_update = 1  # Period to overwrite the target Q-network with the default Q-network  # 500
target_update_tau = 0.0005  # Soften the target model update  # 0.01

conv_layers = None  # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.2, 0.2, )  # Dropout layers

lr = 0.001  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.05  # Minimal and final chance of choosing random action
decay_episodes = 25_000  # Number of episodes to decay from 1.0 to `min_epsilon`

imb_rate = 0.001729  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

# Change Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))

model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes,
                        target_model_update=target_model_update, target_update_tau=target_update_tau)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(rounded_dict(stats))
# ("Gmean", 0.914499) ("Fdot5", 0.756458) ("F1", 0.784689) ("F2", 0.815109) ("TP", 82) ("TN", 56835) ("FP", 29) ("FN", 16)
