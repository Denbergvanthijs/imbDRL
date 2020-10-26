from imbDRL.data import get_train_test_val, load_imdb
from imbDRL.environments import ClassifyEnv
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 50_000  # Total number of episodes
warmup_episodes = 25_000  # Amount of warmup steps to collect data with random policy
memory_length = 50_000  # Max length of the Replay Memory

target_model_update = 5000  # Period to overwrite the target Q-network with the default Q-network
target_update_tau = 1  # Soften the target model update

conv_layers = None  # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.0, 0.2, )  # Dropout layers

lr = 0.00025  # Learning rate
gamma = 0.0  # Discount factor
min_epsilon = 0.1  # Minimal and final chance of choosing random action
decay_episodes = 25_000  # Number of episodes to decay from 1.0 to `min_epsilon`
batch_size = 128

imb_rate = 0.1  # Imbalance rate
min_class = [0]  # Minority classes
maj_class = [1]  # Majority classes
X_train, y_train, X_test, y_test, = load_imdb()
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

# Change Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))

model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes,
                        target_model_update=target_model_update, target_update_tau=target_update_tau, batch_size=batch_size)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
# ('Gmean', 0.33947) ('Fdot5', 0.112328) ('F1', 0.166982) ('F2', 0.325222) ('TP', 1104) ('TN', 1631) ('FP', 10869) ('FN', 146)
