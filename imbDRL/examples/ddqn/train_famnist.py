from imbDRL.data import get_train_test_val, load_image
from imbDRL.environments import ClassifyEnv
from imbDRL.examples.ddqn.example_classes import TrainCustomDDQN
from imbDRL.utils import rounded_dict
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.utils import common

episodes = 120_000 - 50_000  # Total episodes, 120_000 in original paper, the original code only trains every 4 steps
warmup_episodes = 50_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory
collect_steps_per_episode = 0  # Since train_interval=4 in original code and `episodes` is divided by 4
target_model_update = 10_000  # Since train_interval=4 in original code and `episodes` is divided by 4, update target model 4x faster
target_update_tau = 1  # Soften the target model update
n_step_update = 4
batch_size = 32

conv_layers = ((32, (5, 5), 2), (32, (5, 5), 2), )  # Convolutional layers
dense_layers = (256, )  # Dense layers
dropout_layers = None  # Dropout layers

lr = 0.00025  # Learning rate
gamma = 0.5  # Discount factor
min_epsilon = 0.1  # Minimal and final chance of choosing random action
decay_episodes = 100_000  # Number of episodes to decay from 1.0 to `min_epsilon`, divided by 4

imb_rate = 0.04  # Imbalance rate
min_class = [4, 5, 6]  # Minority classes
maj_class = [7, 8, 9]  # Majority classes
X_train, y_train, X_test, y_test, = load_image("famnist")
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(
    X_train, y_train, X_test, y_test, min_class, maj_class, imb_rate=imb_rate, val_frac=0.001)

# Change Python environment to TF environment
train_env = TFPyEnvironment(ClassifyEnv(X_train, y_train, imb_rate))

model = TrainCustomDDQN(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, target_update_tau=target_update_tau,
                        collect_steps_per_episode=collect_steps_per_episode, target_model_update=target_model_update,
                        n_step_update=n_step_update, batch_size=batch_size)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers, common.element_wise_huber_loss)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)  # , X_train, y_train)
print(rounded_dict(stats))
# {'Gmean': 0.943227, 'F1': 0.945219, 'Precision': 0.919584, 'Recall': 0.972324, 'TP': 2916, 'TN': 2745, 'FP': 255, 'FN': 83}
# {'Gmean': 0.911112, 'F1': 0.918146, 'Precision': 0.868093, 'Recall': 0.974325, 'TP': 2922, 'TN': 2556, 'FP': 444, 'FN': 77}
