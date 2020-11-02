from datetime import datetime

import tensorflow as tf
from imbDRL.data import get_train_test_val, load_imdb
from imbDRL.train.bandit import TrainCustomBandit
from imbDRL.utils import get_reward_distribution, rounded_dict
from tf_agents.bandits.environments.classification_environment import \
    ClassificationBanditEnvironment

training_loops = 1_000  # Total training loops
batch_size = 64  # Batch size of each step
steps_per_loop = 64  # Number of steps to take in the environment for each loop

conv_layers = None  # Convolutional layers
dense_layers = (256, 256)  # Dense layers
dropout_layers = (0.0, 0.3)  # Dropout layers

lr = 0.001  # Learning rate
min_epsilon = 0.01  # Minimal and final chance of choosing random action
decay_steps = 250  # Number of steps to decay from 1.0 to `min_epsilon`

model_dir = "./models/" + (NOW := datetime.now().strftime("%Y%m%d_%H%M%S"))
log_dir = "./logs/" + NOW

imb_rate = 0.01  # Imbalance rate
min_class = [0]  # Minority classes
maj_class = [1]  # Majority classes
X_train, y_train, X_test, y_test, = load_imdb()
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

reward_distr = get_reward_distribution(imb_rate)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_env = ClassificationBanditEnvironment(train_ds, reward_distr, batch_size)

model = TrainCustomBandit(training_loops, lr, min_epsilon, decay_steps, model_dir,
                          log_dir, batch_size=batch_size, steps_per_loop=steps_per_loop)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(rounded_dict(stats))
# ("Gmean", 0.28783) ("Fdot5", 0.109092) ("F1", 0.162604) ("F2", 0.319161) ("TP", 1114) ("TN", 1162) ("FP", 11338) ("FN", 136)
