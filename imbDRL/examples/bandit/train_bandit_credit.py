from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.train.bandit import TrainCustomBandit
from tf_agents.bandits.environments.classification_environment import \
    ClassificationBanditEnvironment

training_loops = 2_000  # Total training loops
batch_size = 32  # Batch size of each step
steps_per_loop = 32  # Number of steps to take in the environment for each loop

conv_layers = None   # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.0, 0.2, )  # Dropout layers

lr = 0.001  # Learning Rate
min_epsilon = 0.01  # Minimal chance of choosing random action
decay_steps = 1_000  # Number of episodes to decay from 1.0 to `min_epsilon`
val_every = training_loops // 50  # Validate 50 times during training

model_dir = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
log_dir = "./logs/" + NOW

imb_rate = 0.00173  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, imb_rate, min_class, maj_class)

bernoulli = tfp.distributions.Bernoulli(probs=[[imb_rate, -imb_rate], [-1, 1]], dtype=tf.float32)
reward_distr = (tfp.bijectors.Shift([[imb_rate, -imb_rate], [-1, 1]])
                (tfp.bijectors.Scale([[1, 1], [1, 1]])
                 (bernoulli)))
distr = tfp.distributions.Independent(reward_distr, reinterpreted_batch_ndims=2)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_env = ClassificationBanditEnvironment(train_ds, distr, batch_size)

model = TrainCustomBandit(training_loops, lr, min_epsilon, decay_steps, model_dir, log_dir,
                          batch_size=batch_size, steps_per_loop=steps_per_loop, val_every=val_every)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test)
print(*[(k, round(v, 6)) for k, v in stats.items()])
# ('Gmean', 0.908017) ('Fdot5', 0.412424) ('F1', 0.507837) ('F2', 0.660685) ('TP', 81) ('TN', 56724) ('FP', 140) ('FN', 17)
# ('Gmean', 0.908322) ('Fdot5', 0.487952) ('F1', 0.576512) ('F2', 0.704348) ('TP', 81) ('TN', 56762) ('FP', 102) ('FN', 17)  # 8
# ('Gmean', 0.908354) ('Fdot5', 0.497543) ('F1', 0.584838) ('F2', 0.709282) ('TP', 81) ('TN', 56766) ('FP',  98) ('FN', 17)  # 32
