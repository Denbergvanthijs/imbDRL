import tensorflow as tf
from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.examples.bandit.example_classes import TrainCustomBandit
from imbDRL.utils import get_reward_distribution, rounded_dict
from tf_agents.bandits.environments.classification_environment import \
    ClassificationBanditEnvironment

training_loops = 2_000  # Total training loops
batch_size = 32  # Batch size of each step
steps_per_loop = 32  # Number of steps to take in the environment for each loop

conv_layers = None   # Convolutional layers
dense_layers = (256, 256, )  # Dense layers
dropout_layers = (0.2, 0.2, )  # Dropout layers

lr = 0.001  # Learning Rate
min_epsilon = 0.01  # Minimal chance of choosing random action
decay_steps = training_loops // 3  # Number of episodes to decay from 1.0 to `min_epsilon`

imb_rate = 0.001729  # Imbalance rate
min_class = [1]  # Minority classes
maj_class = [0]  # Majority classes
X_train, y_train, X_test, y_test, = load_creditcard(normalization=True)
X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(X_train, y_train, X_test, y_test, min_class, maj_class, val_frac=0.2)

reward_distr = get_reward_distribution(imb_rate)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_env = ClassificationBanditEnvironment(train_ds, reward_distr, batch_size)
model = TrainCustomBandit(training_loops, lr, min_epsilon, decay_steps, batch_size=batch_size, steps_per_loop=steps_per_loop)

model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(X_val, y_val)
stats = model.evaluate(X_test, y_test, X_train, y_train)
print(rounded_dict(stats))
# ("Gmean", 0.930651) ("Fdot5", 0.557743) ("F1", 0.643939) ("F2", 0.761649) ("TP", 85) ("TN", 56783) ("FP",  81) ("FN", 13)
# ("Gmean", 0.93917 ) ("Fdot5", 0.227749) ("F1", 0.315789) ("F2", 0.514793) ("TP", 87) ("TN", 56498) ("FP", 366) ("FN", 11)
# ("Gmean", 0.950929) ("Fdot5", 0.311189) ("F1", 0.412993) ("F2", 0.613793) ("TP", 89) ("TN", 56620) ("FP", 244) ("FN",  9)
