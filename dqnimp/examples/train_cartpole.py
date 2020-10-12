from datetime import datetime

from dqnimp.train import TrainCartPole, TrainCustom
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment

# Code is based of https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial

episodes = 20_000  # Total episodes
warmup_episodes = 100  # Amount of warmup steps before training
memory_length = 100_000  # Max Replay Memory length

conv_layers = None
dense_layers = (100, )
dropout_layers = None

lr = 0.001  # Learning Rate
gamma = 1.0  # Discount factor
min_epsilon = 0.1  # Minimal chance of choosing random action
decay_episodes = 100  # Number of episodes to decay from 1.0 to `min_epsilon`
ddqn = True

model_dir = "./models/" + (NOW := datetime.now().strftime('%Y%m%d_%H%M%S'))
log_dir = "./logs/" + NOW

train_env = TFPyEnvironment(suite_gym.load('CartPole-v0'))  # Change Python environment to TF environment
val_env = TFPyEnvironment(suite_gym.load('CartPole-v0'))

model = TrainCartPole(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes, model_dir, log_dir)
model.compile(train_env, val_env, conv_layers, dense_layers, dropout_layers)
model.train()
model.evaluate()

q_net = TrainCustom.load_model(model_dir)
