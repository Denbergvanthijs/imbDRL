from imbDRL.examples.ddqn.example_classes import TrainCartPole
from tf_agents.environments import suite_gym
from tf_agents.environments.tf_py_environment import TFPyEnvironment

episodes = 20_000  # Total number of episodes
warmup_episodes = 1_000  # Amount of warmup steps to collect data with random policy
memory_length = 100_000  # Max length of the Replay Memory

conv_layers = None
dense_layers = (128, )  # For CartPole v0, only a small network is needed.
dropout_layers = None

lr = 0.001  # Learning rate
gamma = 0.99  # Discount factor
min_epsilon = 0.1  # Minimal and final chance of choosing random action
decay_episodes = 100  # Number of episodes to decay from 1.0 to `min_epsilon`

# Change OpenAI Gym environment to Python environment to TF environment
train_env = TFPyEnvironment(suite_gym.load("CartPole-v0"))
val_env = TFPyEnvironment(suite_gym.load("CartPole-v0"))

model = TrainCartPole(episodes, warmup_episodes, lr, gamma, min_epsilon, decay_episodes)
model.compile_model(train_env, conv_layers, dense_layers, dropout_layers)
model.train(val_env, 10)  # Validate for 10 episodes each time
model.evaluate(val_env, 10)  # TODO: add video as evaluation
