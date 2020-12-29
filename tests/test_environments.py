import numpy as np
from imbDRL.environments.classifierenv import ClassifierEnv
from tf_agents.environments.utils import validate_py_environment


def test_ClassifierEnv():
    """Tests imbDRL.environments.classifierenv.ClassifierEnv."""
    X = np.arange(10, dtype=np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)

    env = ClassifierEnv(X, y, 0.2)
    validate_py_environment(env, episodes=5)


def test_reset():
    """Tests imbDRL.environments.classifierenv.ClassifierEnv._reset."""
    X = np.arange(10, dtype=np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)

    env = ClassifierEnv(X, y, 0.2)
    env.reset()
    env.step([1])
    ts_restart = env.reset()

    assert env.episode_step == 0
    assert not env._episode_ended
    assert ts_restart.step_type == 0
    assert ts_restart.reward == 0
    assert ts_restart.discount == 1
    assert ts_restart.observation in X  # Next observation is any of the values in X since its shuffled each reset


def test_step():
    """Tests imbDRL.environments.ClassifyEnv._step."""
    X = np.arange(10, dtype=np.float32)
    y = np.ones(10, dtype=np.int32)  # All labels are positive

    env = ClassifierEnv(X, y, 0.2)
    env.reset()
    time_step = env.step([1])  # True Positive
    assert time_step.reward == 1
    time_step = env.step([0])  # False Negative
    assert time_step.reward == -1

    time_step = env.step([1])
    assert time_step.step_type == 0  # Reset since last step was False Negative

    X = np.arange(10, dtype=np.float32)
    y = np.zeros(10, dtype=np.int32)  # All labels are negative

    env = ClassifierEnv(X, y, 0.2)
    env.reset()
    time_step = env.step([0])  # True Negative
    assert time_step.reward == np.array([0.2], dtype=np.float32)
    time_step = env.step([1])  # False Positive
    assert time_step.reward == np.array([-0.2], dtype=np.float32)

    env.reset()
    for _ in range(X.size):
        time_step = env.step([0])  # Take random step
    assert time_step.step_type == 0  # Reset since last step end of dataset
