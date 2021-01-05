from datetime import datetime

import numpy as np
import pytest
from imbDRL.agents.ddqn import TrainDDQN


def test_TrainDDQN(tmp_path):
    """Tests imbDRL.agents.ddqn.TrainDDQN."""
    tmp_models = str(tmp_path / "test_model")  # No support for pathLib https://github.com/tensorflow/tensorflow/issues/37357
    tmp_logs = str(tmp_path / "test_log")

    model = TrainDDQN(10, 10, 0.001, 0.0, 0.1, 5, model_path=tmp_models, log_dir=tmp_logs)
    assert model.model_path == tmp_models
    assert not model.compiled

    NOW = datetime.now().strftime("%Y%m%d")  # yyyymmdd
    model = TrainDDQN(10, 10, 0.001, 0.0, 0.1, 5)
    assert "./models/" + NOW in model.model_path  # yyyymmdd in yyyymmdd_hhmmss


def test_compile_model(tmp_path):
    """Tests imbDRL.agents.ddqn.TrainDDQN.compile_model."""
    tmp_models = str(tmp_path / "test_model")  # No support for pathLib https://github.com/tensorflow/tensorflow/issues/37357
    tmp_logs = str(tmp_path / "test_log")

    model = TrainDDQN(10, 10, 0.001, 0.0, 0.1, 5, model_path=tmp_models, log_dir=tmp_logs)

    with pytest.raises(TypeError) as exc:
        model.compile_model(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32), 0.5, 128, None, None)
    assert "must be tuple or None" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        model.compile_model(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32), 0.5, None, 128, None)
    assert "must be tuple or None" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        model.compile_model(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32), 0.5, None, None, 128)
    assert "must be tuple or None" in str(exc.value)

    model = TrainDDQN(10, 10, 0.001, 0.0, 0.1, 5, model_path=tmp_models, log_dir=tmp_logs)
    assert not model.compiled
    model.compile_model(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32), None, (128, ), None)
    assert model.compiled


def test_train(tmp_path):
    """Tests imbDRL.agents.ddqn.TrainDDQN.train."""
    tmp_models = str(tmp_path / "test_model")  # No support for pathLib https://github.com/tensorflow/tensorflow/issues/37357
    tmp_logs = str(tmp_path / "test_log")

    model = TrainDDQN(10, 10, 0.001, 0.0, 0.1, 5, model_path=tmp_models, log_dir=tmp_logs, val_every=2, memory_length=30)

    with pytest.raises(Exception) as exc:
        model.train()
    assert "must be compiled" in str(exc.value)

    model.compile_model(np.array([1., 1.], dtype=np.float32), np.array([1., 1.], dtype=np.float32), None, (128,), None)
    model.train(np.empty(10, dtype=np.float32), np.random.choice(2, size=10).astype(np.int32))
    assert model.replay_buffer.num_frames().numpy() >= 10 + 10  # 10 for warmup + 1 for each episode
    assert model.global_episode == 10

    model = TrainDDQN(10, 10, 0.001, 0.0, 0.1, 5, model_path=tmp_models, log_dir=tmp_logs, val_every=2)

    with pytest.raises(Exception) as exc:
        model.train()
    assert "must be compiled" in str(exc.value)

    model.compile_model(np.array([1., 1.], dtype=np.float32), np.array([1., 1.], dtype=np.float32), None, (128,), None)
    model.train(np.empty(10, dtype=np.float32), np.random.choice(2, size=10).astype(np.int32))
    assert model.replay_buffer.num_frames() >= 10  # 10 in total since no memory length is defined
    assert model.global_episode == 10
