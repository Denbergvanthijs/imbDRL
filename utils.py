from tf_agents.trajectories import trajectory


def compute_avg_return(environment, policy, num_episodes: int = 10):
    """Computes the average reward over a give number of episodes."""
    total_return = 0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    """Data collection for 1 step."""
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps: int):
    """Collect data for a number of steps. Mainly used for warmup period."""
    for _ in range(steps):
        collect_step(env, policy, buffer)
