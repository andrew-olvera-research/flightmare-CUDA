import numpy as np

from flightgym import QuadrotorEnv_v1


def test_headless_reset_and_step():
    env = QuadrotorEnv_v1(
        "{env: {seed: 1, scene_id: 0, num_envs: 3, num_threads: 1, render: false}}",
        False,
    )
    count = env.getNumOfEnvs()
    assert (count, env.getObsDim(), env.getActDim()) == (3, 12, 4)

    obs = np.full((count, 12), np.nan, dtype=np.float32)
    action = np.zeros((count, 4), dtype=np.float32)
    reward = np.full(count, np.nan, dtype=np.float32)
    done = np.ones(count, dtype=np.bool_)
    extra = np.empty((count, len(env.getExtraInfoNames())), dtype=np.float32)

    try:
        assert env.reset(obs)
        assert np.isfinite(obs).all()
        initial_obs = obs.copy()
        assert env.step(action, obs, reward, done, extra)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward).all()
        assert not done.any()
        assert np.any(obs != initial_obs)
    finally:
        env.close()
