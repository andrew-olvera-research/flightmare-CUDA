"""Small executable PPO training/evaluation example, with explicit host logging."""
import argparse
import json
import torch
from . import Config, TensorEnv, ActorCritic, Rollout, ppo_update, save, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', type=int, default=1024)
    parser.add_argument('--slots', type=int, default=1, help='independent agents per environment')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checkpoint', default='policy.pt')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    config = Config(num_envs=args.envs, slots=args.slots, seed=args.seed)
    env = TensorEnv(config); env.reset(); env.capture()
    if args.evaluate:
        policy, _, data = load(args.checkpoint)
        # Evaluation must use the saved physics/task configuration.
        expected = config.metadata(); actual = data['config']
        for key in ('model','control_dt','reward_coeff'):
            if expected[key] != actual[key]: raise ValueError('evaluation configuration mismatch: '+key)
        history = Rollout(args.steps, env, policy); history.noise.zero_()
        history.collect(resample=False)
        print(json.dumps(dict(mean_reward=history.reward.mean().item(), valid=bool(history.valid.all()))))
    else:
        policy = ActorCritic().to(env.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        history = Rollout(args.steps, env, policy)
        for update in range(args.updates):
            history.collect(); history.finish()
            metrics = ppo_update(policy, optimizer, history)
            if not bool(torch.isfinite(metrics).all()):
                raise RuntimeError('nonfinite training update')
            print(json.dumps(dict(update=update+1, losses=metrics.tolist(), mean_reward=history.reward.mean().item())))
        save(args.checkpoint, policy, optimizer, config, args.updates)
    env.close()


if __name__ == '__main__': main()
