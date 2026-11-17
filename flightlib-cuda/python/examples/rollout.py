"""Minimal resident policy loop. Copies to history are explicit device-to-device work."""
import torch
from flightlib_cuda import Environment

env = Environment(1024, seed=7)
env.reset(random=True)
env.capture(random=True)
weights = torch.full((12, 4), .001, device=env.device)
history = torch.empty((100, 1024, 1, 12), device=env.device)
with torch.no_grad():
    for t in range(100):
        torch.mm(env.outputs['observation'].view(1024, 12), weights, out=env.actions.view(1024, 4))
        env.step(graph=True, random=True)
        history[t].copy_(env.outputs['transition_observation'])
    env.record_usage()  # retain arena until the last external history copy completes
env.synchronize()
print('Completed resident rollout:', history.shape)
