"""Non-RL core example: two environments, three independent drones each."""
import torch
from flightlib_cuda import Core, ModelConfig, CommandMode

light, heavy = ModelConfig(), ModelConfig()
heavy.mass = 1.4
sim = Core(2, 3, models=[light, heavy], model_ids=[0, 1, 0, 1, 0, 1])
with sim.access():
    sim.state_input[2].fill_(5)
    sim.reset()
    sim.command_mode.fill_(CommandMode.COLLECTIVE_BODY_RATES)
    sim.command_payload.zero_()
    sim.command_payload[0].fill_(9.81)
sim.capture(commands=True)
for _ in range(100):
    sim.step(commands=True, graph=True)

# Borrowed device state can feed a controller without CPU conversion.
with sim.access():
    altitude_error = sim.position[2] - 5
    assert altitude_error.is_cuda

# Partial state assignment preserves the selected drone's running motors.
with sim.access():
    sim.state_selected.zero_()
    sim.state_selected[1] = 1
    sim.state_input[:, 1].copy_(sim.state[:, 1])
    sim.state_input[2, 1] = 6
    sim.time_input[1].copy_(sim.time[1])
    sim.set_state()

# Host readback happens only at this explicit boundary.
snapshot = sim.snapshot()
assert torch.isfinite(snapshot['state']).all()
assert snapshot['state'][2, 1] == 6
assert (snapshot['status'] == 0).all()
print('Core example passed; altitude:', snapshot['state'][2].tolist())
