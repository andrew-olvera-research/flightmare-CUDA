# Simulation interfaces

Use `flightlib_cuda.Core` to supply commands and step physics with your own controller or task. `Environment` adds the stock observation, reward, collision and reset logic. The optional `flightrl_cuda.TensorEnv` provides rollout-oriented views. The batch has one lane per environment/aircraft pair. Aircraft in one world can interact when peer collision checks are enabled.

## State and timing

Position is in metres in a right-handed Z-up frame. Attitude is a body-to-world WXYZ quaternion. Linear velocity is in world axes and angular rate is in body axes. Command either rotor thrust or collective thrust and body rates. Set model parameters before capture. Core holds persistent tensors and uses a fixed schedule of RK4 substeps within each control interval. [Core example](../flightlib-cuda/python/examples/core.py)

Outputs borrow device storage. Copy any tensor you want to retain before the next step. In autoreset mode the terminal transition observation and the following policy observation can refer to different poses. Run external graph capture and replay on the owning stream. Sample the terminal exposure before resetting the selected lanes. [Collector](../flightrl-cuda/flightrl_cuda/custom_collector.py) · [Graph and camera example](../tools/newton/README.md)

## Geometry and measurements

Course queries sweep a spherical vehicle proxy against shared box geometry. Peer checks sweep moving spheres within each world and expose candidate-buffer overflow. A hit can terminate an episode. The query reports contact during the interval and leaves integration at the end-of-step pose. Landing and impact response require a separate contact model. [Course rollout](../flightrl-cuda/flightrl_cuda/course_rollout.py)

IMU angular rate is in rad/s and specific force in m/s² in sensor axes. In level hover the aligned accelerometer reads +g on Z. Odometry supplies world position, sensor-to-world attitude and world velocity. The barometer supplies absolute pressure in Pa from local height plus the configured origin altitude. Configure mounts, rate, bias/noise and latency before stepping. Check packet validity, episode and acquisition time before using a measurement. Keep truth available separately for scoring. [Sensor example](../flightrl-cuda/flightrl_cuda/sensor_example.py) · [IMU calibration](../tools/sensors/qualify_imu.py) · [Pressure replay](../tools/sensors/validate_barometer.py)

The Newton pinhole camera renders visible surfaces independently of authored collision proxies. RGB and optional axial depth stay on device. Asset import supports a documented subset of Unity and OpenUSD geometry. [Supported assets](../tools/newton/README.md#using-authored-scenes)

## Build targets

The supported source build uses Linux x86-64, CUDA Toolkit 13.3, Python 3.12 and PyTorch 2.9.1+cu130. Compose targets the visible GPU. To embed native code for multiple devices set `FLIGHTMARE_CUDA_ARCHITECTURES=86-real;89-real;120-real` in the root `.env` before starting Compose. Those targets compiled for both native libraries; execution was qualified on SM 120 and driver 610.88. NVIDIA lists R580 as the floor for limited CUDA 13.x minor-version compatibility. Features using newer PTX can require a newer driver. [Driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)
