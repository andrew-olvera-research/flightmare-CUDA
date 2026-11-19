#include "flightlib_cuda/lifecycle.hpp"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <algorithm>

namespace flightlib_cuda::v1 {
ModelConfig loadModel(const std::string& path) {
  auto node=YAML::LoadFile(path)["quadrotor_dynamics"];
  if(!node) throw std::invalid_argument("missing quadrotor_dynamics");
  ModelConfig c;
  c.mass=node["mass"].as<Scalar>(); c.arm_length=node["arm_l"].as<Scalar>();
  c.motor_min=node["motor_omega_min"].as<Scalar>(); c.motor_max=node["motor_omega_max"].as<Scalar>();
  c.motor_tau=node["motor_tau"].as<Scalar>(); c.kappa=node["kappa"].as<Scalar>();
  auto map=node["thrust_map"].as<std::vector<Scalar>>();
  auto rates=node["omega_max"].as<std::vector<Scalar>>();
  if(map.size()!=3 || rates.size()!=3) throw std::invalid_argument("invalid model vector length");
  std::copy(map.begin(),map.end(),c.thrust_map.begin());
  std::copy(rates.begin(),rates.end(),c.rate_limit.begin());
  if(node["rate_gain"]) c.rate_gain=node["rate_gain"].as<std::array<Scalar,3>>();
  prepareModels({c}); return c;
}
ModelTable prepareModels(const std::vector<ModelConfig>& configs) {
  if(configs.empty()) throw std::invalid_argument("empty model table");
  ModelTable table;
  auto put=[&](const char* name, Index row, Index column, Index rows, Scalar value) {
    if(!std::isfinite(value)) throw std::invalid_argument("nonfinite derived model parameter");
    auto& out=table[std::string("models.")+name]; out.resize(rows*configs.size());
    out[row*configs.size()+column]=value;
  };
  for(Index p=0;p<configs.size();++p) {
    const auto& c=configs[p];
    if(!(c.mass>0 && c.mass<100 && c.arm_length>0 && c.motor_min>=0 &&
         c.motor_max>c.motor_min && c.motor_tau>0 && c.kappa>0))
      throw std::invalid_argument("invalid model parameters");
    for(auto v:{c.mass,c.arm_length,c.motor_min,c.motor_max,c.motor_tau,c.kappa})
      if(!std::isfinite(v)) throw std::invalid_argument("nonfinite model parameter");
    for(auto v:c.rate_limit) if(!(v>0) || !std::isfinite(v)) throw std::invalid_argument("invalid rate limit");
    for(auto gain:c.rate_gain) if(!std::isfinite(gain) || gain<0) throw std::invalid_argument("invalid rate gain");
    // Flightlib computes the scalar factor in double before the Eigen float
    // multiplication. Keep its rounding boundary for randomized models too.
    Eigen::Vector3f j=c.mass/12.0*c.arm_length*c.arm_length*Eigen::Vector3f(4.5f,4.5f,7);
    Eigen::Matrix3f inertia=j.asDiagonal();
    Eigen::Matrix3f inverse=inertia.inverse();
    const Scalar arm=c.arm_length*std::sqrt(0.5);
    // This library may share a process with an oracle/consumer built using
    // different Eigen SIMD flags. Keep this private type distinct from its
    // aligned Matrix4f: weak inline inverse symbols must not coalesce across
    // incompatible stack-alignment assumptions (SSE vs AVX-512).
    using AllocationMatrix = Eigen::Matrix<float,4,4,Eigen::DontAlign>;
    AllocationMatrix allocation;
    allocation << 1,1,1,1, arm,-arm,-arm,arm, -arm,-arm,arm,arm,c.kappa,-c.kappa,c.kappa,-c.kappa;
    AllocationMatrix inv=allocation.inverse();
    put("mass",0,p,1,c.mass); put("motor_tau_inverse",0,p,1,1.0f/c.motor_tau);
    put("motor_speed_bounds",0,p,2,c.motor_min);put("motor_speed_bounds",1,p,2,c.motor_max);
    // updateParams never refreshes the constructor's cached thrust cap.
    const ModelConfig initial;
    const Scalar cap=2000.0f*2000.0f*initial.thrust_map[0]+2000.0f*initial.thrust_map[1]+initial.thrust_map[2];
    put("thrust_bounds",0,p,2,0); put("thrust_bounds",1,p,2,cap);
    for(Index k=0;k<3;++k) {
      put("inertia",k,p,3,j[k]); put("inverse_inertia",k,p,3,inverse(k,k));
      put("thrust_polynomial",k,p,3,c.thrust_map[k]); put("body_rate_limit",k,p,3,c.rate_limit[k]);
      put("rate_gain",k,p,3,c.rate_gain[k]);put("gravity",k,p,3,k==2?-9.81f:0);
    }
    for(Index r=0;r<4;++r) for(Index col=0;col<4;++col) {
      put("allocation",r*4+col,p,16,allocation(r,col));put("inverse_allocation",r*4+col,p,16,inv(r,col));
    }
  }
  return table;
}
std::vector<Scalar> prepareSchedule(Scalar dt,Scalar max_dt,Index capacity) {
  if(!std::isfinite(dt)||dt<0||!std::isfinite(max_dt)||max_dt<=0)
    throw std::invalid_argument("invalid timestep");
  std::vector<Scalar> result;
  while(dt>0) {
    if(result.size()>=capacity) throw std::length_error("substep capacity exceeded");
    const Scalar step=std::min(dt,max_dt), next=dt-step;
    if(next==dt) throw std::invalid_argument("float substep schedule cannot progress");
    result.push_back(step);dt=next;
  }
  return result;
}
void validateSensorPeriods(const std::vector<Scalar>& schedule,
                           const SensorSettings& imu, const SensorSettings& odometry, const SensorSettings& barometer) {
  if (!imu.period_ns && !odometry.period_ns && !barometer.period_ns) return;
  if (schedule.empty()) throw std::invalid_argument("sensors require positive simulation time");
  auto longest = std::uint64_t(std::llround(double(*std::max_element(schedule.begin(), schedule.end())) * 1e9));
  for (auto period : {imu.period_ns, odometry.period_ns, barometer.period_ns})
    if (period && period < longest)
      throw std::invalid_argument("sensor period is shorter than a physics substep");
}
TypedCommand adaptCommand(const LegacyCommand& c) {
  const bool rates=std::isfinite(c.collective)&&std::all_of(c.rates.begin(),c.rates.end(),[](Scalar x){return std::isfinite(x);});
  const bool rotors=std::all_of(c.thrusts.begin(),c.thrusts.end(),[](Scalar x){return std::isfinite(x);});
  if(!std::isfinite(c.time)||rates==rotors) return {};
  return {rotors?CommandMode::RotorThrust:CommandMode::CollectiveBodyRates,c.time,
          rotors?c.thrusts:std::array<Scalar,4>{c.collective,c.rates[0],c.rates[1],c.rates[2]}};
}
void configure(Simulation& sim,const std::vector<ModelConfig>& models,const std::vector<Index>& ids,
               const std::vector<Mask>& active,const std::vector<Scalar>& bounds,Scalar dt,Scalar max_dt) {
  const auto& v=sim.views();const Offset n=v.topology.batch.lanes;
  if(models.size()!=v.models.mass.size()||ids.size()!=n||active.size()!=n||bounds.size()!=6ULL*n)
    throw std::invalid_argument("configuration shape mismatch");
  for(Index i=0;i<n;++i) {
    if(ids[i]>=models.size()||active[i]>1) throw std::invalid_argument("invalid topology");
    for(Index axis=0;axis<3;++axis) {
      const auto lo=bounds[axis*2*n+i], hi=bounds[(axis*2+1)*n+i];
      if(!std::isfinite(lo)||!std::isfinite(hi)||!(lo<hi)) throw std::invalid_argument("invalid world box");
    }
  }
  auto table=prepareModels(models);auto schedule=prepareSchedule(dt,max_dt, v.schedule.substep_dt.size());
  if(v.sensor_truth.samples.data){
    if(std::llround(double(dt)*1e9)!=v.sensor_step_ns)
      throw std::invalid_argument("reconstruct sensors to change the control period");
    validateSensorPeriods(schedule, v.imu.settings, v.odometry.settings, v.barometer.settings);
  }
  if(schedule.size()!=v.schedule.substep_dt.size()) throw std::invalid_argument("schedule length differs from frozen allocation");
  struct Upload {void* dst;const void* src;size_t bytes;};
  struct Context {std::vector<Upload> copies;Scalar dt;}; Context context{{},dt};
  for(const auto& entry:table) context.copies.push_back({sim.buffer(entry.first),entry.second.data(),entry.second.size()*4});
  context.copies.push_back({sim.buffer("topology.model_id"),ids.data(),ids.size()*sizeof(Index)});
  context.copies.push_back({sim.buffer("topology.active"),active.data(),active.size()});
  context.copies.push_back({sim.buffer("topology.world_bounds"),bounds.data(),bounds.size()*4});
  if(!schedule.empty()) context.copies.push_back({sim.buffer("schedule.substep_dt"),schedule.data(),schedule.size()*4});
  sim.enqueueSetup([](const RuntimeViews& views,cudaStream_t stream,const std::vector<BindingRequest>&,void* ptr){
    auto& c=*static_cast<Context*>(ptr);
    try {
      for(const auto& copy:c.copies) if(cudaMemcpyAsync(copy.dst,copy.src,copy.bytes,cudaMemcpyHostToDevice,stream)!=cudaSuccess)
        throw std::runtime_error("configuration upload failed");
      // Setup is deliberately synchronous under the submission lock: another
      // host submitter cannot race source teardown or partially uploaded tables.
      if(cudaStreamSynchronize(stream)!=cudaSuccess)throw std::runtime_error("configuration completion failed");
    } catch(...) {cudaStreamSynchronize(stream);throw;}
    const_cast<RuntimeViews&>(views).schedule.control_dt=c.dt;
  },&context);
}
} // namespace flightlib_cuda::v1
