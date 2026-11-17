#include "flightlib_cuda/environment.hpp"
#include "dynamics_launch.hpp"
#include <Eigen/Geometry>
#include <algorithm>

namespace flightlib_cuda::v1 {
namespace {
struct Inputs {const Scalar* action;const Mask* selected;const Scalar* x;const Scalar* time;};
__device__ void observation(const float* x,float* obs){
  Eigen::Quaternionf q(x[3],x[4],x[5],x[6]);
  // Eigen's ZYX convention. Ordinary lanes use float trig; near branch cuts
  // accurately rounded float intermediates avoid CUDA atan2f/sinf approximation
  // selecting the opposite +/-pi representation of the same orientation.
  Eigen::Matrix3f r=q.toRotationMatrix();float angles[3];
  angles[0]=atan2f(r(1,0),r(0,0));
  float c2=sqrtf(r(2,2)*r(2,2)+r(2,1)*r(2,1));
  if(angles[0]<0){angles[0]+=float(EIGEN_PI);c2=-c2;}
  angles[1]=atan2f(-r(2,0),c2);
  float s=sinf(angles[0]),c=cosf(angles[0]);
  angles[2]=atan2f(s*r(0,2)-c*r(1,2),c*r(1,1)-s*r(0,1));
  if(fabsf(angles[2])>float(EIGEN_PI)-1e-5f||fabsf(c2)<1e-5f){
    angles[0]=float(atan2(double(r(1,0)),double(r(0,0))));
    c2=fabsf(c2);if(angles[0]<0){angles[0]+=float(EIGEN_PI);c2=-c2;}
    angles[1]=float(atan2(double(-r(2,0)),double(c2)));
    s=float(sin(double(angles[0])));c=float(cos(double(angles[0])));
    angles[2]=float(atan2(double(s*r(0,2)-c*r(1,2)),double(c*r(1,1)-s*r(0,1))));
  }
  #pragma unroll
  for(int c=0;c<3;++c){obs[c]=x[c];obs[3+c]=angles[c];obs[6+c]=x[7+c];obs[9+c]=x[10+c];}
}
__device__ void clearOutputs(const TaskOutput& out,Index n){
  #pragma unroll
  for(int c=0;c<12;++c){out.observation.data[12ULL*n+c]=0;out.transition_observation.data[12ULL*n+c]=0;}
  out.reward.data[n]=0;out.done.data[n]=0;out.valid.data[n]=0;
}
__device__ std::uint64_t mix(std::uint64_t x){
  x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
__device__ float uniform(std::uint64_t key,std::uint64_t counter){
  return float(mix(key+0x9e3779b97f4a7c15ULL*(counter+1))>>40)*(2.0f/16777216.0f)-1.0f;
}
__global__ void seedKernel(RuntimeViews v,std::uint64_t seed){
  Index n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=v.topology.batch.lanes)return;
  v.episodes.random_key(n)=mix(seed+n+0x9e3779b97f4a7c15ULL);
  v.episodes.random_counter(n)=0;
}
// Reset validates all samples and the derived observation before committing.
__device__ bool resetLane(RuntimeViews v,Inputs in,Index n,ResetSource source){
  float x[25]={},obs[12],time=0;const Offset N=v.topology.batch.lanes;
  if(source==ResetSource::Explicit){
    if(in.selected[n]!=1){v.diagnostics.status(n)=LaneStatus::InvalidReset;return false;}
    time=in.time[n];bool valid=isfinite(time);
    #pragma unroll
    for(int c=0;c<25;++c){x[c]=in.x[c*N+n];valid &= isfinite(x[c]);}
    if(!valid){v.diagnostics.status(n)=LaneStatus::InvalidReset;return false;}
  }else{
    auto key=v.episodes.random_key(n),counter=v.episodes.random_counter(n);
    #pragma unroll
    for(int c=0;c<3;++c){x[c]=uniform(key,counter+c);x[7+c]=uniform(key,counter+3+c);}
    x[2]=fabsf(x[2]+5);float norm=0;
    #pragma unroll
    for(int c=0;c<4;++c){x[3+c]=uniform(key,counter+6+c);norm+=x[3+c]*x[3+c];}
    if(norm==0)x[3]=1;
    else {norm=sqrtf(norm);
      #pragma unroll
      for(int c=0;c<4;++c)x[3+c]/=norm;}
  }
  observation(x,obs);
  #pragma unroll
  for(int c=0;c<12;++c)if(!isfinite(obs[c])){v.diagnostics.status(n)=LaneStatus::InvalidReset;return false;}
  #pragma unroll
  for(int c=0;c<25;++c)v.state.x(c,n)=x[c];
  v.state.time(n)=time;
  #pragma unroll
  for(int c=0;c<4;++c){v.state.motor_omega(c,n)=0;v.state.motor_thrust(c,n)=0;v.commands.payload(c,n)=0;}
  v.commands.mode(n)=CommandMode::RotorThrust;v.commands.time(n)=0;
  v.episodes.command_time(n)=0;v.episodes.step(n)=0;++v.episodes.episode(n);
  if(source==ResetSource::Random)v.episodes.random_counter(n)+=10;
  #pragma unroll
  for(int c=0;c<12;++c)v.task_output.observation.data[12ULL*n+c]=obs[c];
  v.diagnostics.status(n)=LaneStatus::Ok;return true;
}
__global__ void prepareActions(RuntimeViews v,Inputs in){
  Index n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=v.topology.batch.lanes)return;
  v.task_output.valid.data[n]=0;
  if(!v.topology.active(n)){v.diagnostics.status(n)=LaneStatus::Inactive;return;}
  float command[4],time=v.episodes.command_time(n)+v.schedule.control_dt;
  bool valid=isfinite(time);Index p=v.topology.model_id(n);
  if(p>=v.models.mass.size()){v.diagnostics.status(n)=LaneStatus::InvalidCommand;return;}
  #pragma unroll
  for(int c=0;c<4;++c){float raw=in.action[4ULL*n+c];
    command[c]=raw*v.task_parameters.action_scale(c,n)+v.task_parameters.action_mean(c,n);
    valid &= isfinite(raw)&&isfinite(command[c]);}
  // Validate physical input before changing the independent command clock.
  if(!valid){v.diagnostics.status(n)=LaneStatus::InvalidCommand;return;}
  if(!isfinite(v.state.time(n))){v.diagnostics.status(n)=LaneStatus::InvalidState;return;}
  #pragma unroll
  for(int c=0;c<25;++c)if(!isfinite(v.state.x(c,n))){v.diagnostics.status(n)=LaneStatus::InvalidState;return;}
  #pragma unroll
  for(int c=0;c<4;++c)v.commands.payload(c,n)=fminf(v.models.thrust_bounds(1,p),fmaxf(v.models.thrust_bounds(0,p),command[c]));
  v.commands.mode(n)=CommandMode::RotorThrust;v.commands.time(n)=time;
  v.episodes.command_time(n)=time;v.task_output.valid.data[n]=1;
}
template<bool InlineReset, bool TerminalReward=true>
__global__ void taskKernel(RuntimeViews v,Inputs in,ResetSource source){
  Index n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=v.topology.batch.lanes)return;
  if(v.diagnostics.status(n)!=LaneStatus::Ok){clearOutputs(v.task_output,n);return;}
  float x[13],obs[12];
  #pragma unroll
  for(int c=0;c<13;++c)x[c]=v.state.x(c,n);
  observation(x,obs);float reward=0,action_norm=0;
  #pragma unroll
  for(int group=0;group<4;++group){float norm=0;
    #pragma unroll
    for(int c=0;c<3;++c){int i=group*3+c;float delta=obs[i]-v.task_parameters.goal(i,n);norm+=delta*delta;}
    reward+=v.task_parameters.reward_coeff(group,n)*norm;
    float raw=in.action[4ULL*n+group];action_norm+=raw*raw;
  }
  reward+=v.task_parameters.reward_coeff(4,n)*sqrtf(action_norm);reward+=.1f;
  bool terminal=x[2]<=.02f;if(TerminalReward&&terminal)reward+=-.02f;
  bool valid=isfinite(reward);
  #pragma unroll
  for(int c=0;c<12;++c)valid &= isfinite(obs[c]);
  if(!valid){v.diagnostics.status(n)=LaneStatus::NumericalFailure;clearOutputs(v.task_output,n);return;}
  #pragma unroll
  for(int c=0;c<12;++c){
    // Terminal lanes in the fused path materialize only the returned reset obs.
    if(!InlineReset||!terminal)v.task_output.observation.data[12ULL*n+c]=obs[c];
    v.task_output.transition_observation.data[12ULL*n+c]=obs[c];
  }
  v.task_output.reward.data[n]=reward;v.task_output.done.data[n]=terminal;v.task_output.valid.data[n]=1;++v.episodes.step(n);
  if constexpr(InlineReset)if(terminal&&!resetLane(v,in,n,source))clearOutputs(v.task_output,n);
}
__global__ void resetKernel(RuntimeViews v,Inputs in,ResetSource source,bool automatic){
  Index n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=v.topology.batch.lanes)return;
  if(!v.topology.active(n)){clearOutputs(v.task_output,n);v.diagnostics.status(n)=LaneStatus::Inactive;return;}
  if(automatic){if(!v.task_output.valid.data[n]||!v.task_output.done.data[n])return;}
  else {v.diagnostics.status(n)=LaneStatus::Ok;if(in.selected[n]==0)return;
    if(in.selected[n]!=1){v.diagnostics.status(n)=LaneStatus::InvalidReset;clearOutputs(v.task_output,n);return;}
    clearOutputs(v.task_output,n);}
  if(!resetLane(v,in,n,source))clearOutputs(v.task_output,n);
}
TensorSpec spec(DType type,std::initializer_list<Offset> shape){TensorSpec s{type,Index(shape.size()),{}};std::copy(shape.begin(),shape.end(),s.shape.begin());return s;}
std::array<TensorSpec,4> schemas(const Simulation& s){auto b=s.views().topology.batch;return {
  spec(DType::Float32,{b.environments,b.slots_per_environment,4}),spec(DType::UInt8,{b.lanes}),
  spec(DType::Float32,{25,b.lanes}),spec(DType::Float32,{b.lanes})};}
void validate(const Simulation& s,const Bindings& b){auto expected=schemas(s);
  if(b.inputs().size()!=4)throw std::invalid_argument("wrong task input count");
  for(int i=0;i<4;++i){auto actual=b.inputs()[i].tensor.spec;if(actual.dtype!=expected[i].dtype||actual.rank!=expected[i].rank||actual.shape!=expected[i].shape)throw std::invalid_argument("wrong task input schema");}}
Inputs pointers(const std::vector<BindingRequest>& b){return {static_cast<const Scalar*>(b[0].tensor.data),static_cast<const Mask*>(b[1].tensor.data),static_cast<const Scalar*>(b[2].tensor.data),static_cast<const Scalar*>(b[3].tensor.data)};}
void validateSource(ResetSource source){if(source!=ResetSource::Explicit&&source!=ResetSource::Random)throw std::invalid_argument("unknown reset source");}
}
TaskConfig legacyTaskConfig(Scalar mass){if(!std::isfinite(mass)||mass<=0)throw std::invalid_argument("invalid task mass");TaskConfig c;c.action_mean.fill(mass*9.81f/4);c.action_scale.fill(mass*2*9.81f/4);return c;}
void seedEnvironment(Simulation& s,std::uint64_t seed){
  s.enqueueSetup([](const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>&,void* ptr){
    seedKernel<<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v,*static_cast<std::uint64_t*>(ptr));
  },&seed);
}
void configureTask(Simulation& s,const std::vector<TaskConfig>& configs){
  Offset N=s.views().topology.batch.lanes;if(configs.size()!=1&&configs.size()!=N)throw std::invalid_argument("wrong task parameter count");
  std::vector<float> mean(4*N),scale(4*N),goal(12*N),reward(5*N);
  for(Offset n=0;n<N;++n){auto& c=configs[configs.size()==1?0:n];
    for(int i=0;i<12;++i){if(!std::isfinite(c.goal[i]))throw std::invalid_argument("nonfinite task goal");goal[i*N+n]=c.goal[i];}
    for(int i=0;i<5;++i){if(!std::isfinite(c.reward_coeff[i]))throw std::invalid_argument("nonfinite task reward");reward[i*N+n]=c.reward_coeff[i];}
    for(int i=0;i<4;++i){if(!std::isfinite(c.action_mean[i])||!std::isfinite(c.action_scale[i]))throw std::invalid_argument("nonfinite task scaling");mean[i*N+n]=c.action_mean[i];scale[i*N+n]=c.action_scale[i];}}
  struct Copy {void* dst;const float* src;size_t bytes;};std::array<Copy,4> copies{{
    {s.buffer("task_parameters.action_mean"),mean.data(),mean.size()*4},{s.buffer("task_parameters.action_scale"),scale.data(),scale.size()*4},
    {s.buffer("task_parameters.goal"),goal.data(),goal.size()*4},{s.buffer("task_parameters.reward_coeff"),reward.data(),reward.size()*4}}};
  s.enqueueSetup([](const RuntimeViews&,cudaStream_t stream,const std::vector<BindingRequest>&,void* ctx){
    auto& copies=*static_cast<std::array<Copy,4>*>(ctx);try{
      for(auto c:copies)if(cudaMemcpyAsync(c.dst,c.src,c.bytes,cudaMemcpyHostToDevice,stream)!=cudaSuccess)throw std::runtime_error("task setup copy failed");
      if(cudaStreamSynchronize(stream)!=cudaSuccess)throw std::runtime_error("task setup completion failed");
    }catch(...){cudaStreamSynchronize(stream);throw;}
  },&copies);
}
Bindings bindTaskInputs(Simulation& s,BindingRequest action,BindingRequest selected,BindingRequest x,BindingRequest time){
  auto shapes=schemas(s);std::vector<BindingRequest> inputs{action,selected,x,time};for(int i=0;i<4;++i)inputs[i].expected=shapes[i];return s.bind(std::move(inputs));}
namespace {
struct EnvironmentContext {ResetSource source;TaskExecution execution;};
void environmentLaunch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>& b,void* ptr){
  auto ctx=*static_cast<EnvironmentContext*>(ptr);auto in=pointers(b);unsigned blocks=(Offset{v.topology.batch.lanes}+127)/128;
  if(ctx.execution==TaskExecution::FirstLaneReset){
    prepareActions<<<1,1,0,stream>>>(v,in);detail::launchDynamicsFirst(v,stream,v.task_output.valid.data);
    taskKernel<true><<<1,1,0,stream>>>(v,in,ctx.source);return;
  }
  prepareActions<<<blocks,128,0,stream>>>(v,in);detail::launchDynamics(v,stream,v.task_output.valid.data);
  if(ctx.execution==TaskExecution::FusedReset)taskKernel<true><<<blocks,128,0,stream>>>(v,in,ctx.source);
  else if(ctx.execution==TaskExecution::NoReset)taskKernel<false,false><<<blocks,128,0,stream>>>(v,in,ctx.source);
  else {taskKernel<false><<<blocks,128,0,stream>>>(v,in,ctx.source);resetKernel<<<blocks,128,0,stream>>>(v,in,ctx.source,true);}
}
void resetLaunch(const RuntimeViews& v,cudaStream_t stream,const std::vector<BindingRequest>& b,void* ctx){
  resetKernel<<<(Offset{v.topology.batch.lanes}+127)/128,128,0,stream>>>(v,pointers(b),*static_cast<ResetSource*>(ctx),false);
}
void validateExecution(TaskExecution execution){if(execution!=TaskExecution::FusedReset&&execution!=TaskExecution::SeparateReset&&execution!=TaskExecution::NoReset&&execution!=TaskExecution::FirstLaneReset)throw std::invalid_argument("unknown task execution");}
}
void resetEnvironment(Simulation& s,const Bindings& b,ResetSource source){validate(s,b);validateSource(source);s.enqueue(resetLaunch,&source,&b);}
void resetEnvironmentFrom(Simulation& s,const Bindings& b,cudaStream_t stream,ResetSource source){validate(s,b);validateSource(source);s.enqueueFrom(stream,resetLaunch,&source,&b);}
void stepEnvironment(Simulation& s,const Bindings& b,ResetSource source,TaskExecution execution){
  validate(s,b);validateSource(source);validateExecution(execution);EnvironmentContext ctx{source,execution};s.enqueue(environmentLaunch,&ctx,&b);
}
void stepEnvironmentFrom(Simulation& s,const Bindings& b,cudaStream_t stream,ResetSource source,TaskExecution execution){
  validate(s,b);validateSource(source);validateExecution(execution);EnvironmentContext ctx{source,execution};s.enqueueFrom(stream,environmentLaunch,&ctx,&b);
}
Graph captureEnvironment(Simulation& s,const Bindings& b,ResetSource source,TaskExecution execution){
  validate(s,b);validateSource(source);validateExecution(execution);EnvironmentContext ctx{source,execution};return s.capture(environmentLaunch,&ctx,&b);
}
}
