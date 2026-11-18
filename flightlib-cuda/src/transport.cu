#include "flightlib_cuda/transport.hpp"
#include <limits>

namespace flightlib_cuda::v1 {
namespace {
struct Pose {
  std::uint64_t episode;
  Index lane, valid;
  float values[7];
  Index padding;
};
template<int C> struct Measurement {
  std::uint64_t metadata[6];
  Index lane, valid;
  float values[C];
};
static_assert(sizeof(Pose)==48 && sizeof(Measurement<6>)==80 && sizeof(Measurement<10>)==96);

__global__ void poses(RuntimeViews v,const Index* lanes,Index size,Pose* rows,Index* count){
  const auto i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=size)return;
  if(i==0)*count=size;
  const Index lane=lanes[i];
  Pose row{};
  row.lane=lane;
  if(lane<v.topology.batch.lanes){
    row.episode=v.episodes.episode(lane);
    row.valid=v.task_output.valid.data[lane];
    if(row.valid){
      double norm2=0;
      for(int c=3;c<7;++c){double q=v.state.x(c,lane);norm2+=q*q;}
      row.valid=norm2>0 && isfinite(norm2);
      if(row.valid){
        for(int c=0;c<3;++c)row.values[c]=v.state.x(c,lane);
        double inv=rsqrt(norm2);
        for(int c=3;c<7;++c)row.values[c]=float(v.state.x(c,lane)*inv);
      }
    }
  }
  rows[i]=row;
}
template<int C>
__global__ void measurements(SensorBuffers b,const Index* lanes,Index size,
                             Measurement<C>* rows,Index* count){
  const auto i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=size)return;
  const Index lane=lanes[i];
  if(lane>=b.values.shape[1])return;
  Index due=0;
  while(due<b.values.shape[0] && b.metadata(0,due,lane))++due;
  if(!due)return;
  const auto first=atomicAdd(count,due);
  for(Index s=0;s<due;++s){
    auto& r=rows[first+s];
    r.lane=lane;r.valid=b.valid(s,lane);
    for(int f=0;f<6;++f)r.metadata[f]=b.metadata(f,s,lane);
    for(int c=0;c<C;++c)r.values[c]=b.values(s,lane,c);
  }
}
TensorSpec rowsSpec(const RuntimeViews& v,ExportKind kind,Offset actors){
  Offset slots=1,width=sizeof(Pose);
  if(kind==ExportKind::Imu){slots=v.imu.values.shape[0];width=sizeof(Measurement<6>);}
  else if(kind==ExportKind::Odometry){slots=v.odometry.values.shape[0];width=sizeof(Measurement<10>);}
  else if(kind!=ExportKind::Pose)throw std::invalid_argument("invalid export kind");
  if(!actors || actors>v.topology.batch.lanes || !slots ||
     slots>std::numeric_limits<Index>::max()/actors)
    throw std::invalid_argument("invalid export capacity or disabled sensor");
  return {DType::UInt8,2,{actors*slots,width,0}};
}
}
Bindings bindExport(Simulation& sim,ExportKind kind,BindingRequest lanes,
                    BindingRequest rows,BindingRequest count){
  const Offset n=lanes.tensor.spec.shape[0];
  lanes.expected={DType::UInt32,1,{n,0,0}};
  rows.expected=rowsSpec(sim.views(),kind,n);
  count.expected={DType::UInt32,1,{1,0,0}};
  if(reinterpret_cast<std::uintptr_t>(rows.tensor.data)%8)
    throw std::invalid_argument("unaligned export records");
  return sim.bind({lanes,rows,count});
}
void packExportFrom(Simulation& sim,const Bindings& b,ExportKind kind,cudaStream_t stream,cudaEvent_t ready){
  const auto& inputs=b.inputs();
  if(inputs.size()!=3)throw std::invalid_argument("invalid export bindings");
  auto n=inputs[0].tensor.spec.shape[0];
  auto rows=rowsSpec(sim.views(),kind,n);
  if(inputs[0].tensor.spec.dtype!=DType::UInt32 || inputs[0].tensor.spec.rank!=1 ||
     inputs[1].tensor.spec.dtype!=rows.dtype || inputs[1].tensor.spec.rank!=rows.rank ||
     inputs[1].tensor.spec.shape!=rows.shape ||
     reinterpret_cast<std::uintptr_t>(inputs[1].tensor.data)%8 ||
     inputs[2].tensor.spec.dtype!=DType::UInt32 || inputs[2].tensor.spec.rank!=1 ||
     inputs[2].tensor.spec.shape[0]!=1)throw std::invalid_argument("wrong export schema");
  struct Capture { ExportKind kind; cudaEvent_t ready; } capture{kind,ready};
  sim.enqueueFrom(stream,[](const RuntimeViews& v,cudaStream_t st,const std::vector<BindingRequest>& a,void* arg){
    auto capture=*static_cast<Capture*>(arg);
    auto lanes=static_cast<const Index*>(a[0].tensor.data);
    auto size=static_cast<Index>(a[0].tensor.spec.shape[0]);
    auto blocks=(Offset{size}+127)/128;
    auto out=a[1].tensor.data;
    auto count=static_cast<Index*>(a[2].tensor.data);
    if(capture.kind==ExportKind::Pose){
      poses<<<blocks,128,0,st>>>(v,lanes,size,static_cast<Pose*>(out),count);
    }else{
      if(cudaMemsetAsync(count,0,sizeof(Index),st)!=cudaSuccess)throw std::runtime_error("clear export count");
      if(capture.kind==ExportKind::Imu)
        measurements<6><<<blocks,128,0,st>>>(v.imu,lanes,size,static_cast<Measurement<6>*>(out),count);
      else
        measurements<10><<<blocks,128,0,st>>>(v.odometry,lanes,size,static_cast<Measurement<10>*>(out),count);
    }
    if(capture.ready && cudaEventRecord(capture.ready,st)!=cudaSuccess)
      throw std::runtime_error("record export completion");
  },&capture,&b);
}
}
