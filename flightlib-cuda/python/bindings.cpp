#include "flightlib_cuda/environment.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/dlpack.h>
#include <cuda_profiler_api.h>
namespace py=pybind11;
using namespace flightlib_cuda::v1;
namespace {
struct Export {
  Simulation owner;DLManagedTensor tensor{};std::int64_t shape[3]{},stride[3]{};
  explicit Export(Simulation s):owner(std::move(s)){}
};
py::capsule exportTensor(Simulation s,const std::string& name){
  for(const auto& b:s.allocationPlan().buffers)if(b.name==name){
    if(!b.bytes)throw std::invalid_argument("empty tensor export");
    auto* out=new Export(s);auto& t=out->tensor;
    t.manager_ctx=out;t.deleter=[](DLManagedTensor* p){delete static_cast<Export*>(p->manager_ctx);};
    t.dl_tensor.data=s.buffer(name);t.dl_tensor.device={kDLCUDA,s.device()};t.dl_tensor.ndim=b.spec.rank;
    t.dl_tensor.dtype={std::uint8_t(b.spec.dtype==DType::Float32?kDLFloat:kDLUInt),
      std::uint8_t(b.spec.dtype==DType::UInt8?8:b.spec.dtype==DType::UInt64?64:32),1};
    std::int64_t stride=1;for(int i=int(b.spec.rank)-1;i>=0;--i){out->shape[i]=b.spec.shape[i];out->stride[i]=stride;stride*=out->shape[i];}
    t.dl_tensor.shape=out->shape;t.dl_tensor.strides=out->stride;
    return py::capsule(&t,"dltensor",[](PyObject* capsule){
      if(PyCapsule_IsValid(capsule,"dltensor")){
        auto* t=static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule,"dltensor"));t->deleter(t);
      }
    });
  }
  throw std::invalid_argument("unknown tensor");
}
BindingRequest request(py::object tensor,int device,DType dtype){
  auto torch=py::module_::import("torch");
  if(!py::isinstance(tensor,torch.attr("Tensor"))||!tensor.attr("is_cuda").cast<bool>()||
     tensor.attr("device").attr("index").cast<int>()!=device||tensor.attr("requires_grad").cast<bool>())
    throw std::invalid_argument("input must be a CUDA tensor on the simulation device without requires_grad");
  std::string expected=dtype==DType::Float32?"torch.float32":"torch.uint8";
  if(py::str(tensor.attr("dtype")).cast<std::string>()!=expected||!tensor.attr("is_contiguous")().cast<bool>())
    throw std::invalid_argument("input dtype/layout mismatch; no implicit conversions");
  // A detached view retains the original storage when the caller replaces its tensor.
  tensor=tensor.attr("detach")();
  auto shape=tensor.attr("shape").cast<std::vector<Offset>>(),stride=tensor.attr("stride")().cast<std::vector<Offset>>();
  if(shape.empty()||shape.size()>3)throw std::invalid_argument("invalid rank");
  TensorSpec spec{dtype,Index(shape.size()),{}};std::array<Offset,3> strides{};
  std::copy(shape.begin(),shape.end(),spec.shape.begin());std::copy(stride.begin(),stride.end(),strides.begin());
  void* ptr=reinterpret_cast<void*>(tensor.attr("data_ptr")().cast<std::uintptr_t>());
  auto* retained=tensor.release().ptr();
  std::shared_ptr<void> owner(retained,[](void* p){py::gil_scoped_acquire gil;Py_DECREF(static_cast<PyObject*>(p));});
  return {{ptr,spec,strides},spec,owner};
}
struct NativeEnvironment {
  Simulation sim;Bindings bindings;Graph graph;bool bound=false,captured=false,captured_random=true;
  NativeEnvironment(Index E,Index A,float dt,int device,std::uint64_t seed,
                    const std::vector<ModelConfig>& models,const std::vector<Index>& ids,
                    const std::vector<Mask>& active,const std::vector<float>& bounds):sim([&]{
    RuntimeConfig c;c.environments=E;c.slots=A;c.models=models.size();c.substeps=prepareSchedule(dt).size();c.device=device;c.seed=seed;return c;}()){
    configure(sim,models,ids,active,bounds,dt);
    std::vector<TaskConfig> tasks;for(auto id:ids)tasks.push_back(legacyTaskConfig(models.at(id).mass));configureTask(sim,tasks);
  }
  void bind(py::object a,py::object selected,py::object x,py::object time){
    auto b=bindTaskInputs(sim,request(a,sim.device(),DType::Float32),request(selected,sim.device(),DType::UInt8),
                         request(x,sim.device(),DType::Float32),request(time,sim.device(),DType::Float32));
    graph=Graph{};captured=false;bindings=std::move(b);bound=true;
  }
  void stable(){
    if(!bound)throw std::logic_error("bind inputs first");
    for(auto& b:bindings.inputs()){
      auto owner=py::reinterpret_borrow<py::object>(static_cast<PyObject*>(b.lifetime.get()));
      if(owner.attr("data_ptr")().cast<std::uintptr_t>()!=reinterpret_cast<std::uintptr_t>(b.tensor.data))
        throw std::invalid_argument("bound tensor storage was resized; rebind and recapture");
    }
  }
  void reset(std::uintptr_t stream,bool random){stable();resetEnvironmentFrom(sim,bindings,reinterpret_cast<cudaStream_t>(stream),random?ResetSource::Random:ResetSource::Explicit);}
  void capture(bool random){stable();graph=captureEnvironment(sim,bindings,random?ResetSource::Random:ResetSource::Explicit);captured=true;captured_random=random;}
  void step(std::uintptr_t stream,bool replay,bool random){stable();auto s=reinterpret_cast<cudaStream_t>(stream);
    if(replay){if(!captured)throw std::logic_error("capture before graph replay");if(random!=captured_random)throw std::invalid_argument("reset mode differs from captured graph");sim.replayFrom(graph,s);}
    else stepEnvironmentFrom(sim,bindings,s,random?ResetSource::Random:ResetSource::Explicit);
  }
};
}
PYBIND11_MODULE(_native,m){
  m.def("profile_start",[]{if(cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("profiler start failed");});
  m.def("profile_stop",[]{if(cudaProfilerStop()!=cudaSuccess)throw std::runtime_error("profiler stop failed");});
  py::class_<ModelConfig>(m,"ModelConfig").def(py::init<>())
    .def_readwrite("mass",&ModelConfig::mass).def_readwrite("arm_length",&ModelConfig::arm_length)
    .def_readwrite("motor_min",&ModelConfig::motor_min).def_readwrite("motor_max",&ModelConfig::motor_max)
    .def_readwrite("motor_tau",&ModelConfig::motor_tau).def_readwrite("kappa",&ModelConfig::kappa)
    .def_readwrite("thrust_map",&ModelConfig::thrust_map).def_readwrite("rate_limit",&ModelConfig::rate_limit);
  py::class_<TaskConfig>(m,"TaskConfig").def(py::init<>())
    .def_readwrite("action_mean",&TaskConfig::action_mean).def_readwrite("action_scale",&TaskConfig::action_scale)
    .def_readwrite("goal",&TaskConfig::goal).def_readwrite("reward_coeff",&TaskConfig::reward_coeff);
  m.def("load_model",&loadModel);m.def("legacy_task_config",&legacyTaskConfig);
  py::class_<NativeEnvironment>(m,"NativeEnvironment")
    .def(py::init<Index,Index,float,int,std::uint64_t,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&>())
    .def_property_readonly("arena_bytes",[](NativeEnvironment& e){return e.sim.allocationPlan().arena_bytes;})
    .def("bind",&NativeEnvironment::bind).def("reset",&NativeEnvironment::reset)
    .def("capture",&NativeEnvironment::capture).def("step",&NativeEnvironment::step)
    .def("tensor",[](NativeEnvironment& e,const std::string& name){return exportTensor(e.sim,name);})
    .def("wait",[](NativeEnvironment& e,std::uintptr_t s){e.sim.waitOn(reinterpret_cast<cudaStream_t>(s));})
    .def("record_usage",[](NativeEnvironment& e,std::uintptr_t s){e.sim.recordUsage(reinterpret_cast<cudaStream_t>(s));})
    .def("synchronize",[](NativeEnvironment& e){e.sim.synchronize();})
    .def("configure_task",[](NativeEnvironment& e,const std::vector<TaskConfig>& c){configureTask(e.sim,c);})
    .def("configure",[](NativeEnvironment& e,const std::vector<ModelConfig>& m,const std::vector<Index>& ids,const std::vector<Mask>& active,const std::vector<float>& bounds,float dt){configure(e.sim,m,ids,active,bounds,dt);});
}
