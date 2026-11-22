#include "flightlib_cuda/environment.hpp"
#include "flightlib_cuda/operations.hpp"
#include "flightlib_cuda/transport.hpp"
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
  std::string expected=dtype==DType::Float32?"torch.float32":dtype==DType::UInt32?"torch.uint32":"torch.uint8";
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
void stableAllocations(const std::vector<BindingRequest>& inputs){
  for(auto& b:inputs){
    auto owner=py::reinterpret_borrow<py::object>(static_cast<PyObject*>(b.lifetime.get()));
    if(owner.attr("data_ptr")().cast<std::uintptr_t>()!=reinterpret_cast<std::uintptr_t>(b.tensor.data))
      throw std::invalid_argument("bound tensor storage was resized; rebind and recapture");
  }
}
void stableBindings(const Bindings& bindings){stableAllocations(bindings.inputs());}
struct NativeCore {
  Simulation sim;Bindings state_input,command_input;Graph graph;
  bool state_bound=false,command_bound=false,captured=false,captured_commands=false;
  std::vector<ModelConfig> models;
  NativeCore(Index E,Index A,float dt,int device,const std::vector<ModelConfig>& m,
             const std::vector<Index>& ids,const std::vector<Mask>& active,const std::vector<float>& bounds):sim([&]{
    RuntimeConfig c;c.environments=E;c.slots=A;c.models=m.size();c.substeps=prepareSchedule(dt).size();c.device=device;return c;}()){
    setup(m,ids,active,bounds,dt);
  }
  void setup(const std::vector<ModelConfig>& m,const std::vector<Index>& ids,const std::vector<Mask>& active,const std::vector<float>& bounds,float dt){
    auto next=m;configure(sim,m,ids,active,bounds,dt);models.swap(next);
  }
  void bindState(py::object selected,py::object x,py::object time){
    state_input=bindStateInput(sim,request(selected,sim.device(),DType::UInt8),request(x,sim.device(),DType::Float32),request(time,sim.device(),DType::Float32));
    state_bound=true;
  }
  void bindCommand(py::object selected,py::object mode,py::object time,py::object payload){
    auto b=bindCommandInput(sim,request(selected,sim.device(),DType::UInt8),request(mode,sim.device(),DType::UInt8),
                            request(time,sim.device(),DType::Float32),request(payload,sim.device(),DType::Float32));
    command_input=std::move(b);command_bound=true;graph=Graph{};captured=false;
  }
  void checkCommands(){if(!command_bound)throw std::logic_error("bind commands first");stableBindings(command_input);}
  void state(std::uintptr_t stream,bool reset){
    if(!state_bound)throw std::logic_error("bind state input first");stableBindings(state_input);
    if(reset)resetStateFrom(sim,state_input,reinterpret_cast<cudaStream_t>(stream));
    else setStateFrom(sim,state_input,reinterpret_cast<cudaStream_t>(stream));
  }
  void command(std::uintptr_t stream){checkCommands();setCommandFrom(sim,command_input,reinterpret_cast<cudaStream_t>(stream));}
  void step(std::uintptr_t stream,bool commands,bool replay){
    if(commands)checkCommands();auto s=reinterpret_cast<cudaStream_t>(stream);
    if(replay){if(!captured)throw std::logic_error("capture before graph replay");
      if(commands!=captured_commands)throw std::invalid_argument("command mode differs from captured graph");sim.replayFrom(graph,s);}
    else if(commands)runDynamicsFrom(sim,command_input,s);else stepDynamicsFrom(sim,s);
  }
  void capture(bool commands){if(commands)checkCommands();graph=commands?captureRun(sim,command_input):captureDynamics(sim);captured=true;captured_commands=commands;}
};
struct NativeOperation {
  Simulation sim;CoreOperation op;Bindings bindings;
  NativeOperation(NativeCore& core,CoreOperation operation,py::object input,py::object output):sim(core.sim),op(operation),
    bindings(bindOperation(sim,op,request(input,sim.device(),DType::Float32),request(output,sim.device(),DType::Float32))){}
  void run(std::uintptr_t stream,float dt){stableBindings(bindings);evaluateOperationFrom(sim,op,bindings,reinterpret_cast<cudaStream_t>(stream),dt);}
};
template<std::size_t N>
void components(py::dict d,const char* key,float (&out)[N]){
  auto x=d[key].cast<std::array<float,N>>();std::copy(x.begin(),x.end(),out);
}
SensorSettings sensorSettings(py::dict d){
  SensorSettings s{};
  if(d.empty())return s;
  s.period_ns=d["period_ns"].cast<std::uint64_t>();s.phase_ns=d["phase_ns"].cast<std::uint64_t>();
  s.latency_ns=d["latency_ns"].cast<std::uint64_t>();s.seed=d["seed"].cast<std::uint64_t>();
  s.dropout=d["dropout"].cast<float>();components(d,"mount",s.mount);components(d,"offset",s.offset);
  components(d,"noise",s.noise);components(d,"bias",s.bias);components(d,"walk",s.walk);
  components(d,"limit",s.limit);components(d,"attitude_std",s.attitude_std);
  if(d.contains("sea_level_pa"))s.sea_level_pa=d["sea_level_pa"].cast<float>();
  if(d.contains("origin_altitude_m"))s.origin_altitude_m=d["origin_altitude_m"].cast<float>();
  components(d,"increment_width",s.increment_width);
  return s;
}
struct NativeEnvironment {
  Simulation sim;Bindings bindings;Graph graph;bool bound=false,captured=false,captured_random=true;
  Graph transition_graph;
  TaskExecution execution=TaskExecution::FusedReset;
  void autoReset(bool enabled){execution=enabled?TaskExecution::FusedReset:TaskExecution::NoReset;graph=Graph{};captured=false;}
  void testStep(std::uintptr_t stream,bool random){stable();
    stepEnvironmentFrom(sim,bindings,reinterpret_cast<cudaStream_t>(stream),random?ResetSource::Random:ResetSource::Explicit,TaskExecution::FirstLaneReset);
  }
  NativeEnvironment(Index E,Index A,float dt,int device,std::uint64_t seed,
                    const std::vector<ModelConfig>& models,const std::vector<Index>& ids,
                    const std::vector<Mask>& active,const std::vector<float>& bounds,Index capacity=0,float radius=0,bool terminate=false,
                    py::dict imu={},py::dict odometry={},py::dict barometer={},py::dict network={}):sim([&]{
    auto schedule=prepareSchedule(dt);
    RuntimeConfig c;c.environments=E;c.slots=A;c.models=models.size();c.substeps=schedule.size();c.device=device;c.seed=seed;
    c.imu=sensorSettings(imu);c.odometry=sensorSettings(odometry);
    c.barometer=sensorSettings(barometer);
    if(!network.empty()){
      auto period=std::llround(double(dt)*1e9);
      if(period<1 || period>UINT32_MAX)throw std::invalid_argument("unsupported network tick");
      c.network_tick_ns=period;
      c.carrier_group_size=network["carrier_group_size"].cast<Index>();
      c.hidden_per_edge=network["hidden_per_edge"].cast<Index>();
      c.network_fanin=network["fanin"].cast<Index>();
      if(!network["range_edges_m"].is_none()){
        c.network_sample_loss=true;
        c.network_range_edges=network["range_edges_m"].cast<std::vector<Scalar>>();
      }
      c.network_offsets=network["offsets_ns"].cast<std::vector<Index>>();
      if(!c.network_fanin)throw std::invalid_argument("network requires positive receiver fanin");
      c.shared_profile={network["slot_ns"].cast<Index>(),network["window_slots"].cast<Index>(),
        network["ns_per_byte"].cast<Index>(),network["preamble_ns"].cast<Index>(),
        network["recovery_ns"].cast<Index>(),network["seed"].cast<Index>()};
    }
    validateSensorPeriods(schedule,c.imu,c.odometry,c.barometer);
    if(c.imu.period_ns||c.odometry.period_ns||c.barometer.period_ns)c.sensor_step_ns=std::llround(double(dt)*1e9);
    c.candidates_per_environment=capacity;c.agent_collision_radius=radius;c.terminate_on_agent_collision=terminate;return c;}()){
    configure(sim,models,ids,active,bounds,dt);
    std::vector<TaskConfig> tasks;for(auto id:ids)tasks.push_back(legacyTaskConfig(models.at(id).mass));configureTask(sim,tasks);
  }
  void bind(py::object a,py::object selected,py::object x,py::object time,py::tuple scene){
    std::vector<BindingRequest> collision;
    if(scene.size()){
      if(scene.size()!=2)throw std::invalid_argument("expected scene nodes and options");
      for(auto tensor:scene)collision.push_back(request(py::reinterpret_borrow<py::object>(tensor),
        sim.device(),DType::Float32));
    }
    auto b=bindTaskInputs(sim,request(a,sim.device(),DType::Float32),request(selected,sim.device(),DType::UInt8),
                         request(x,sim.device(),DType::Float32),request(time,sim.device(),DType::Float32),std::move(collision));
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
  void capture(bool random,bool poses=false){stable();auto source=random?ResetSource::Random:ResetSource::Explicit;
    captured=false;graph=captureEnvironment(sim,bindings,source,execution);
    transition_graph=poses && execution==TaskExecution::FusedReset?
      captureEnvironment(sim,bindings,source,TaskExecution::DeferredReset):Graph{};
    captured=true;captured_random=random;}
  void step(std::uintptr_t stream,bool replay,bool random,bool poses=false){stable();auto s=reinterpret_cast<cudaStream_t>(stream);
    bool defer=poses && execution==TaskExecution::FusedReset;
    if(replay){if(!captured)throw std::logic_error("capture before graph replay");if(random!=captured_random)throw std::invalid_argument("reset mode differs from captured graph");sim.replayFrom(defer?transition_graph:graph,s);}
    else stepEnvironmentFrom(sim,bindings,s,random?ResetSource::Random:ResetSource::Explicit,defer?TaskExecution::DeferredReset:execution);
  }
  void recordStep(std::uintptr_t stream,bool random,py::object before_reset){
    bool defer=!before_reset.is_none() && execution==TaskExecution::FusedReset;
    stable();recordEnvironmentStep(sim,bindings,reinterpret_cast<cudaStream_t>(stream),
      random?ResetSource::Random:ResetSource::Explicit,defer?TaskExecution::DeferredReset:execution);
    if(!before_reset.is_none())before_reset();
    if(defer)recordEnvironmentFinish(sim,bindings,reinterpret_cast<cudaStream_t>(stream),
      random?ResetSource::Random:ResetSource::Explicit);
  }
  void recordReset(std::uintptr_t stream,bool random){
    stable();recordEnvironmentReset(sim,bindings,reinterpret_cast<cudaStream_t>(stream),
      random?ResetSource::Random:ResetSource::Explicit);
  }
  void finish(std::uintptr_t stream,bool random){if(execution==TaskExecution::FusedReset)
    finishResetFrom(sim,bindings,reinterpret_cast<cudaStream_t>(stream),random?ResetSource::Random:ResetSource::Explicit);}
};
struct NativeExport {
  Simulation sim; ExportKind kind; Bindings bindings;
  std::vector<BindingRequest> owners;
  py::object ready;
  cudaEvent_t event;
  bool submitted=false;
  NativeExport(NativeEnvironment& env,ExportKind k,py::object lanes,py::object rows,py::object count):
    sim(env.sim),kind(k),owners{request(lanes,sim.device(),DType::UInt32),
      request(rows,sim.device(),DType::UInt8),request(count,sim.device(),DType::UInt32)} {
    // The slot owns this registration and can disconnect before the runtime dies.
    auto borrowed=owners;
    for(auto& r:borrowed)r.lifetime.reset();
    bindings=bindExport(sim,k,borrowed[0],borrowed[1],borrowed[2]);
    auto cuda=py::module_::import("torch.cuda");
    ready=cuda.attr("Event")();
    ready.attr("record")(cuda.attr("current_stream")(sim.device()));
    event=reinterpret_cast<cudaEvent_t>(ready.attr("cuda_event").cast<std::uintptr_t>());
  }
  void pack(std::uintptr_t stream){
    stableAllocations(owners);
    try {
      packExportFrom(sim,bindings,kind,reinterpret_cast<cudaStream_t>(stream),event);
      submitted=true;
    } catch(...) {
      // A failed CUDA submission may have enqueued work before reporting the error.
      try { sim.synchronize(); } catch(...) {}
      throw;
    }
  }
  ~NativeExport(){if(submitted)cudaEventSynchronize(event);}
};
}
PYBIND11_MODULE(_native,m){
  using flightlib_cuda::packet::Frame;
  m.attr("frame_offsets")=py::make_tuple(offsetof(Frame,id),offsetof(Frame,receivers),offsetof(Frame,erased),
      offsetof(Frame,receiver_base),offsetof(Frame,bytes),offsetof(Frame,all_peers),offsetof(Frame,attempt_id),offsetof(Frame,handle),offsetof(Frame,expiry_ns),offsetof(Frame,fixture_label));
  using flightlib_cuda::packet::Reception;
  m.attr("completion_offsets")=py::make_tuple(offsetof(Reception,admission),offsetof(Reception,completed),
      offsetof(Reception,collided),offsetof(Reception,expired),offsetof(Reception,handle),offsetof(Reception,fixture_label),
        offsetof(Reception,submitted),offsetof(Reception,air_start),offsetof(Reception,air_end));
  m.def("profile_start",[]{if(cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("profiler start failed");});
  m.def("profile_stop",[]{if(cudaProfilerStop()!=cudaSuccess)throw std::runtime_error("profiler stop failed");});
  py::class_<ModelConfig>(m,"ModelConfig").def(py::init<>())
    .def_readwrite("mass",&ModelConfig::mass).def_readwrite("arm_length",&ModelConfig::arm_length)
    .def_readwrite("motor_min",&ModelConfig::motor_min).def_readwrite("motor_max",&ModelConfig::motor_max)
    .def_readwrite("motor_tau",&ModelConfig::motor_tau).def_readwrite("kappa",&ModelConfig::kappa)
    .def_readwrite("thrust_map",&ModelConfig::thrust_map).def_readwrite("rate_limit",&ModelConfig::rate_limit)
    .def_readwrite("rate_gain",&ModelConfig::rate_gain);
  py::class_<TaskConfig>(m,"TaskConfig").def(py::init<>())
    .def_readwrite("ctbr",&TaskConfig::ctbr)
    .def_readwrite("action_mean",&TaskConfig::action_mean).def_readwrite("action_scale",&TaskConfig::action_scale)
    .def_readwrite("goal",&TaskConfig::goal).def_readwrite("reward_coeff",&TaskConfig::reward_coeff);
  m.def("load_model",&loadModel);m.def("legacy_task_config",&legacyTaskConfig);
  py::enum_<CoreOperation>(m,"CoreOperation")
    .value("DERIVATIVE",CoreOperation::Derivative).value("CONTROLLER",CoreOperation::Controller)
    .value("MOTORS",CoreOperation::Motors).value("OMEGA_TO_THRUST",CoreOperation::OmegaToThrust)
    .value("THRUST_TO_OMEGA",CoreOperation::ThrustToOmega).value("CLAMP_THRUST",CoreOperation::ClampThrust)
    .value("CLAMP_MOTOR_OMEGA",CoreOperation::ClampMotorOmega).value("CLAMP_BODY_RATES",CoreOperation::ClampBodyRates)
    .value("CONSTRAIN_WORLD",CoreOperation::ConstrainWorld);
  py::class_<NativeOperation>(m,"NativeOperation")
    .def(py::init<NativeCore&,CoreOperation,py::object,py::object>()).def("run",&NativeOperation::run);
  m.def("prepare_schedule",[](float dt,float max_dt){return prepareSchedule(dt,max_dt);},py::arg("control_dt"),py::arg("max_dt")=.0025f);
  py::class_<NativeCore>(m,"NativeCore")
    .def(py::init<Index,Index,float,int,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&>())
    .def_property_readonly("arena_bytes",[](NativeCore& c){return c.sim.allocationPlan().arena_bytes;})
    .def_property_readonly("model_configs",[](NativeCore& c){return c.models;})
    .def("bind_state",&NativeCore::bindState).def("bind_command",&NativeCore::bindCommand)
    .def("state",&NativeCore::state).def("set_command",&NativeCore::command)
    .def("step",&NativeCore::step).def("capture",&NativeCore::capture).def("configure",&NativeCore::setup)
    .def("tensor",[](NativeCore& c,const std::string& name){return exportTensor(c.sim,name);})
    .def("wait",[](NativeCore& c,std::uintptr_t s){c.sim.waitOn(reinterpret_cast<cudaStream_t>(s));})
    .def("record_usage",[](NativeCore& c,std::uintptr_t s){c.sim.recordUsage(reinterpret_cast<cudaStream_t>(s));})
    .def("synchronize",[](NativeCore& c){c.sim.synchronize();});
  py::class_<NativeEnvironment>(m,"NativeEnvironment")
    .def(py::init<Index,Index,float,int,std::uint64_t,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&>())
    .def(py::init<Index,Index,float,int,std::uint64_t,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&,Index,float,bool>())
    .def(py::init<Index,Index,float,int,std::uint64_t,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&,Index,float,bool,py::dict,py::dict>())
    .def(py::init<Index,Index,float,int,std::uint64_t,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&,Index,float,bool,py::dict,py::dict,py::dict>())
    .def(py::init<Index,Index,float,int,std::uint64_t,const std::vector<ModelConfig>&,const std::vector<Index>&,const std::vector<Mask>&,const std::vector<float>&,Index,float,bool,py::dict,py::dict,py::dict,py::dict>())
    .def_property_readonly("arena_bytes",[](NativeEnvironment& e){return e.sim.allocationPlan().arena_bytes;})
    .def("bind",&NativeEnvironment::bind,py::arg("action"),py::arg("selected"),py::arg("state"),py::arg("time"),py::arg("scene")=py::tuple()).def("reset",&NativeEnvironment::reset).def("auto_reset",&NativeEnvironment::autoReset)
    .def("test_step",&NativeEnvironment::testStep)
    .def("seed",[](NativeEnvironment& e,std::uint64_t seed){seedEnvironment(e.sim,seed);})
    .def("capture",&NativeEnvironment::capture,py::arg("random"),py::arg("poses")=false)
    .def("step",&NativeEnvironment::step,py::arg("stream"),py::arg("replay"),py::arg("random"),py::arg("poses")=false)
    .def("record_step",&NativeEnvironment::recordStep,py::arg("stream"),py::arg("random"),py::arg("before_reset")=py::none()).def("record_reset",&NativeEnvironment::recordReset)
    .def("finish_transition",&NativeEnvironment::finish)
    .def("tensor",[](NativeEnvironment& e,const std::string& name){return exportTensor(e.sim,name);})
    .def("wait",[](NativeEnvironment& e,std::uintptr_t s){e.sim.waitOn(reinterpret_cast<cudaStream_t>(s));})
    .def("record_usage",[](NativeEnvironment& e,std::uintptr_t s){e.sim.recordUsage(reinterpret_cast<cudaStream_t>(s));})
    .def("synchronize",[](NativeEnvironment& e){e.sim.synchronize();})
    .def("configure_task",[](NativeEnvironment& e,const std::vector<TaskConfig>& c){configureTask(e.sim,c);})
    .def("configure",[](NativeEnvironment& e,const std::vector<ModelConfig>& m,const std::vector<Index>& ids,const std::vector<Mask>& active,const std::vector<float>& bounds,float dt){configure(e.sim,m,ids,active,bounds,dt);});
  py::enum_<ExportKind>(m,"ExportKind").value("POSE",ExportKind::Pose).value("IMU",ExportKind::Imu).value("BAROMETER",ExportKind::Barometer).value("ODOMETRY",ExportKind::Odometry);
  py::class_<NativeExport>(m,"NativeExport").def(py::init<NativeEnvironment&,ExportKind,py::object,py::object,py::object>())
    .def_readonly("ready",&NativeExport::ready).def("pack",&NativeExport::pack);
}
