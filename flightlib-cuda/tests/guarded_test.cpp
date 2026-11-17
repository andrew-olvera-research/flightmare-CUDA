// Detect writes across fields inside the single arena, which allocation-level
// memcheck cannot distinguish. This complements, not replaces, device sanitizers.
#include "flightlib_cuda/environment.hpp"
#include "flightlib_cuda/operations.hpp"
#include <algorithm>
#include <iostream>
#include <tuple>

using namespace flightlib_cuda::v1;
#define CHECK(x) do { if (!(x)) throw std::runtime_error("line " + std::to_string(__LINE__) + ": " #x); } while (false)
#define CUDA(x) CHECK((x) == cudaSuccess)
constexpr size_t guard = 256;
constexpr unsigned char marker = 0xa5;

struct Guards {
  struct Gap { unsigned char* pointer; size_t bytes; std::string name; };
  std::vector<Gap> gaps;
  void add(void* p, size_t bytes, const std::string& name) {
    CHECK(bytes >= guard);
    gaps.push_back({static_cast<unsigned char*>(p), bytes, name});
    CUDA(cudaMemset(p, marker, bytes));
  }
  void check() const {
    for (const auto& gap : gaps) {
      std::vector<unsigned char> data(gap.bytes);
      CUDA(cudaMemcpy(data.data(), gap.pointer, gap.bytes, cudaMemcpyDeviceToHost));
      if (!std::all_of(data.begin(), data.end(), [](auto c) { return c == marker; }))
        throw std::runtime_error("guard overwritten: " + gap.name);
    }
  }
  void arena(Simulation& sim) {
    sim.synchronize();
    const auto& plan = sim.allocationPlan();
    const auto& first = plan.buffers.front();
    auto* base = static_cast<unsigned char*>(sim.buffer(first.name)) - first.offset;
    Offset end = 0;
    for (const auto& b : plan.buffers) if (b.bytes) {
      add(base + end, b.offset - end, "before " + b.name);
      end = b.offset + b.bytes;
    }
    add(base + end, plan.arena_bytes - end, "arena end");
  }
};

template<class T> BindingRequest input(const std::vector<T>& data, DType type,
                                     std::initializer_list<Offset> shape, Guards& guards) {
  void* base = nullptr;
  const auto bytes = data.size() * sizeof(T);
  CUDA(cudaMalloc(&base, bytes + 2 * guard));
  auto owner = std::shared_ptr<void>(base, [](void* p) { cudaFree(p); });
  auto* p = static_cast<unsigned char*>(base) + guard;
  guards.add(base, guard, "external prefix");
  guards.add(p + bytes, guard, "external suffix");
  CUDA(cudaMemcpy(p, data.data(), bytes, cudaMemcpyHostToDevice));
  TensorSpec spec{type, static_cast<Index>(shape.size()), {}};
  std::copy(shape.begin(), shape.end(), spec.shape.begin());
  std::array<Offset, 3> strides{};
  Offset stride = 1;
  for (Index i = spec.rank; i-- > 0;) { strides[i] = stride; stride *= spec.shape[i]; }
  return {{p, spec, strides}, spec, owner};
}
template<class T> std::vector<T> download(const T* p, size_t count) {
  std::vector<T> result(count);
  CUDA(cudaMemcpy(result.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost));
  return result;
}

void exercise(Index E, Index A) {
  const Offset N = Offset{E} * A;
  RuntimeConfig config;
  config.environments = E; config.slots = A; config.substeps = prepareSchedule(.01f).size();
  config.debug_guards = true;
  Simulation sim(config);
  Guards guards;
  guards.arena(sim);
  // Negative control: prove the checker observes a write at a field boundary.
  CUDA(cudaMemset(guards.gaps.front().pointer, 0, 1));
  bool detected = false;
  try { guards.check(); } catch (const std::runtime_error&) { detected = true; }
  CHECK(detected);
  CUDA(cudaMemset(guards.gaps.front().pointer, marker, 1));
  guards.check();

  std::vector<Mask> active(N, 1), selected(N, 1);
  if (N > 1) active.back() = 0;
  std::vector<float> bounds(6*N), state(25*N, 0), time(N, 0), action(4*N, -1), command(4*N, 0);
  for (Offset n = 0; n < N; ++n) {
    bounds[n] = bounds[2*N+n] = -20;
    bounds[N+n] = bounds[3*N+n] = bounds[5*N+n] = 20;
    state[3*N+n] = 1;
    state[2*N+n] = n % 2 ? 5 : 0; // exercise terminal and nonterminal lanes
    command[n] = 9.81f;
  }
  configure(sim, {ModelConfig{}}, std::vector<Index>(N, 0), active, bounds, .01f);
  configureTask(sim, {TaskConfig{}});
  auto si = input(selected, DType::UInt8, {N}, guards);
  auto xi = input(state, DType::Float32, {25,N}, guards);
  auto ti = input(time, DType::Float32, {N}, guards);
  auto ai = input(action, DType::Float32, {E,A,4}, guards);
  auto mi = input(std::vector<CommandMode>(N, CommandMode::CollectiveBodyRates), DType::UInt8, {N}, guards);
  auto ci = input(command, DType::Float32, {4,N}, guards);
  auto state_inputs = bindStateInput(sim, si, xi, ti);
  auto command_inputs = bindCommandInput(sim, si, mi, ti, ci);
  auto task_inputs = bindTaskInputs(sim, ai, si, xi, ti);
  auto check = [&] { sim.synchronize(); guards.check(); };
  resetState(sim, state_inputs); setState(sim, state_inputs); check();
  setCommand(sim, command_inputs); stepDynamics(sim); check();
  runDynamics(sim, command_inputs); check();
  { auto graph = captureRun(sim, command_inputs); sim.replay(graph); check(); }
  for (auto [op, nr, no] : {
         std::tuple{CoreOperation::Derivative,25,25}, {CoreOperation::Controller,3,4},
         {CoreOperation::Motors,4,8}, {CoreOperation::OmegaToThrust,4,4},
         {CoreOperation::ThrustToOmega,4,4}, {CoreOperation::ClampThrust,4,4},
         {CoreOperation::ClampMotorOmega,4,4}, {CoreOperation::ClampBodyRates,3,3},
         {CoreOperation::ConstrainWorld,25,25}}) {
    std::vector<float> data(nr*N, 1);
    if (nr == 25) data = state;
    auto in = input(data, DType::Float32, {Offset(nr),N}, guards);
    auto out = input(std::vector<float>(no*N, -999), DType::Float32, {Offset(no),N}, guards);
    auto bindings = bindOperation(sim, op, in, out);
    evaluateOperation(sim, op, bindings, .01f); check();
    auto status = download(sim.views().diagnostics.status.data, N);
    for (Offset n = 0; n < N; ++n) CHECK(status[n] == (active[n] ? LaneStatus::Ok : LaneStatus::Inactive));
    auto output = download(static_cast<float*>(out.tensor.data), no*N);
    for (int c = 0; c < no; ++c) for (Offset n = 0; n < N; ++n)
      CHECK(active[n] ? std::isfinite(output[c*N+n]) : output[c*N+n] == -999);
    CHECK(download(static_cast<float*>(in.tensor.data), nr*N) == data);
  }
  for (auto source : {ResetSource::Explicit, ResetSource::Random}) {
    seedEnvironment(sim, 29);
    resetEnvironment(sim, task_inputs, source); check();
    for (auto execution : {TaskExecution::FusedReset, TaskExecution::SeparateReset,
                           TaskExecution::NoReset, TaskExecution::FirstLaneReset}) {
      resetEnvironment(sim, task_inputs); check();
      stepEnvironment(sim, task_inputs, source, execution); check();
      auto graph = captureEnvironment(sim, task_inputs, source, execution);
      for (int i = 0; i < 3; ++i) sim.replay(graph);
      check();
    }
  }
  // Partial reset must preserve every nonselected lane, including the block tail.
  auto before = download(sim.views().state.x.data, 25*N);
  for (Offset n = 0; n < N; ++n) selected[n] = n % 2;
  CUDA(cudaMemcpy(si.tensor.data, selected.data(), N, cudaMemcpyHostToDevice));
  resetEnvironment(sim, task_inputs, ResetSource::Random); check();
  auto after = download(sim.views().state.x.data, 25*N);
  for (Offset n = 0; n < N; ++n) if (!selected[n] || !active[n])
    for (Offset c = 0; c < 25; ++c) CHECK(before[c*N+n] == after[c*N+n]);
  CHECK(download(static_cast<float*>(xi.tensor.data), 25*N) == state);
  CHECK(download(static_cast<float*>(ai.tensor.data), 4*N) == action);
  CHECK(download(static_cast<float*>(ci.tensor.data), 4*N) == command);
  std::cout << "guarded E=" << E << " A=" << A << " passed\n";
}
int main() {
  try {
    for (Index n : {1,31,32,33,127,128,129,257}) {
      exercise(1,n);
      if (n != 1) exercise(n,1);
    }
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
