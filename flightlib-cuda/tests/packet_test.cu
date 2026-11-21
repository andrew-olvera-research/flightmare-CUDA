#include "packet_fixture.cuh"
#include <algorithm>
#include <deque>
#include <iostream>
#include <random>

using namespace packet_fixture;
void require(bool ok) {
  if (!ok) throw std::runtime_error("packet timeline mismatch");
}
int main() {
  // Hand timeline: 2 bytes at 3 ns/byte + 4 ns propagation arrive at 10.
  Channel single(1, 2, 5);
  single.profiles.upload({{3,4}});
  single.attempts.upload({{7,10,2}});
  single.run(nullptr);
  require(single.results.read()[0].delivered == 0);
  single.attempts.upload({{}});
  single.run(nullptr);
  require(single.results.read()[0].delivered == 1);
  auto first = single.output.read()[0];
  require(first.id == 7 && first.arrival_ns == 10);

  Channel whole(1, 2, 10);
  whole.profiles.upload({{3,4}});
  whole.attempts.upload({{7,10,2}});
  whole.run(nullptr);
  auto unsplit = whole.output.read()[0];
  require(whole.results.read()[0].delivered == 1 &&
          unsplit.id == first.id && unsplit.arrival_ns == first.arrival_ns);
  auto at_limit = whole.state.read();
  at_limit[0].now = ~Time{0} - 1;
  whole.state.upload(at_limit);
  whole.run(nullptr);
  require(whole.results.read()[0].admission == Admission::ClockOverflow);
  require(whole.state.read()[0].now == at_limit[0].now);

  constexpr unsigned n = 257, capacity = 4;
  Channel channel(n, capacity, 1);
  std::vector<Profile> profiles(n);
  for (unsigned i=0; i<n; ++i) profiles[i] = {i%5, i%9};
  channel.profiles.upload(profiles);
  std::vector<std::deque<Event>> pending(n);
  std::vector<Time> busy(n);
  std::mt19937 rng(22);
  Time now = 0;
  for (unsigned step=0; step<160; ++step) {
    channel.tick = 1 + rng()%8;
    std::vector<Attempt> attempts(n);
    for (auto& a : attempts) a = {rng(), rng()%200, unsigned(rng()%14)};
    channel.attempts.upload(attempts);
    channel.run(nullptr);
    auto results = channel.results.read();
    auto events = channel.output.read();
    for (unsigned i=0; i<n; ++i) {
      auto& q = pending[i];
      auto a = attempts[i];
      auto p = profiles[i];
      auto finish = std::max(now,busy[i]) + a.bytes*p.ns_per_byte;
      auto arrival = finish + p.propagation_ns;
      Admission expected = Admission::Idle;
      if (a.bytes) {
        if (q.size() == capacity) expected = Admission::Full;
        else if (arrival > now+a.ttl_ns) expected = Admission::Deadline;
        else {
          expected = Admission::Accepted;
          busy[i] = finish;
          q.push_back({a.id,arrival});
        }
      }
      require(results[i].admission == expected);
      unsigned count = 0;
      while (!q.empty() && q.front().arrival_ns <= now+channel.tick) {
        Event actual = events[i*capacity+count++];
        require(actual.id == q.front().id && actual.arrival_ns == q.front().arrival_ns);
        q.pop_front();
      }
      require(count == results[i].delivered);
    }
    now += channel.tick;
  }
  // Exercise bounded storage across repeated construction and saturated runs.
  // cudaMemGetInfo is diagnostic on a shared GPU; conservation is a hard gate.
  size_t free_before, free_after, total;
  checked(cudaMemGetInfo(&free_before, &total));
  constexpr unsigned links = 8192, ticks = 1024;
  for (unsigned cycle=0; cycle<8; ++cycle) {
    Channel loaded(links, 4, 10);
    loaded.profiles.upload(std::vector<Profile>(links, {6,0}));
    loaded.attempts.upload(std::vector<Attempt>(links, {1,1000,10}));
    for (unsigned t=0; t<ticks; ++t) loaded.run(nullptr);
    for (const Link& link : loaded.state.read()) {
      require(link.delivered == ticks/6);
      require(link.accepted == link.delivered + link.size);
      require(link.accepted + link.rejected_full == ticks);
    }
  }
  checked(cudaMemGetInfo(&free_after, &total));
  std::cout << "67108864 loaded link intervals; free device bytes before/after: "
            << free_before << "/" << free_after << "\n";
  std::cout << "41120 link intervals matched the CPU reference\n";
}
