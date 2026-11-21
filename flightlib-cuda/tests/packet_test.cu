#include "packet_fixture.cuh"
#include <algorithm>
#include <deque>
#include <iostream>
#include <random>
#include <string>

using namespace packet_fixture;
void require(bool ok) {
  if (!ok) throw std::runtime_error("packet timeline mismatch");
}
// One lane per independent three-repetition measurement. Each loss must still
// consume transmitter airtime. Artificial 1 ns ticks make the three sends queue.
void replayBursts(const std::vector<unsigned>& masks) {
  Channel ch(static_cast<unsigned>(masks.size()), 3, 1);
  ch.profiles.upload(std::vector<Profile>(masks.size(), {10,0}));
  std::vector<Attempt> attempts(masks.size());
  for (unsigned rep=0; rep<3; ++rep) {
    for (size_t i=0; i<masks.size(); ++i)
      attempts[i] = {3*i+rep, 2000, 32, 1u-((masks[i]>>rep)&1u)};
    ch.attempts.upload(attempts);
    ch.run(nullptr);
    for (auto r: ch.results.read())
      require(r.admission == Admission::Accepted && r.delivered == 0);
  }
  ch.tick = 960;
  ch.attempts.upload(std::vector<Attempt>(masks.size()));
  ch.run(nullptr);
  auto events = ch.output.read();
  auto results = ch.results.read();
  auto state = ch.state.read();
  for (size_t i=0; i<masks.size(); ++i) {
    unsigned count = 0;
    for (unsigned rep=0; rep<3; ++rep) {
      if ((masks[i]>>rep)&1u) {
        auto event = events[3*i+count++];
        require(event.id == 3*i+rep && event.arrival_ns == (rep+1)*320);
      }
    }
    require(results[i].delivered == count && state[i].modeled_loss == 3-count &&
            state[i].accepted == 3 && state[i].tx_free == 960 && state[i].size == 0);
  }
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--replay") {
    std::vector<unsigned> masks;
    char c;
    size_t bursts = 0;
    while (std::cin.get(c)) {
      if (c < '0' || c > '7') throw std::invalid_argument("expected burst mask digit");
      masks.push_back(static_cast<unsigned>(c-'0'));
      ++bursts;
      if (masks.size() == 4096) {
        replayBursts(masks);
        masks.clear();
      }
    }
    require(std::cin.eof() && bursts);
    if (!masks.empty()) replayBursts(masks);
    std::cout << bursts << "\n";
    return 0;
  }
  if (argc != 1) return 2;
  replayBursts({0,1,2,3,4,5,6,7});
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
