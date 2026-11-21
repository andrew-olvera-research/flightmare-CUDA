#include "shared_fixture.cuh"
#include "flightlib_cuda/dynamics.hpp"
#include <iostream>
#include <string>
#include <tuple>
using namespace packet_fixture;

#define require(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)

// A consumer with three immutable offer rows and a reusable output slab.
void continuation() {
  namespace plant = flightlib_cuda::v1;
  constexpr float dt = .001f;
  constexpr unsigned tick_ns = 1000000;
  plant::RuntimeConfig config;
  config.environments = config.slots = config.substeps = 1;
  plant::Simulation sim(config);
  plant::configure(sim, {plant::ModelConfig{}}, {0}, {1}, {-10,10,-10,10,-10,10}, dt);
  auto mode = plant::CommandMode::RotorThrust;
  checked(cudaMemcpy(sim.buffer("commands.mode"), &mode, sizeof(mode), cudaMemcpyHostToDevice));
  plant::stepDynamics(sim);
  sim.synchronize();
  auto plantTime = [&] {
    float t;
    checked(cudaMemcpy(&t, sim.views().state.time.data, sizeof(t), cudaMemcpyDeviceToHost));
    return t;
  };
  require(plantTime() == dt);

  SharedProfile profile{1,1,2,0,0,22};
  SharedChannel single(1,1,tick_ns,profile), resumed(1,1,tick_ns,profile);
  Buffer<unsigned> offsets(3);
  offsets.upload({0,20,50});
  Buffer<Frame> offers(3);
  offers.upload({{4,1,0,0,5,0,7,10,44,0}, {5,1,0,0,10,0,8,11,45,30},
                 {6,0,1,0,20,1,9,12,46,0}});
  Buffer<Reception> complete(3), slab(2);
  Buffer<TimelineCursor> cursor(1);
  advanceSharedSchedule(1,1,tick_ns,profile,single.media.data,offers.data,single.senders.data,
                        complete.data,nullptr,nullptr,3,offsets.data);
  auto expected = complete.read();
  auto drain = [&](unsigned budget, unsigned rows, unsigned next, unsigned written, Time horizon) {
    continueSharedSchedule(1,1,tick_ns,profile,resumed.media.data,offers.data,resumed.senders.data,
                           slab.data,nullptr,nullptr,3,offsets.data,cursor.data,budget,rows);
    auto c = cursor.read()[0];
    require(c.next == next && c.written == written && c.through_ns == horizon);
    require(resumed.media.read()[0].now == horizon && plantTime() == dt);
  };
  drain(3,0,0,0,0);
  drain(0,1,0,0,0);
  const Time horizons[]{20,50,tick_ns};
  for (unsigned i=0; i<3; ++i) {
    drain(i == 1 ? 1 : 3, i == 1 ? 2 : 1, i+1, 1, horizons[i]);
    auto row = slab.read()[0];
    auto ref = expected[i];
    auto fields = [](const Reception& r) {
      return std::make_tuple(r.id,r.arrival,r.receivers,r.receiver_base,r.completed,r.collided,
                             r.all_peers,r.admission,r.submitted,r.attempt_id,r.handle,r.expired,
                             r.fixture_label,r.air_start,r.air_end);
    };
    require(fields(row) == fields(ref));
  }
  drain(3,1,3,0,tick_ns);
}

int measurement(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--saturation") {
    // 9 us submission cadence keeps the sender backlogged during recovery.
    // 32 disjoint replicate domains for each contender count.
    for (unsigned n=1; n<=8; ++n) {
      SharedChannel ch(32,n,9000,{9000,16,148,32000,72000,22});
      ch.offers.upload(std::vector<Frame>(32*n,{1,1,0,0,1000}));
      for (unsigned i=0; i<12000; ++i) ch.run(nullptr);
      auto states = ch.senders.read();
      for (unsigned e=0; e<32; ++e) {
        Time delivered = 0;
        for (unsigned j=0; j<n; ++j) delivered += states[e*n+j].received_copies;
        std::cout << n << ',' << e << ',' << delivered << '\n';
      }
    }
    return 0;
  }
  if(argc==2 && std::string(argv[1])=="--broadcast-replay"){
    std::vector<unsigned> masks;
    unsigned long long total=0;
    auto replay=[&]{
      auto domains=static_cast<unsigned>(masks.size());
      SharedChannel ch(domains,3,100,{1,1,1,0,0,22});
      std::vector<Frame> offers(size_t(domains)*3);
      for(unsigned e=0;e<domains;++e)
        offers[3*e]={total+e,0,((~masks[e])&3u)<<1,0,32,true};
      ch.offers.upload(offers);ch.run(nullptr);
      auto results=ch.results.read();
      for(unsigned e=0;e<domains;++e){
        auto r=results[3*e];
        require(r.completed && r.id==total+e && r.arrival==32);
        for(unsigned rx=1;rx<=2;++rx)
          require(receivedBy(r,0,rx,3)==bool(masks[e] & (1u<<(rx-1))));
      }
      total+=domains;masks.clear();
    };
    char c;
    while(std::cin.get(c)){
      require(c>='0' && c<='3');masks.push_back(c-'0');
      if(masks.size()==4096)replay();
    }
    require(std::cin.eof());if(!masks.empty())replay();
    require(total>0);std::cout<<total<<"\n";return 0;
  }
  return 2;
}

void collisions() {
  SharedChannel ch(2,2,5,{1,1,1,0,2,22});
  // Two simultaneous senders collide in medium 0. In medium 1 a broadcast
  // takes ten ns and delivers to two of three receivers.
  ch.offers.upload({{1,1,0,0,10},{2,1,0,0,10},{3,7,2,64,10},{}});
  ch.run(nullptr);
  for (auto r: ch.results.read()) require(!r.completed);
  ch.run(nullptr);
  auto r = ch.results.read();
  require(r[0].collided && r[1].collided && !r[0].receivers && !r[1].receivers);
  require(r[2].completed && !r[2].collided && r[2].receivers==5 &&
          r[2].arrival==10 && r[2].receiver_base==64 && r[2].id==3 &&
          r[2].admission==Admission::Full);
  auto m = ch.media.read();
  require(m[0].collisions==1 && m[0].transmissions==2 && m[0].busy_ns==10);
  require(m[1].collisions==0 && m[1].transmissions==1 && m[1].busy_ns==10);
  // New offer in medium 1 waits for the previous transmission's recovery.
  ch.offers.upload({{}, {}, {}, {4,1,0,0,2}});
  ch.run(nullptr);
  r = ch.results.read();
  require(r[3].completed && r[3].arrival==14 && r[3].receivers==1);

}

void wideBroadcast() {
  // One transmission addresses every other station across a block boundary.
  SharedChannel wide(1,257,10,{1,1,1,0,0,22});
  std::vector<Frame> frames(257);
  frames[128] = {7,0,3,128,10,true}; // erase self (already excluded) and station 129
  wide.offers.upload(frames);
  wide.run(nullptr);
  auto broadcast = wide.results.read()[128];
  require(wide.media.read()[0].busy_ns==10 && wide.senders.read()[128].received_copies==255);
  for(unsigned rx=0; rx<257; ++rx)
    require(receivedBy(broadcast,128,rx,257)==(rx!=128 && rx!=129));

}

void partition() {
  // Same offers, different idle advance boundaries, including partial slots.
  SharedProfile profile{7,16,1,3,4,42};
  SharedChannel whole(1,19,2000,profile), split(1,19,5,profile);
  std::vector<Frame> frames(19,{8,1,0,0,17});
  whole.offers.upload(frames);
  split.offers.upload(frames);
  whole.run(nullptr);
  split.run(nullptr);
  split.offers.upload(std::vector<Frame>(19));
  for(unsigned i=1; i<400; ++i) split.run(nullptr);
  auto expected = whole.senders.read(), actual = split.senders.read();
  for(unsigned i=0; i<19; ++i) {
    require(actual[i].completions==1 && expected[i].completions==1 &&
            actual[i].finish==expected[i].finish &&
            actual[i].received_copies==expected[i].received_copies);
  }
}

int main(int argc, char** argv) {
  if (argc > 2) return 2;
  std::string name = argc == 1 ? "" : argv[1];
  if (name == "--saturation" || name == "--broadcast-replay") return measurement(argc,argv);
  if (name.empty() || name == "--continuation") continuation();
  if (name.empty() || name == "--collisions") collisions();
  if (name.empty() || name == "--broadcast") wideBroadcast();
  if (name.empty() || name == "--partition") partition();
  return name.empty() || name == "--continuation" || name == "--collisions" ||
         name == "--broadcast" || name == "--partition" ? 0 : 2;
}
