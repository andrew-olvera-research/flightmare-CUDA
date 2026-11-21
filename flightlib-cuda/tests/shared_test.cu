#include "shared_fixture.cuh"
#include <iostream>
#include <string>
using namespace packet_fixture;

void require(bool ok) { if (!ok) throw std::runtime_error("shared channel mismatch"); }

int main(int argc, char** argv) {
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
  require(argc == 1);
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

  // Same offers, different idle advance boundaries, including partial slots.
  SharedProfile profile{7,16,1,3,4,42};
  SharedChannel whole(1,19,2000,profile), split(1,19,5,profile);
  frames.assign(19,{8,1,0,0,17});
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
