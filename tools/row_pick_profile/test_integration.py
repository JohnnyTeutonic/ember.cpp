"""CPU-only production integration contract and exact-source checks.

These tests do not execute CUDA or establish a full-model performance gate.
"""
import importlib.util
from pathlib import Path
import subprocess
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def body(source, start):
    a = source.index(start)
    b = source.index("{", a)
    depth, end = 1, b + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[a:end]


def test_production_bodies_equal_frozen_component(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_component", HERE / "prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    subprocess.run(["python3", str(HERE / "prepare.py"), "--root", str(ROOT),
                    "--output", str(tmp_path)], check=True)
    generated = (tmp_path / "selector.cuh").read_text()
    integrated = (ROOT / "row_pick_partitioned.cuh").read_text()
    assert body(generated, "__global__ void partition_rows_kernel(") == body(
        integrated, "__global__ void partition_rows_kernel(")
    assert body((HERE / "profile.cu").read_text(), "__global__ void merge_parts(") == body(
        integrated, "__global__ void merge_parts(")
    # The existing selector is untouched; its component-generated body is exact.
    assert prepare.extract((ROOT / "verify_kernels.cu").read_text()) in generated


def test_policy_and_mode_cache(tmp_path):
    source = r"""
#include "decode_pick_policy.h"
#include <cassert>
#include <stdexcept>
int main() {
  for (const char* v : {static_cast<const char*>(nullptr), "", "0"}) {
    assert(ember::decode_pick_partitions(v, false) == 0);
    assert(ember::decode_pick_partitions(v, true) == 0);
  }
  assert(ember::decode_pick_partitions("4", true) == 4);
  for (const char* v : {"4", "2", "16", "garbage", " 4", "04"}) {
    bool refused = false;
    try { (void)ember::decode_pick_partitions(v, false); }
    catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
  }
  int cache1, cache2;
  for (bool oldpick : {false, true}) for (bool newpick : {false, true})
    for (int oldparts : {0, 4}) for (int newparts : {0, 4}) {
      bool same = oldpick == newpick && (!newpick || oldparts == newparts);
      assert(ember::decode_pick_graph_matches(&cache1, &cache1, oldpick, newpick, oldparts, newparts) == same);
      assert(!ember::decode_pick_graph_matches(&cache1, &cache2, oldpick, newpick, oldparts, newparts));
    }
}
"""
    cpp, exe = tmp_path / "policy.cpp", tmp_path / "policy"
    cpp.write_text(textwrap.dedent(source))
    subprocess.run(["g++", "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_lazy_embedding_from_actual_generation_helper(tmp_path):
    api = (ROOT / "api.cpp").read_text()
    helper = body(api, "auto ensure_host_embedding = [&]()") + ";"
    source = r"""
#include <cassert>
#include <vector>
struct Model {
 int calls = 0; bool valid = true;
 std::vector<float> lookup_embedding(int id) { ++calls; return valid ? std::vector<float>{float(id)} : std::vector<float>{}; }
};
int main() {
 Model model; Model* model_ = &model;
 int input_token_id = 17;
 std::vector<float> current_data_host{999};
 bool host_embedding_ready = false;
 HELPER
 // A successful graph never needs this lazy host-only value.
 assert(model.calls == 0);
 // Diagnostics or graph fallback ask for the current input, never stale data.
 assert(ensure_host_embedding());
 assert(model.calls == 1 && current_data_host == std::vector<float>{17});
 assert(ensure_host_embedding() && model.calls == 1);
 // Failed lookup is remembered, remains empty, and cannot reuse a prior token.
 model.valid = false; host_embedding_ready = false; input_token_id = 18;
 assert(!ensure_host_embedding() && current_data_host.empty());
 assert(!ensure_host_embedding() && model.calls == 2);
}
""".replace("HELPER", helper)
    cpp, exe = tmp_path / "embedding.cpp", tmp_path / "embedding"
    cpp.write_text(textwrap.dedent(source))
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_capture_allocation_lifetime_and_dispatch_source():
    model = (ROOT / "model.cpp").read_text()
    graph = body(model, "std::vector<float> TinyLlamaModel::forward_device_graph(")
    assert graph.index("decode_pick_parts()") < graph.index("if (!enabled")
    assert graph.index("cudaMalloc(&decode_pick_parts_dev_") < graph.index("cudaStreamBeginCapture")
    assert "decode_graph_pick_parts_, parts" in graph
    assert "decode_graph_pick_parts_ = parts;" in graph
    assert "if (parts == 4)" in graph
    assert "ember_verify::select_rows_partitioned(logits_dev_, 1, vs, decode_pick_dev_, decode_pick_parts_dev_, st);" in graph
    assert "ember_verify::select_rows(logits_dev_, 1, vs, decode_pick_dev_, st);" in graph
    destructor = body(model, "TinyLlamaModel::~TinyLlamaModel()")
    assert destructor.index("cudaStreamSynchronize(decode_stream_)") < destructor.index("cudaGraphExecDestroy")
    assert destructor.index("cudaGraphExecDestroy") < destructor.index("cudaFree(decode_pick_parts_dev_)")
    api = (ROOT / "api.cpp").read_text()
    assert "defer_host_embedding = want_device_pick && model_->decode_pick_parts() == 4;" in api
    fallback = api[api.index("if (logits.empty() && !picked_on_device)"):]
    assert fallback.index("if (!ensure_host_embedding())") < fallback.index("cudaMemcpy(model_->get_x_dev()")


def test_actual_merge_body_with_negative_threshold_and_subnormal_peaks(tmp_path):
    merge = body((ROOT / "row_pick_partitioned.cuh").read_text(), "__global__ void merge_parts(")
    merge = merge.replace("__global__ ", "")
    record = body((ROOT / "verify_kernels.h").read_text(), "struct RowPick {") + ";"
    source = r"""
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
using std::min; using std::max;
RECORD
struct Dim { int x = 0; } threadIdx, blockIdx;
MERGE
RowPick summarize(const std::vector<float>& x, int begin, int end) {
 RowPick r{-INFINITY, -INFINITY, INT_MAX, -1, 0};
 for (int i=begin;i<end;++i) {if(std::isnan(x[i]))r.nan=1;else r.max=std::fmax(r.max,x[i]);}
 for (int i=begin;i<end;++i) {
  if(x[i]==r.max) {r.first=min(r.first,i);r.last=max(r.last,i);}
  else if(x[i]<r.max)r.second=std::fmax(r.second,x[i]);
 }
 return r;
}
int main() {
 std::mt19937 rng(20261031);
 std::uniform_real_distribution<float> uniform(-30,30);
 for (int n : {1,2,3,31,257,4096,16384}) for(int fixture=0;fixture<13;++fixture) {
  std::vector<float>x(n); for(float&v:x)v=uniform(rng);
  if(fixture==1)for(float&v:x)v=-std::abs(v)-.25f;
  if(fixture==2) {std::fill(x.begin(),x.end(),-2.f);x[0]=0;if(n>1)x[n-1]=-1e-5f;}
  if(fixture==3) {std::fill(x.begin(),x.end(),-2.f);x[0]=std::numeric_limits<float>::denorm_min();if(n>1)x[n-1]=-std::numeric_limits<float>::denorm_min();}
  if(fixture==4)std::fill(x.begin(),x.end(),NAN);
  if(fixture==5)std::fill(x.begin(),x.end(),-INFINITY);
  if(fixture==6) {std::fill(x.begin(),x.end(),-0.f);x[n/2]=0.f;}
  if(fixture==7) {std::fill(x.begin(),x.end(),0.f);x[n/2]=-0.f;}
  if(fixture==8) {x[0]=INFINITY;x[n-1]=INFINITY;}
  if(fixture==9) {x[0]=40.f;x[n-1]=40.f;}
  if(fixture==10) {x[0]=40.f;if(n>1)x[n-1]=std::nextafter(40.f,-INFINITY);}
  if(fixture==11)x[n/2]=NAN;
  if(fixture==12)std::fill(x.begin(),x.end(),-2.f);
  RowPick parts[4];for(int p=0;p<4;++p)parts[p]=summarize(x,n*p/4,n*(p+1)/4);
  RowPick expected=summarize(x,0,n), actual;
  merge_parts(parts,&actual,4);
  // CPU merge semantics only; signed-zero bit identity remains a device gate.
  assert(actual.max==expected.max && actual.second==expected.second);
  assert(actual.first==expected.first && actual.last==expected.last && actual.nan==expected.nan);
 }
}
""".replace("RECORD", record).replace("MERGE", merge)
    cpp, exe = tmp_path / "merge.cpp", tmp_path / "merge"
    cpp.write_text(textwrap.dedent(source))
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


@pytest.mark.parametrize("tamper", ["none", "instruction", "missing", "architecture", "wrong_additions", "duplicate"])
def test_sass_comparison_never_masks_instructions(tamper):
    spec = importlib.util.spec_from_file_location("check_sass", HERE / "check_integration_sass.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    prefix = "arch = sm_89\n"
    name = "_GLOBAL__N__12345678_17_verify_kernels_cu_test_original"
    old = prefix + "Function : " + name + "\n/*0000*/ MOV R1, R2; /*0x012345*/\n"
    new = old.replace("12345678", "abcdef01")
    new += "Function : partition_rows_kernel\n/*0000*/ EXIT; /*0x123*/\n"
    new += "Function : merge_parts\n/*0000*/ EXIT; /*0x456*/\n"
    if tamper == "none":
        assert check.compare(old, new)["existing_functions_equal"] == 1
        return
    if tamper == "instruction":
        new = new.replace("MOV R1, R2", "MOV R1, R3")
    elif tamper == "missing":
        new = new.replace(name.replace("12345678", "abcdef01"), "missing_original")
    elif tamper == "architecture":
        new = new.replace("sm_89", "sm_86")
    elif tamper == "wrong_additions":
        new = new.replace("merge_parts", "other_added_kernel")
    elif tamper == "duplicate":
        new += "Function : " + name + "\n/*0000*/ MOV R1, R2; /*0x012345*/\n"
    with pytest.raises(ValueError):
        check.compare(old, new)


def test_actual_cuda_dispatch_fragment_with_mock_transport(tmp_path):
    api = (ROOT / "api.cpp").read_text()
    a = api.index("      const bool want_pick = want_device_pick;")
    fragment = api[a:api.index("\n#else", a)]
    helper = body(api, "auto ensure_host_embedding = [&]()") + ";"
    greedy = body(api, "static int greedy_from_pick(")
    record = body((ROOT / "verify_kernels.h").read_text(), "struct RowPick {") + ";"
    source = r"""
#include <cassert>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
namespace ember_verify { RECORD }
GREEDY
bool debug_rows = false;
bool logits_debug_on() {return debug_rows;}
struct Logger {static void error(const char*) {}};
int copies = 0;
constexpr int cudaMemcpyHostToDevice = 0;
int cudaMemcpy(void* dst, const void* src, size_t bytes, int) {++copies;std::memcpy(dst,src,bytes);return 0;}
#define gpuErrchk(x) ((void)(x))
struct Model {
 int calls=0, graph_calls=0, row_calls=0, forward_calls=0;
 bool available=true, valid=true, near_tie=false;
 float device=0;
 std::vector<float> lookup_embedding(int id) {++calls;return valid?std::vector<float>{float(id)}:std::vector<float>{};}
 std::vector<float> forward_device_graph(int,int,int*,ember_verify::RowPick* p) {
  ++graph_calls;
  if(p)p->first=-1;
  if(!available)return {};
  if(!p)return {0,0,1};
  *p={1,near_tie?.999999f:0,2,2,0};return {};
 }
 std::vector<float> decode_row_logits() {++row_calls;return {0,0,1};}
 float* get_x_dev() {return &device;}
 std::vector<float> forward_device(float*,int,int*,std::nullptr_t) {
  ++forward_calls;assert(device==17);return {0,0,1};
 }
};
int main() {
 for(int test=0;test<6;++test) {
  Model model;Model* model_=&model;
  model.near_tie=test==1;model.available=test<3||test==5;model.valid=test!=4;
  debug_rows=test==2;copies=0;
  bool want_device_pick=test!=5, picked_on_device=false, host_embedding_ready=false;
  bool completed=false;
  int input_token_id=17,pos=5,kv_cache_=0,next_token_id=31;
  float temperature=1;
  std::mt19937 rng_(0);
  std::vector<float> current_data_host{999}, logits;
  auto host_phase=[](const char*){};
  HELPER
  if(!want_device_pick)ensure_host_embedding();
  for(int once=0;once<1;++once) {
   FRAGMENT
   completed=true;
  }
  assert(model.graph_calls==1);
  if(test==0) {
   assert(completed&&picked_on_device&&next_token_id==2);
   assert(model.calls==0&&model.row_calls==0&&model.forward_calls==0&&copies==0);
  } else if(test==1||test==2) {
   assert(completed&&!picked_on_device&&logits==std::vector<float>({0,0,1}));
   assert(model.calls==0&&model.row_calls==1&&model.forward_calls==0&&copies==0);
  } else if(test==3) {
   assert(completed&&!picked_on_device&&logits==std::vector<float>({0,0,1}));
   assert(model.calls==1&&model.row_calls==0&&model.forward_calls==1&&copies==1);
  } else if(test==4) {
   assert(!completed&&!picked_on_device&&current_data_host.empty());
   assert(model.calls==1&&model.row_calls==0&&model.forward_calls==0&&copies==0);
  } else {
   assert(completed&&!picked_on_device&&logits==std::vector<float>({0,0,1}));
   assert(model.calls==1&&model.row_calls==0&&model.forward_calls==0&&copies==0);
  }
 }
}
""".replace("RECORD", record).replace("GREEDY", greedy).replace("HELPER", helper).replace("FRAGMENT", fragment)
    cpp, exe = tmp_path / "dispatch.cpp", tmp_path / "dispatch"
    cpp.write_text(textwrap.dedent(source))
    subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
