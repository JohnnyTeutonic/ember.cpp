// Component experiment only. Production selector and inference defaults are unchanged.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
struct RowPick {float max, second; int first,last,nan;};
static_assert(sizeof(RowPick)==20,"production wire layout");
#include "selector.cuh"
#define CU(call) do {cudaError_t e=(call); if(e!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e));}while(0)
__global__ void merge_parts(const RowPick* parts, RowPick* out,int count){
 if(threadIdx.x) return;
 const RowPick* p=parts+blockIdx.x*count;
 RowPick r{-INFINITY,-INFINITY,INT_MAX,-1,0};
 for(int i=0;i<count;++i){r.max=fmaxf(r.max,p[i].max);r.nan|=p[i].nan;}
 for(int i=0;i<count;++i){
  if(p[i].max==r.max){r.first=min(r.first,p[i].first);r.last=max(r.last,p[i].last);r.second=fmaxf(r.second,p[i].second);}
  else if(p[i].max<r.max) r.second=fmaxf(r.second,p[i].max);
 }
 out[blockIdx.x]=r;
}
uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
bool equal(const RowPick&a,const RowPick&b){return bits(a.max)==bits(b.max)&&bits(a.second)==bits(b.second)&&a.first==b.first&&a.last==b.last&&a.nan==b.nan;}
RowPick oracle(const float* row,int v){
 RowPick r{-INFINITY,-INFINITY,INT_MAX,-1,0};
 for(int i=0;i<v;++i){if(std::isnan(row[i]))r.nan=1;else r.max=std::fmax(r.max,row[i]);}
 for(int i=0;i<v;++i){if(row[i]==r.max){r.first=std::min(r.first,i);r.last=std::max(r.last,i);}else if(row[i]<r.max)r.second=std::fmax(r.second,row[i]);}
 return r;
}
bool semantic_equal(const RowPick&a,const RowPick&b){return a.max==b.max&&a.second==b.second&&a.first==b.first&&a.last==b.last&&a.nan==b.nan;}
int decision(const RowPick&p,float temp){if(p.nan||!std::isfinite(p.max)||p.first<0||p.last<0)return -1;if(temp<.05f)return p.first;return ((p.second-p.max)*(1.f/temp)<-1e-5f)?p.last:-1;}
using Clock=std::chrono::steady_clock;
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("usage: row_pick_profile NEW.jsonl NEW_inputs.bin");
 {std::ifstream existing(argv[2]);if(existing.good())throw std::runtime_error("input capture exists");}
 std::ofstream inputs(argv[2],std::ios::binary);if(!inputs)throw std::runtime_error("input capture open failed");
 {std::ifstream existing(argv[1]);if(existing.good())throw std::runtime_error("output exists");}
 std::ofstream log(argv[1]);if(!log)throw std::runtime_error("cannot open output");
 cudaDeviceProp prop{};CU(cudaGetDeviceProperties(&prop,0));
 log<<"{\"kind\":\"device\",\"name\":\""<<prop.name<<"\",\"sm\":"<<prop.major*10+prop.minor<<",\"host_wall_scope\":\"n1 only; exact8B H2D and20B D2H\"}\n";log.flush();
 cudaStream_t stream;CU(cudaStreamCreate(&stream));
 float* device;RowPick *out,*partial,*host;int *step,*hoststep;
 CU(cudaMalloc(&device,16*128256*sizeof(float)));CU(cudaMalloc(&out,16*sizeof(RowPick)));CU(cudaMalloc(&partial,64*16*sizeof(RowPick)));
 CU(cudaMallocHost(&host,16*sizeof(RowPick)));CU(cudaMalloc(&step,8));CU(cudaMallocHost(&hoststep,8));hoststep[0]=17;hoststep[1]=101;
 std::mt19937 rng(20261031);auto dist=[](std::mt19937& generator){return (static_cast<int>(generator()%65536)-32768)*0.0009765625f;};
 int checks=0;
 for(int v:{1,31,257,4096,16384,32000,32768,65536,128256})for(int n:{1,4,16}){
  std::vector<float> row(v*n);
  for(int fixture=0;fixture<12;++fixture){
   for(auto&x:row)x=dist(rng);
   for(int r=0;r<n;++r){float* x=row.data()+r*v;
    if(fixture==1)std::fill(x,x+v,2.f);
    if(fixture==2)std::fill(x,x+v,-INFINITY);
    if(fixture==3)std::fill(x,x+v,NAN);
    if(fixture==4){x[0]=INFINITY;x[v-1]=INFINITY;}
    if(fixture==5)x[v/2]=NAN;
    if(fixture==6){std::fill(x,x+v,-0.f);x[v/2]=0.f;}
    if(fixture==7){std::fill(x,x+v,0.f);x[v/2]=-0.f;}
    if(fixture==8){x[0]=40;x[v-1]=40;}
    if(fixture==9){x[v/2]=40; if(v>1)x[v/2-1]=std::nextafter(40.f,-INFINITY);}
    if(fixture==10){x[0]=0.f;if(v>1)x[v-1]=-1e-5f;}
    if(fixture==11){x[0]=std::numeric_limits<float>::denorm_min();if(v>1)x[v-1]=-std::numeric_limits<float>::denorm_min();}
   }
   inputs.write(reinterpret_cast<const char*>(row.data()),row.size()*4);if(!inputs)throw std::runtime_error("input capture write failed");
   CU(cudaMemcpyAsync(device,row.data(),row.size()*4,cudaMemcpyHostToDevice,stream));
   select_rows_kernel<<<n,256,0,stream>>>(device,v,out);CU(cudaGetLastError());CU(cudaMemcpyAsync(host,out,n*20,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
   std::vector<RowPick> original(host,host+n);
   for(int r=0;r<n;++r)if(!semantic_equal(original[r],oracle(row.data()+r*v,v)))throw std::runtime_error("original disagrees with independent oracle");
   for(int parts:{4,16,64}){
    partition_rows_kernel<<<dim3(n,parts),256,0,stream>>>(device,v,partial,parts);merge_parts<<<n,32,0,stream>>>(partial,out,parts);CU(cudaGetLastError());
    CU(cudaMemcpyAsync(host,out,n*20,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
    for(int r=0;r<n;++r){if(!equal(host[r],original[r])){log<<"{\"kind\":\"MISMATCH\",\"vocab\":"<<v<<",\"n\":"<<n<<",\"fixture\":"<<fixture<<",\"parts\":"<<parts<<",\"row\":"<<r<<",\"original\":["<<bits(original[r].max)<<","<<bits(original[r].second)<<","<<original[r].first<<","<<original[r].last<<","<<original[r].nan<<"],\"candidate\":["<<bits(host[r].max)<<","<<bits(host[r].second)<<","<<host[r].first<<","<<host[r].last<<","<<host[r].nan<<"]}\n";log.flush();throw std::runtime_error("candidate bitwise field mismatch");}
     for(float t:{.01f,.05f,.7f,1.f})if(decision(host[r],t)!=decision(original[r],t))throw std::runtime_error("fallback mismatch");++checks;
    }
   }
  }
  log<<"{\"kind\":\"correctness\",\"vocab\":"<<v<<",\"n\":"<<n<<",\"cumulative_rows\":"<<checks<<"}\n";log.flush();
 }
 // Correctness completes for the entire matrix before any timing is accepted.
 for(int v:{4096,16384,32000,32768,65536,128256})for(int n:{1,4,16}){
  std::vector<float> row(v*n);for(auto&x:row)x=dist(rng);inputs.write(reinterpret_cast<const char*>(row.data()),row.size()*4);if(!inputs)throw std::runtime_error("input capture write failed");
   CU(cudaMemcpyAsync(device,row.data(),row.size()*4,cudaMemcpyHostToDevice,stream));CU(cudaStreamSynchronize(stream));
  for(int batch=0;batch<10;++batch)for(int order=0;order<4;++order){
   int parts=((batch+order)%4==0)?0:((batch+order)%4==1?4:((batch+order)%4==2?16:64));
   auto launch=[&](){if(!parts)select_rows_kernel<<<n,256,0,stream>>>(device,v,out);else{partition_rows_kernel<<<dim3(n,parts),256,0,stream>>>(device,v,partial,parts);merge_parts<<<n,32,0,stream>>>(partial,out,parts);}};
   for(int i=0;i<100;++i)launch();CU(cudaStreamSynchronize(stream));
   cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));CU(cudaEventRecord(a,stream));for(int i=0;i<1000;++i)launch();CU(cudaEventRecord(b,stream));CU(cudaEventSynchronize(b));float ms;CU(cudaEventElapsedTime(&ms,a,b));CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
   log<<"{\"kind\":\"kernel_trial\",\"vocab\":"<<v<<",\"n\":"<<n<<",\"parts\":"<<parts<<",\"batch\":"<<batch<<",\"microseconds\":"<<ms<<"}\n";log.flush();
   if(n==1){
    for(int transfer_only=0;transfer_only<2;++transfer_only){
     cudaGraph_t graph;cudaGraphExec_t exec;CU(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));if(!transfer_only)launch();CU(cudaStreamEndCapture(stream,&graph));CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
     auto invoke=[&](){CU(cudaMemcpyAsync(step,hoststep,8,cudaMemcpyHostToDevice,stream));CU(cudaGraphLaunch(exec,stream));CU(cudaMemcpyAsync(host,out,20,cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));};
     for(int i=0;i<100;++i)invoke();auto start=Clock::now();for(int i=0;i<1000;++i)invoke();double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/1000;
     log<<"{\"kind\":\"host_trial\",\"vocab\":"<<v<<",\"n\":1,\"parts\":"<<parts<<",\"transfer_only\":"<<transfer_only<<",\"batch\":"<<batch<<",\"microseconds\":"<<us<<"}\n";log.flush();CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
    }
   }
  }
 }
 log<<"{\"kind\":\"COMPLETE\",\"checked_rows\":"<<checks<<"}\n";log.flush();
 CU(cudaFree(device));CU(cudaFree(out));CU(cudaFree(partial));CU(cudaFree(step));CU(cudaFreeHost(host));CU(cudaFreeHost(hoststep));CU(cudaStreamDestroy(stream));return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}