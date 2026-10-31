// Exact partition and merge bodies from the frozen standalone component
// experiment (tools/row_pick_profile/prepare.py and profile.cu). Included inside
// verify_kernels.cu's anonymous namespace after kSelThreads is declared.
// Only the explicit experimental decode path calls these kernels.
__global__ void partition_rows_kernel(const float* __restrict__ logits, int vocab, RowPick* __restrict__ picks, int parts) {
  const float* l = logits + (size_t)blockIdx.x * vocab;
  __shared__ float s_f[kSelThreads / 32];
  __shared__ int s_a[kSelThreads / 32];
  __shared__ int s_b[kSelThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  float mx = -INFINITY;
  int nan = 0;
  for (int i = ((long long)vocab * blockIdx.y / parts) + threadIdx.x; i < (long long)vocab * (blockIdx.y + 1) / parts; i += blockDim.x) {
    const float v = l[i];
    if (v != v) nan = 1;
    else mx = fmaxf(mx, v);
  }
  for (int o = 16; o > 0; o >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    nan |= __shfl_xor_sync(0xffffffff, nan, o);
  }
  if (lane == 0) { s_f[warp] = mx; s_a[warp] = nan; }
  __syncthreads();
  mx = s_f[0];
  nan = s_a[0];
  for (int w = 1; w < nwarps; ++w) { mx = fmaxf(mx, s_f[w]); nan |= s_a[w]; }
  __syncthreads();
  int first = INT_MAX, last = -1;
  float second = -INFINITY;
  for (int i = ((long long)vocab * blockIdx.y / parts) + threadIdx.x; i < (long long)vocab * (blockIdx.y + 1) / parts; i += blockDim.x) {
    const float v = l[i];
    if (v == mx) {
      first = min(first, i);
      last = max(last, i);
    } else if (v < mx) {
      second = fmaxf(second, v);
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    first = min(first, __shfl_xor_sync(0xffffffff, first, o));
    last = max(last, __shfl_xor_sync(0xffffffff, last, o));
    second = fmaxf(second, __shfl_xor_sync(0xffffffff, second, o));
  }
  if (lane == 0) { s_f[warp] = second; s_a[warp] = first; s_b[warp] = last; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < nwarps; ++w) {  // warp 0's values are this thread's
      second = fmaxf(second, s_f[w]);
      first = min(first, s_a[w]);
      last = max(last, s_b[w]);
    }
    RowPick p;
    p.max = mx;
    p.second = second;
    p.first = first;
    p.last = last;
    p.nan = nan;
    picks[blockIdx.x * parts + blockIdx.y] = p;
  }
}

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
