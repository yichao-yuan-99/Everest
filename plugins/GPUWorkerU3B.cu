#include <stdio.h>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "helpers.cuh"
#include "data.h"

using namespace corelib;
using namespace corelib::data;

__device__ static void reduceEarlyIdx(const int *i_S, int &beg, int end, int i_g) {
  if (end == beg) { beg = end; return;}
  if (i_S[end - 1] <= i_g) {
    beg = end;
    return;
  } else if (i_S[beg] > i_g) {
    return;
  }

  // binary search the first value >= i_g, at this point i_S has at least 2 elements
  int p = 0;
  while (beg < end) {
    p = (beg + end) / 2;
    if (i_S[p] > i_g) {
      end = p;
    } else {
      beg = p + 1;
    }
  }
}

struct TContext {
  int stackbeg0, stackbeg1, stackbeg2, stackbeg3;
  int stackend0, stackend1, stackend2, stackend3;
  int m0, m1, m2, m3, m4;

  int tl;

  unsigned char stackio;
  int level;
  int beg, end;
};


__device__ static int firstLarger(int *arr, int beg, int end, int v) {
  while (beg < end) {
    int mid = (beg + end) / 2;
    if (arr[mid] <= v) {
      beg = mid + 1;
    } else {
      end = mid;
    }
  }
  return end;
}

__device__ static void probeEnd(const int *i_S, int beg, int &end, int tl, const TemporalEdge *Eg) {
  while (beg < end) {
    int mid = (beg + end) / 2;
    if (Eg[i_S[mid]].t <= tl) {
      beg = mid + 1;
    } else {
      end = mid;
    }
  }
}

__device__ static void dumpContext(
  int stackbeg0, int stackbeg1, int stackbeg2, int stackbeg3,
  int stackend0, int stackend1, int stackend2, int stackend3,
  int m0, int m1, int m2, int m3, int m4,
  
  
  int tl,
  unsigned char stackio,
  int level,
  int beg, int end, const int *i_S,
  
  // graph
  const TemporalEdge *Eg,
  const int *inEdgesV,
  const int *outEdgesV,
  
  // mem
  TContext *offload,
  int *offtop,
  int *offload_width
) {
  int width = 0;
  probeEnd(i_S, beg, end, tl, Eg);
  width = max(width, end - beg);
  auto stackiob = stackio;
  switch (level) {
    case 4: 
      stackiob >>= 1;
      i_S = stackiob & 1 ? outEdgesV : inEdgesV;
      probeEnd(i_S, stackbeg3, stackend3, tl, Eg);
      width = max(width, stackend3 - stackbeg3);
    case 3: 
      stackiob >>= 1;
      i_S = stackiob & 1 ? outEdgesV : inEdgesV;
      probeEnd(i_S, stackbeg2, stackend2, tl, Eg);
      width = max(width, stackend2 - stackbeg2);
    case 2:
      stackiob >>= 1;
      i_S = stackiob & 1 ? outEdgesV : inEdgesV;
      probeEnd(i_S, stackbeg1, stackend1, tl, Eg);
      width = max(width, stackend1 - stackbeg1);
    case 1:
      stackiob >>= 1;
      i_S = stackiob & 1 ? outEdgesV : inEdgesV;
      probeEnd(i_S, stackbeg0, stackend0, tl, Eg);
      width = max(width, stackend0 - stackbeg0);
  }

  if (width == 0) return;

  int offpos = atomicAdd(offtop, 1);

  // each thread 
  offload[offpos].beg = beg;
  offload[offpos].end = end;
  offload[offpos].level = level;
  offload[offpos].stackio = stackio;
  offload[offpos].tl = tl;

  offload_width[offpos] = width;

  offload[offpos].stackbeg0 = stackbeg0;
  offload[offpos].stackbeg1 = stackbeg1;
  offload[offpos].stackbeg2 = stackbeg2;
  offload[offpos].stackbeg3 = stackbeg3;
  offload[offpos].stackend0 = stackend0;
  offload[offpos].stackend1 = stackend1;
  offload[offpos].stackend2 = stackend2;
  offload[offpos].stackend3 = stackend3;
  offload[offpos].m0 = m0;
  offload[offpos].m1 = m1;
  offload[offpos].m2 = m2;
  offload[offpos].m3 = m3;
  offload[offpos].m4 = m4;

}

__global__ static void MotifMatching_Expand(
  int work,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR,
  const int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,

  // motif
  const MotifEdgeInfoV1 *minfo, int numem,

  // runtime
  int *yeild,
  int *source,
  TContext *offload,
  int *offtop,
  int *offtopn,
  int *offload_width,

  int *chunk_offset,

  unsigned long long *gcount
) {
  auto laneid = threadIdx.x % 32;

  int tid;
  if (laneid == 0) tid = atomicAdd(source, 32);
  tid = __shfl_sync(0xffffffff, tid, 0);
  if (tid >= work) return;

  int stackbeg0 = 0, stackbeg1 = 0, stackbeg2 = 0, stackbeg3 = 0;
  int stackend0 = 0, stackend1 = 0, stackend2 = 0, stackend3 = 0;
  int m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;

  unsigned char stackio = 0;
  int tl;
  unsigned long long count = 0;
  int i_g = tid;
  int level = 0;
  int beg = 0, end = 0, base;
  MotifEdgeInfoV1 mi;
  const int *i_S = nullptr;
  clock_t timeup = 0;
  
  
  while (tid < work) { // valid block
    tid = (tid / 32) + chunk_offset[laneid];

    if (tid < work) {
      auto loc = firstLarger(offload_width, 0, *offtop, tid);
      int offset = loc ? tid - offload_width[loc - 1] : tid;

      // load context from loc
      level = offload[loc].level;
      tl = offload[loc].tl;
      stackio = offload[loc].stackio;

      beg = offload[loc].beg + offset;
      end = min(beg + 1, offload[loc].end);

      stackbeg0 = offload[loc].stackbeg0 + offset;
      stackbeg1 = offload[loc].stackbeg1 + offset;
      stackbeg2 = offload[loc].stackbeg2 + offset;
      stackbeg3 = offload[loc].stackbeg3 + offset;
      stackend0 = min(stackbeg0 + 1, offload[loc].stackend0);
      stackend1 = min(stackbeg1 + 1, offload[loc].stackend1);
      stackend2 = min(stackbeg2 + 1, offload[loc].stackend2);
      stackend3 = min(stackbeg3 + 1, offload[loc].stackend3);
      m0 = offload[loc].m0;
      m1 = offload[loc].m1;
      m2 = offload[loc].m2;
      m3 = offload[loc].m3;
      m4 = offload[loc].m4;


      mi = minfo[level];
      mi.io = stackio & 1;
      i_S = mi.io ? outEdgesV : inEdgesV;
      if (mi.constraintNode >= 0) mi.constraintNode = mi.io ? mi.baseNode : mi.constraintNode;
    } else {
      end = beg = 0;
    }

    int loopCnt = 0;
    while (true) {
      i_g = numeg;
      int node;
      for (; beg < end; beg++) {
        auto idx = i_S[beg];
        auto eg = Eg[idx];
        if (eg.t > tl) {
          end = beg;
          break;
        }
        node = mi.io ? eg.v : eg.u;
        bool checked = true;
        if (mi.constraintNode < 0) {
          switch (mi.mappedNodes) {
            case 5: checked = (m4 != node);
            case 4: checked = checked && (m3 != node);
            case 3: checked = checked && (m2 != node);
            case 2: checked = checked && (m1 != node) && (m0 != node);
          }
        } else {
          switch (mi.constraintNode) {
            case 4: checked = (m4 == node); break; 
            case 3: checked = (m3 == node); break; 
            case 2: checked = (m2 == node); break;
            case 1: checked = (m1 == node); break;
            case 0: checked = (m0 == node); break;
          }
        }
        // nodeConstraint(checked, node);
        if (checked) {
          i_g = idx; 
          beg++;
          break;
        }
      }

      bool alive = (level || (i_g < numeg));

      if (i_g < numeg) { 
        if (level == numem - 2) { 
          count++;
        } else {
          switch (mi.mappedNodes) {
            case 2: m2 = node;
            case 3: m3 = node; 
            case 4: m4 = node; 
          }
          switch (level) {
            case 0: stackbeg0 = beg; stackend0 = end;
            case 1: stackbeg1 = beg; stackend1 = end;
            case 2: stackbeg2 = beg; stackend2 = end;
            case 3: stackbeg3 = beg; stackend3 = end;
          }
          level++; 

          mi = minfo[level];
          stackio <<= 1;
          if (mi.io >= 0) {
            switch (mi.baseNode) { 
              case 0: base = m0; break;
              case 1: base = m1; break;
              case 2: base = m2; break;
              case 3: base = m3; break;
              case 4: base = m4;
            }
            beg = mi.arrR[base];
            end = mi.arrR[base + 1];
            i_S = mi.arrV;
          } else {
            int base0, base1;
            switch (mi.baseNode) { 
              case 0: base0 = m0; break;
              case 1: base0 = m1; break;
              case 2: base0 = m2; break;
              case 3: base0 = m3; break;
              case 4: base0 = m4;
            }
            switch (mi.constraintNode) { 
              case 0: base1 = m0; break;
              case 1: base1 = m1; break;
              case 2: base1 = m2; break;
              case 3: base1 = m3; break;
              case 4: base1 = m4;
            }
            int beg0 = inEdgesR[base0];
            int end0 = inEdgesR[base0 + 1];
            int beg1 = outEdgesR[base1];
            int end1 = outEdgesR[base1 + 1];
            if (end0 - beg0 < end1 - beg1) {
              mi.io = 0;
              beg = beg0;
              end = end0;
              i_S = inEdgesV;
            } else {
              mi.io = 1;
              mi.constraintNode = mi.baseNode;
              beg = beg1;
              end = end1;
              i_S = outEdgesV;
            }
          }
          stackio |= mi.io ? 1 : 0;

          reduceEarlyIdx(i_S, beg, end, i_g);
        }
      } else { 
        if (level) {
          level--;
          switch (level) {
            case 0: beg = stackbeg0; end = stackend0; break;
            case 1: beg = stackbeg1; end = stackend1; break;
            case 2: beg = stackbeg2; end = stackend2; break;
            case 3: beg = stackbeg3; end = stackend3; break;
          }
          stackio >>= 1;

          mi = minfo[level];
          mi.io = stackio & 1;
          i_S = mi.io ? outEdgesV : inEdgesV;
          if(mi.constraintNode >= 0) mi.constraintNode = mi.io ? mi.baseNode : mi.constraintNode;
        }
      }

      if (loopCnt % 1024 == 0 && *yeild) {
        if (laneid == 0) {
          timeup = clock() + 100000;
        } 
        timeup = __shfl_sync(0xffffffff, timeup, 0);
      }

      if (loopCnt % 64 == 0 && timeup && __shfl_sync(0xffffffff, clock(), 0) > timeup) {
        break;
      }

      if (__any_sync(0xffffffff, alive) == 0) break;

      if (laneid == 0) loopCnt++;
      loopCnt = __shfl_sync(0xffffffff, loopCnt, 0);
    }

    if (laneid == 0) tid = atomicAdd(source, 32);
    tid = __shfl_sync(0xffffffff, tid, 0);
  } // end of outer loop

  for (int offset = 16; offset > 0; offset /= 2)
    count += __shfl_down_sync(0xffffffff, count, offset);

  if (laneid == 0) atomicAdd(gcount, count);

  atomicAdd(yeild, 1);
  
  if ((level || (beg < end))) {
    dumpContext(
      stackbeg0, stackbeg1, stackbeg2, stackbeg3,
      stackend0, stackend1, stackend2, stackend3,
      m0, m1, m2, m3, m4,
      
  
      tl,
  
      stackio, level, beg, end, i_S,
  
      Eg,
      inEdgesV,
      outEdgesV,
  
      offload,
      offtopn,
      offload_width
    );
  }
}


__global__ static void MotifMatching_dispatch(
  int work, int delta,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR,
  const int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,

  // motif
  const MotifEdgeInfoV1 *minfo, int numem,

  // runtime
  int *yeild,
  int *source,
  TContext *offload,
  int *offtop,
  int *offtopn,
  int *offload_width,
  
  unsigned long long *gcount
) {

  auto laneid = threadIdx.x % 32;
  int tid;
  if (laneid == 0) tid = atomicAdd(source, 32);
  tid = __shfl_sync(0xffffffff, tid, 0);
  if (tid >= work) return;

  int stackbeg0 = 0, stackbeg1 = 0, stackbeg2 = 0, stackbeg3 = 0;
  int stackend0 = 0, stackend1 = 0, stackend2 = 0, stackend3 = 0;
  int m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;

  unsigned char stackio = 0;
  int tl;
  unsigned long long count = 0;
  int i_g = tid;
  int level = 0;
  int beg = 0, end = 0, base;
  MotifEdgeInfoV1 mi;
  const int *i_S = nullptr;
  clock_t timeup = 0;
  
  while (tid < work) { // valid block
    tid += laneid;

    bool fcheck = tid < work && (Eg[tid].u != Eg[tid].v);
    // nodeConstraint(fcheck, Eg[tid].u);
    // nodeConstraint(fcheck, Eg[tid].v);

    if (fcheck) {
      m0 = Eg[tid].u;
      m1 = Eg[tid].v;
      tl = Eg[tid].t + delta;
      i_g = tid;
      mi = minfo[level];
      stackio <<= 1;
      if (mi.io >= 0) {
        switch (mi.baseNode) { 
          case 0: base = m0; break;
          case 1: base = m1; break;
          case 2: base = m2; break;
          case 3: base = m3; break;
          case 4: base = m4;
        }
        beg = mi.arrR[base];
        end = mi.arrR[base + 1];
        i_S = mi.arrV;
      } else {
        int base0, base1;
        switch (mi.baseNode) { 
          case 0: base0 = m0; break;
          case 1: base0 = m1; break;
          case 2: base0 = m2; break;
          case 3: base0 = m3; break;
          case 4: base0 = m4;
        }
        switch (mi.constraintNode) { 
          case 0: base1 = m0; break;
          case 1: base1 = m1; break;
          case 2: base1 = m2; break;
          case 3: base1 = m3; break;
          case 4: base1 = m4;

        }
        int beg0 = inEdgesR[base0];
        int end0 = inEdgesR[base0 + 1];
        int beg1 = outEdgesR[base1];
        int end1 = outEdgesR[base1 + 1];
        if (end0 - beg0 < end1 - beg1) {
          mi.io = 0;
          beg = beg0;
          end = end0;
          i_S = inEdgesV;
        } else {
          mi.io = 1;
          mi.constraintNode = mi.baseNode;
          beg = beg1;
          end = end1;
          i_S = outEdgesV;
        }
      }
      stackio |= mi.io ? 1 : 0;

      reduceEarlyIdx(i_S, beg, end, i_g);
    } else {
      end = beg = 0;
    }


    int loopCnt = 0;
    while (true) {
      i_g = numeg;
      int node;
      for (; beg < end; beg++) {
        auto idx = i_S[beg];
        auto eg = Eg[idx];
        if (eg.t > tl) {
          end = beg;
          break;
        }
        node = mi.io ? eg.v : eg.u;
        bool checked = true;
        if (mi.constraintNode < 0) {
          switch (mi.mappedNodes) {
            case 5: checked = (m4 != node);
            case 4: checked = checked && (m3 != node);
            case 3: checked = checked && (m2 != node);
            case 2: checked = checked && (m1 != node) && (m0 != node);
          }
        } else {
          switch (mi.constraintNode) {
            case 4: checked = (m4 == node); break; 
            case 3: checked = (m3 == node); break; 
            case 2: checked = (m2 == node); break;
            case 1: checked = (m1 == node); break;
            case 0: checked = (m0 == node); break;
          }
        }
        // nodeConstraint(checked, node);
        if (checked) {
          i_g = idx; 
          beg++;
          break;
        }
      }

      bool alive = (level || (i_g < numeg));

      if (i_g < numeg) { 
        if (level == numem - 2) { 
          count++;
        } else {
          switch (mi.mappedNodes) {
            case 2: m2 = node;
            case 3: m3 = node; 
            case 4: m4 = node; 
          }
          switch (level) {
            case 0: stackbeg0 = beg; stackend0 = end;
            case 1: stackbeg1 = beg; stackend1 = end;
            case 2: stackbeg2 = beg; stackend2 = end;
            case 3: stackbeg3 = beg; stackend3 = end;
          }

          level++; 

          mi = minfo[level];
          stackio <<= 1;
          if (mi.io >= 0) {
            switch (mi.baseNode) { 
              case 0: base = m0; break;
              case 1: base = m1; break;
              case 2: base = m2; break;
              case 3: base = m3; break;
              case 4: base = m4;
            }
            beg = mi.arrR[base];
            end = mi.arrR[base + 1];
            i_S = mi.arrV;
          } else {
            int base0, base1;
            switch (mi.baseNode) { 
              case 0: base0 = m0; break;
              case 1: base0 = m1; break;
              case 2: base0 = m2; break;
              case 3: base0 = m3; break;
              case 4: base0 = m4;
            }
            switch (mi.constraintNode) { 
              case 0: base1 = m0; break;
              case 1: base1 = m1; break;
              case 2: base1 = m2; break;
              case 3: base1 = m3; break;
              case 4: base1 = m4;
            }
            int beg0 = inEdgesR[base0];
            int end0 = inEdgesR[base0 + 1];
            int beg1 = outEdgesR[base1];
            int end1 = outEdgesR[base1 + 1];
            if (end0 - beg0 < end1 - beg1) {
              mi.io = 0;
              beg = beg0;
              end = end0;
              i_S = inEdgesV;
            } else {
              mi.io = 1;
              mi.constraintNode = mi.baseNode;
              beg = beg1;
              end = end1;
              i_S = outEdgesV;
            }
          }
          stackio |= mi.io ? 1 : 0;

          reduceEarlyIdx(i_S, beg, end, i_g);
        }
      } else { 
        if (level) {
          level--;
          switch (level) {
            case 0: beg = stackbeg0; end = stackend0; break;
            case 1: beg = stackbeg1; end = stackend1; break;
            case 2: beg = stackbeg2; end = stackend2; break;
            case 3: beg = stackbeg3; end = stackend3; break;
          }

          stackio >>= 1;

          mi = minfo[level];
          mi.io = stackio & 1;
          i_S = mi.io ? outEdgesV : inEdgesV;
          if(mi.constraintNode >= 0) mi.constraintNode = mi.io ? mi.baseNode : mi.constraintNode;
        }

      }

      if (loopCnt % 1024 == 0 && *yeild) {
        if (laneid == 0) {
          timeup = clock() + 100000;
        } 
        timeup = __shfl_sync(0xffffffff, timeup, 0);
      }

      if (loopCnt % 64 == 0 && timeup && __shfl_sync(0xffffffff, clock(), 0) > timeup) {
        break;
      }
        
      if (__any_sync(0xffffffff, alive) == 0) break;

      if (laneid == 0) loopCnt++;
      loopCnt = __shfl_sync(0xffffffff, loopCnt, 0);
    }

    if (laneid == 0) tid = atomicAdd(source, 32);
    tid = __shfl_sync(0xffffffff, tid, 0);
  } // end of outer loop

  for (int offset = 16; offset > 0; offset /= 2)
    count += __shfl_down_sync(0xffffffff, count, offset);

  if (laneid == 0) atomicAdd(gcount, count);

  atomicAdd(yeild, 1); 
  
  if ((level || (beg < end))) {
    dumpContext(
      stackbeg0, stackbeg1, stackbeg2, stackbeg3,
      stackend0, stackend1, stackend2, stackend3,
      m0, m1, m2, m3, m4,
      
  
      tl,
  
      stackio, level, beg, end, i_S,
  
      Eg,
      inEdgesV,
      outEdgesV,
  
      offload,
      offtop,
      offload_width
    );
  }
}

// --------------------------------- 

static unsigned long long TMotifMatchingGPUImpl(
                          int numBlocksA, int numBlocksB, int sizeBlock,
                          int work, int delta,
                          // graph
                          const TemporalEdge *Eg, int numeg,
                          const int *inEdgesV, const int *inEdgesR, const int *outEdgesV, const int *outEdgesR,
                          const int *nodeFeature,
                          // motif
                          const MotifEdgeInfoV1 *minfo, int numem,
                        
                          // runtime mem
                          int *yeild,
                          int *source,
                          TContext *offload,
                          int *offtop,
                          int *offtopn,
                          int *offload_width,

                          unsigned long long *gcount 
                          ) {
  cudaError_t err;
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);
  // inspector_u3b <<< 1, 32, 0, stream2 >>> (work);
  MotifMatching_dispatch <<< numBlocksA, sizeBlock, 0, stream1 >>>(
      work, delta,
      Eg, numeg,
      inEdgesV, inEdgesR, outEdgesV, outEdgesR,
      nodeFeature,

      minfo, numem,

      yeild,
      source,
      offload,
      offtop,
      offtopn,
      offload_width,

      gcount
  );
  err = cudaGetLastError();

  if ( err != cudaSuccess ) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));       
      exit(-1);
  }

  int chunk_offset_h[32];
  int *chunk_offset;
  gpuErrchk(cudaMalloc(&chunk_offset, sizeof(int) * 32));
  gpuErrchk(cudaMemset(chunk_offset, 0, sizeof(int) * 32));

  int offload_cnt = 0;
  gpuErrchk(cudaMemcpy(&offload_cnt, offtop, sizeof(int), cudaMemcpyDeviceToHost));

  while (offload_cnt > 0) {
    int t = 0;
    gpuErrchk(cudaMemcpy(source, &t, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(yeild, &t, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(offtopn, &t, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(offtop, &offload_cnt, sizeof(int), cudaMemcpyHostToDevice));
    

    // prefix sum
    thrust::device_ptr<int> dev_ptr(offload_width);

    auto sump = thrust::inclusive_scan(dev_ptr, dev_ptr + offload_cnt, dev_ptr);
    auto sum = *(sump - 1);
    for (int i = 0; i < 32; i++) {
      chunk_offset_h[i] = i * (sum / 32 + (sum % 32 != 0));
    }
    gpuErrchk(cudaMemcpy(chunk_offset, &chunk_offset_h, sizeof(int) * 32, cudaMemcpyHostToDevice));

    MotifMatching_Expand <<< numBlocksB, sizeBlock >>> (sum,
      Eg, numeg,
      inEdgesV, inEdgesR, outEdgesV, outEdgesR,
      nodeFeature,

      minfo, numem,

      yeild,
      source,
      offload,
      offtop,
      offtopn,
      offload_width,

      chunk_offset,

      gcount
    );

    err = cudaGetLastError();
    
    if ( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    gpuErrchk(cudaMemcpy(&offload_cnt, offtopn, sizeof(int), cudaMemcpyDeviceToHost));
  }


  cudaDeviceSynchronize();  

  gpuErrchk(cudaFree(chunk_offset));

  unsigned long long count_h;
  gpuErrchk(cudaMemcpy(&count_h, gcount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  return count_h;
}

struct GPUWorkerU3B : public GPUWorker {
  // Execution Mem
  unsigned long long *count_d;

  int *yeild_d;
  int *source_d;
  TContext *offload_d;
  int *offtop, *offtopn;
  int *offload_width_d;

  GPUWorkerU3B(int gpu, int sizeBlock = 96);
  unsigned long long run() override;
  virtual void take(MineJob &job) override;
  void update_job() override;
  ~GPUWorkerU3B() override;
};

GPUWorkerU3B::GPUWorkerU3B(int gpu, int sizeBlock) : GPUWorker("U3B", gpu, sizeBlock) {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaMalloc(&count_d, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(count_d, 0, sizeof(unsigned long long)));

  gpuErrchk(cudaMalloc(&yeild_d, sizeof(int)));
  gpuErrchk(cudaMemset(yeild_d, 0, sizeof(int)));
  gpuErrchk(cudaMalloc(&source_d, sizeof(int)));
  gpuErrchk(cudaMemset(source_d, 0, sizeof(int)));
  gpuErrchk(cudaMalloc(&offload_d, 2092 * 96 * sizeof(TContext)));
  gpuErrchk(cudaMemset(offload_d, 0, 2092 * 96 * sizeof(TContext)));
  gpuErrchk(cudaMalloc(&offtop, sizeof(int)));
  gpuErrchk(cudaMalloc(&offtopn, sizeof(int)));
  gpuErrchk(cudaMemset(offtop, 0, sizeof(int)));
  gpuErrchk(cudaMemset(offtopn, 0, sizeof(int)));
  gpuErrchk(cudaMalloc(&offload_width_d, 2092 * 96 * sizeof(int)));
  gpuErrchk(cudaMemset(offload_width_d, 0, 2092 * 96 * sizeof(int)));
}

GPUWorkerU3B::~GPUWorkerU3B() {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaFree(count_d));
  gpuErrchk(cudaFree(yeild_d));
  gpuErrchk(cudaFree(source_d));
  gpuErrchk(cudaFree(offload_d));
  gpuErrchk(cudaFree(offtop));
  gpuErrchk(cudaFree(offtopn));
  gpuErrchk(cudaFree(offload_width_d));
}

unsigned long long GPUWorkerU3B::run() {
  gpuErrchk(cudaSetDevice(gpu_));
  // Get device properties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_);

  int maxBlocksPerSM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, MotifMatching_dispatch, sizeBlock_, 0);
  auto numBlocksA = maxBlocksPerSM * deviceProp.multiProcessorCount;

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, MotifMatching_Expand, sizeBlock_, 0);
  auto numBlocksB = maxBlocksPerSM * deviceProp.multiProcessorCount;

  auto &dd = job_->data;

  count_ = TMotifMatchingGPUImpl(
      numBlocksA, numBlocksB, sizeBlock_, 
      job_->end, job_->delta,

      dd->Eg_d, dd->graphNumEdges,
      dd->inEdgesV_d, dd->inEdgesR_d, dd->outEdgesV_d, dd->outEdgesR_d,
      dd->nodefeatures_d,

      dd->minfo(), dd->motifNumEdges,

      yeild_d,
      source_d,
      offload_d,
      offtop,
      offtopn,
      offload_width_d,

      count_d
  );
  return count_;
}


void GPUWorkerU3B::update_job() {
  job_->beg = job_->end;
}

void GPUWorkerU3B::take(MineJob &job) {
  gpuErrchk(cudaSetDevice(gpu_));
  GPUWorker::take(job);

  gpuErrchk(cudaMemset(yeild_d, 0, sizeof(int)));
  gpuErrchk(cudaMemset(offtop, 0, sizeof(int)));
  gpuErrchk(cudaMemset(offtopn, 0, sizeof(int)));
  gpuErrchk(cudaMemcpy(source_d, &(job_->beg), sizeof(int), cudaMemcpyHostToDevice));
}

extern "C" {
  GPUWorker *getWorker(int gpu);
}

GPUWorker *getWorker(int gpu) {
  return new GPUWorkerU3B(gpu);
}