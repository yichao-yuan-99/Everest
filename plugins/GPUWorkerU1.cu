#include <stdio.h>
#include <cuda.h>
#include "helpers.cuh"

#include "data.h"

using namespace corelib;
using namespace corelib::data;

/*
 * U1 implements motif pre-decoding, which is a subpart of the load balancing optimization
 */

// ---------- GPU kernels ---------- 
// return the first position that is >= i_g

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

__global__ static void MotifMatching_dispatch(int work, int delta,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR,
  const int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,

  // motif
  const MotifEdgeInfoV1 *minfo, int numem,

  // runtime
  unsigned long long *gcount
  ) {
  // ** SEC. 1, First Edge & Initialization
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto laneid = threadIdx.x % 32;

  int Estackbeg[4], Estackend[4], MapMg[5];
  unsigned char stackio = 0;

  int tl;
  unsigned long long count = 0;

  auto i_g = tid;
  auto level = 0;

  int beg, end, base;

  auto mi = minfo[level];
  const int *i_S;


  if (Eg[tid].u == Eg[tid].v) goto EXITKERNEL;
  if (tid >= work) goto EXITKERNEL;

  MapMg[0] = Eg[tid].u;
  MapMg[1] = Eg[tid].v;

  tl = Eg[tid].t + delta;

  if (mi.io >= 0) {
    base = MapMg[mi.baseNode];
    beg = mi.arrR[base];
    end = mi.arrR[base + 1];
    i_S = mi.arrV;
  } else {
    int base0 = MapMg[mi.baseNode];
    int base1 = MapMg[mi.constraintNode];
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
  stackio <<= 1;
  stackio |= mi.io ? 1 : 0;

  reduceEarlyIdx(i_S, beg, end, i_g);

  // ** SEC. 2, Tree exploration
  while (true) {
    i_g = numeg;
    int node;
    for (; beg < end; beg++) {
      auto idx = i_S[beg];
      auto eg = Eg[idx];
      if (eg.t > tl) break;
      int matchid = -1;
      node = mi.io ? eg.v : eg.u;
      for (int i = mi.mappedNodes - 1; i >= 0; i--) {
        if (MapMg[i] == node) {
          matchid = i;
          break;
        }
      }
      bool checked = (mi.constraintNode < 0 && matchid <0) || (mi.constraintNode >= 0 && matchid == mi.constraintNode);
      if (checked) {
        i_g = idx; 
        beg++;
        break;
      }
    }

    if (i_g < numeg) { 
      if (level == numem - 2) {
        count++;
      } else {
        if (mi.constraintNode < 0) MapMg[-mi.constraintNode] = node;
        Estackbeg[level] = beg;
        Estackend[level] = end;
        level++;
        stackio <<= 1;

        mi = minfo[level];
        if (mi.io >= 0) {
          base = MapMg[mi.baseNode];
          beg = mi.arrR[base];
          end = mi.arrR[base + 1];
          i_S = mi.arrV;
        } else {
          int base0 = MapMg[mi.baseNode];
          int base1 = MapMg[mi.constraintNode];
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
        beg = Estackbeg[level]; 
        end = Estackend[level];
        stackio >>= 1;

        mi = minfo[level];
        mi.io = stackio & 1;
        i_S = mi.io ? outEdgesV : inEdgesV;
        if(mi.constraintNode >= 0) mi.constraintNode = mi.io ? mi.baseNode : mi.constraintNode;
      } else {
        break;
      }
    }
  }

EXITKERNEL:
  // SEC. 3 Output
  for (int offset = 16; offset > 0; offset /= 2)
    count += __shfl_down_sync(0xffffffff, count, offset);

  if (laneid == 0) atomicAdd(gcount, count);
}

// --------------------------------- 

static unsigned long long TMotifMatchingGPUImpl( int numBlocks, int sizeBlock, int work, int delta,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR, const  int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,
  // motif
  const MotifEdgeInfoV1 *minfo, int numem,

  // runtime mem
  unsigned long long *gcount
                          ) {
    // start the GPU kernel
    MotifMatching_dispatch <<< numBlocks, sizeBlock >>>(
      work, delta,
      Eg, numeg,
      inEdgesV, inEdgesR, outEdgesV, outEdgesR,
      nodeFeature,

      minfo, numem,

      gcount
    );
    cudaError_t err = cudaGetLastError();

    if ( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
        exit(-1);
    }

    // get the result back
    unsigned long long count_h;
    gpuErrchk(cudaMemcpy(&count_h, gcount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    return count_h;
}

struct GPUWorkerU1 : public GPUWorker {
  // Execution Mem
  unsigned long long *count_d;

  GPUWorkerU1(int gpu, int sizeBlock = 96);
  unsigned long long run() override;
  void update_job() override;
  ~GPUWorkerU1() override;
};

GPUWorkerU1::GPUWorkerU1(int gpu, int sizeBlock) : GPUWorker("U1", gpu, sizeBlock) {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaMalloc(&count_d, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(count_d, 0, sizeof(unsigned long long)));
}

GPUWorkerU1::~GPUWorkerU1() {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaFree(count_d));
}

unsigned long long GPUWorkerU1::run() {
  gpuErrchk(cudaSetDevice(gpu_));

  auto &dd = job_->data;

  count_ = TMotifMatchingGPUImpl(
      numBlocks(), sizeBlock_, 
      job_->end, job_->delta,

      dd->Eg_d, dd->graphNumEdges,
      dd->inEdgesV_d, dd->inEdgesR_d, dd->outEdgesV_d, dd->outEdgesR_d,
      dd->nodefeatures_d,

      dd->minfo(), dd->motifNumEdges,

      count_d
  );
  return count_;
}

void GPUWorkerU1::update_job() {
  job_->beg = job_->end;
}

extern "C" {
  GPUWorker *getWorker(int gpu);
}

GPUWorker *getWorker(int gpu) {
  return new GPUWorkerU1(gpu);
}