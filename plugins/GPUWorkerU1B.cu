#include <stdio.h>
#include <cuda.h>
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
  unsigned long long *gcount
  ) {
  
  auto laneid = threadIdx.x % 32;

  auto tid = blockDim.x * blockIdx.x + threadIdx.x;

  int stackbeg0, stackbeg1, stackbeg2, stackbeg3;
  int stackend0, stackend1, stackend2, stackend3;
  int m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;
  
  unsigned char stackio = 0;
  int tl;
  unsigned long long count = 0;
  auto i_g = tid;
  auto level = 0;
  int beg = 0, end = 0, base;
  MotifEdgeInfoV1 mi;
  const int *i_S = nullptr;

  bool fcheck = tid < work && (Eg[tid].u != Eg[tid].v);

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
    end = beg;
  }


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
      } else {
        break;
      }
    }

  }

  for (int offset = 16; offset > 0; offset /= 2)
    count += __shfl_down_sync(0xffffffff, count, offset);

  if (laneid == 0) atomicAdd(gcount, count);
}

// --------------------------------- 

static unsigned long long TMotifMatchingGPUImpl( 
  int numBlocks, int sizeBlock, 
  
  int work, int delta,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR, const int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,
  // motif
  const MotifEdgeInfoV1 *minfo, int numem,

  // runtime mem
  unsigned long long *gcount
                          ) {
  cudaError_t err;
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);

  // start the GPU kernel
  MotifMatching_dispatch <<< numBlocks, sizeBlock >>>(
    work, delta,
    Eg, numeg,
    inEdgesV, inEdgesR, outEdgesV, outEdgesR,
    nodeFeature,

    minfo, numem,

    gcount
  );
  err = cudaGetLastError();

  if ( err != cudaSuccess ) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));       
      exit(-1);
  }

  // get the result back
  unsigned long long count_h;
  gpuErrchk(cudaMemcpy(&count_h, gcount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  return count_h;

}

struct GPUWorkerU1B : public GPUWorker {
  // Execution Mem
  unsigned long long *count_d;

  GPUWorkerU1B(int gpu, int sizeBlock = 96);
  unsigned long long run() override;
  void update_job() override;
  ~GPUWorkerU1B() override;
};

GPUWorkerU1B::GPUWorkerU1B(int gpu, int sizeBlock) : GPUWorker("U1B", gpu, sizeBlock) {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaMalloc(&count_d, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(count_d, 0, sizeof(unsigned long long)));
}

GPUWorkerU1B::~GPUWorkerU1B() {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaFree(count_d));
}

unsigned long long GPUWorkerU1B::run() {
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

void GPUWorkerU1B::update_job() {
  job_->beg = job_->end;
}

extern "C" {
  GPUWorker *getWorker(int gpu);
}

GPUWorker *getWorker(int gpu) {
  return new GPUWorkerU1B(gpu);
}