#include <stdio.h>
#include <cuda.h>
#include "helpers.cuh"

#include "data.h"

using namespace corelib;
using namespace corelib::data;

/*
 * u0 version implement search result memorization.
 */

// ---------- GPU kernels ---------- 
// return the first position that is >= i_g
__device__ static int binarySearchGeq(const int *i_S, int beg, int end, int i_g) {
    int p = 0, ans = 0;
    while (beg < end) {
        p = (beg + end) / 2;
        if (i_S[p] >= i_g) {
            end = p;
        } else {
            ans = p + 1;
            beg = p + 1;
        }
    }
    return ans;
}


__device__ static void reduceEarlyIdx(const int *i_S, int &beg, int end, int i_g) {
    if (i_S[end - 1] < i_g) {
        beg = end;
        return;
    } else if (i_S[beg] >= i_g) {
        return;
    }

    // binary search the first value >= i_g, at this point i_S has at least 2 elements
    beg = binarySearchGeq(i_S, beg, end, i_g);
}

__device__ void TMotifMatching_genBegEnd(const TemporalEdge *Eg, int numeg, int i_g,
                                    const TemporalEdge *Em, int numnm, int i_m,
                                    int *MapMg,
                                    const int *inEdgesV, const int *inEdgesR, 
                                    const int *outEdgesV, const int *outEdgesR,
                                    int tl, int &beg, int &end, const int *&i_S
                                    ) {
    auto em = Em[i_m];
    int umg = MapMg[em.u], vmg = MapMg[em.v];

    if (umg >= 0 && vmg >= 0) {
        int ibeg = inEdgesR[vmg], iend = inEdgesR[vmg + 1];
        int obeg = outEdgesR[umg], oend = outEdgesR[umg + 1];
        if (iend - ibeg < oend - obeg) {
            i_S = inEdgesV;
            beg = ibeg;
            end = iend;
        } else {
            i_S = outEdgesV;
            beg = obeg;
            end = oend;
        }
    } else if (umg >= 0) {
        int obeg = outEdgesR[umg], oend = outEdgesR[umg + 1];
        i_S = outEdgesV;
        beg = obeg;
        end = oend;
    } else if (vmg >= 0) {
        int ibeg = inEdgesR[vmg], iend = inEdgesR[vmg + 1];
        i_S = inEdgesV;
        beg = ibeg;
        end = iend;
    }

    if (end == beg) return;
    if (i_S) reduceEarlyIdx(i_S, beg, end, i_g);
}

__global__ static void MotifMatching_dispatch( int work, int delta,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR,
  const int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,

  // motif
  const TemporalEdge *Em, int numem, int numnm,

  // runtime
  unsigned long long *gcount
  ) {
  // ** SEC. 1, First Edge & Initialization
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  auto laneid = threadIdx.x % 32;
  unsigned long long count = 0;

  int MapMg[5], edgeCount[5], Estackbeg[5], Estackend[5];
  const int *Estackis[4];
  MapMg[0] = -1;
  MapMg[1] = -1;
  MapMg[2] = -1;
  MapMg[3] = -1;
  MapMg[4] = -1;

  edgeCount[0] = 0;
  edgeCount[1] = 0;
  edgeCount[2] = 0;
  edgeCount[3] = 0;
  edgeCount[4] = 0;

  int stp = 0;
  int i_g = tid + 1;
  int i_m = 1;

  auto em = Em[i_m];
  const int *i_S = nullptr; // nullptr implies from i_g...numeg
  int beg = i_g, end = numeg;
  int tl;

  // check if loop is perserved
  bool perserve_loop = (Em[0].u == Em[0].v && Eg[tid].u == Eg[tid].v) || 
              (Em[0].u != Em[0].v && Eg[tid].u != Eg[tid].v);
  if (!perserve_loop) goto EXITKERNEL;
  if (tid >= work) goto EXITKERNEL;

  // map the first edge; 
  MapMg[Em[0].u] = Eg[tid].u;
  MapMg[Em[0].v] = Eg[tid].v;
  edgeCount[Em[0].u]++;
  edgeCount[Em[0].v]++;
  // set the time limit
  tl = Eg[tid].t + delta;


  // ** SEC. 2, Tree exploration
  TMotifMatching_genBegEnd(Eg, numeg, i_g, Em, numnm, i_m, MapMg, inEdgesV, inEdgesR, outEdgesV, outEdgesR, tl, beg, end, i_S);

  while (true) {
    int umg = MapMg[em.u], vmg = MapMg[em.v];
    i_g = numeg;
    for (; beg < end; beg++) {
      auto idx = i_S ? i_S[beg] : beg;
      auto eg = Eg[idx];
      if (eg.t > tl) {
        end = beg;
        break;
      }
      bool perserve_loop = (em.u == em.v && eg.u == eg.v) || (em.u != em.v && eg.u != eg.v);
      bool uIsMapped = false, vIsMapped = false;
      for (int i = 0; i < numnm; i++) {
        if (MapMg[i] == eg.u) uIsMapped = true;
        if (MapMg[i] == eg.v) vIsMapped = true;
      }
      bool canMapu = eg.u == umg || (umg == -1 && !uIsMapped);
      bool canMapv = eg.v == vmg || (vmg == -1 && !vIsMapped);
      if (perserve_loop && canMapu && canMapv) {
        i_g = idx; 
        beg++;
        break;
      }
    }

    if (i_g < numeg) { 
      auto eg = Eg[i_g];
      if (i_m == numem - 1) { // last edge
        count++;
      } else {
        MapMg[em.u] = eg.u;
        MapMg[em.v] = eg.v;
        Estackbeg[stp] = beg;
        Estackend[stp] = end;
        Estackis[stp] = i_S;
        stp++;
        edgeCount[em.u]++;  // mark the edge inuse;
        edgeCount[em.v]++;

        i_m++; // step forward an edge in M, update em
        em = Em[i_m];
        i_g++; // finish this edge
        TMotifMatching_genBegEnd(Eg, numeg, i_g, Em, numnm, i_m, MapMg, inEdgesV, inEdgesR, outEdgesV, outEdgesR, tl, beg, end, i_S);
      }
    } else { // Trace back if current edge is "too old"
      if (stp) {
        stp--;
        beg = Estackbeg[stp]; 
        end = Estackend[stp]; 
        i_S = Estackis[stp];
        i_m--;
        em = Em[i_m];
        if (--edgeCount[em.u] == 0) {
          MapMg[em.u] = -1;
        }
        if (--edgeCount[em.v] == 0) {
          MapMg[em.v] = -1;
        }
      } else {
        goto EXITKERNEL;
      }
    }
  }

  // SEC. 3 Output
EXITKERNEL:
  for (int offset = 16; offset > 0; offset /= 2)
    count += __shfl_down_sync(0xffffffff, count, offset);

  if (laneid == 0) atomicAdd(gcount, count);
}

// --------------------------------- 

static unsigned long long TMotifMatchingGPUImpl( int numBlocks, int sizeBlock, int work, int delta,
  // graph
  const TemporalEdge *Eg, int numeg,
  const int *inEdgesV, const int *inEdgesR, const int *outEdgesV, const int *outEdgesR,
  const int *nodeFeature,
  // motif
  const TemporalEdge *Em, int numem, int numnm,

  // runtime mem
  unsigned long long *gcount
                          ) {
    // start the GPU kernel
    MotifMatching_dispatch <<< numBlocks, sizeBlock >>>(
      work, delta,
      Eg, numeg,
      inEdgesV, inEdgesR, outEdgesV, outEdgesR,
      nodeFeature,

      Em, numem, numnm,

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

struct GPUWorkerU0 : public GPUWorker {
  // Execution Mem
  unsigned long long *count_d;

  GPUWorkerU0(int gpu, int sizeBlock = 96);
  unsigned long long run() override;
  void update_job() override;
  ~GPUWorkerU0() override;
};

GPUWorkerU0::GPUWorkerU0(int gpu, int sizeBlock) : GPUWorker("U0", gpu, sizeBlock) {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaMalloc(&count_d, sizeof(unsigned long long)));
  gpuErrchk(cudaMemset(count_d, 0, sizeof(unsigned long long)));
}

GPUWorkerU0::~GPUWorkerU0() {
  gpuErrchk(cudaSetDevice(gpu_));
  gpuErrchk(cudaFree(count_d));
}

unsigned long long GPUWorkerU0::run() {
  gpuErrchk(cudaSetDevice(gpu_));
  if (!job_) {
    throw std::runtime_error("run a GPU worker without a job");
  }

  gpuErrchk(cudaSetDevice(this->gpu_));

  auto &dd = job_->data;

  count_ = TMotifMatchingGPUImpl(
      numBlocks(), sizeBlock_, 
      job_->end, job_->delta,

      dd->Eg_d, dd->graphNumEdges,
      dd->inEdgesV_d, dd->inEdgesR_d, dd->outEdgesV_d, dd->outEdgesR_d,
      dd->nodefeatures_d,

      dd->Em_d(), dd->motifNumEdges, dd->motifNumVertices,

      count_d
  );
  return count_;
}

void GPUWorkerU0::update_job() {
  job_->beg = job_->end;
}

extern "C" {
  GPUWorker *getWorker(int gpu);
}

GPUWorker *getWorker(int gpu) {
  return new GPUWorkerU0(gpu);
}