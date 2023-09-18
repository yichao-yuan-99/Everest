#include "data.h"
#include "helpers.cuh"
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>

#include <thread>
#include <cassert>
#include <algorithm>

namespace corelib::data {

size_t calcTotalBytes(size_t E, size_t V) {
  return (sizeof(TemporalEdge) + sizeof(int) * 3) * E + (sizeof(int) * 2) * (V + 1) + (sizeof(int) * V);
}

MappedFileSourceMemory::MappedFileSourceMemory(std::filesystem::path pathToFile)
  : mfile_(pathToFile) {
  auto flag = cudaHostRegisterDefault | cudaHostRegisterReadOnly;
  gpuErrchk(cudaHostRegister((void *) data(), size(), flag));
}

MappedFileSourceMemory::~MappedFileSourceMemory() {
  gpuErrchk(cudaHostUnregister((void *) data()));
}

MappedFileSinkMemory::MappedFileSinkMemory(std::filesystem::path pathToFile, size_t size) {
  boost::iostreams::mapped_file_params params;
  params.new_file_size = size;
  params.flags = boost::iostreams::mapped_file::mapmode::readwrite;
  params.path = pathToFile;
  mfile_.open(params);

  auto flag = cudaHostRegisterDefault;
  gpuErrchk(cudaHostRegister((void *) data(), this->size(), flag));
}

 MappedFileSinkMemory::MappedFileSinkMemory(std::filesystem::path pathToFile) :
  mfile_(pathToFile) {
  auto flag = cudaHostRegisterDefault;
  gpuErrchk(cudaHostRegister((void *) data(), this->size(), flag));
}

MappedFileSinkMemory::~MappedFileSinkMemory() {
  gpuErrchk(cudaHostUnregister((void *) data()));
}

void FeatureLoader::createFeatureCache() {
  size_t bytes = eloader_->edgeListLength() * sizeof(int);
  MappedFileSourceMemory rands(pathToRands_);
  MappedFileSinkMemory FE(cacheFEPath(), bytes);
  memcpy(FE.data(), rands.data(), bytes);

  // [NOTE] potential bug for egelistLength > INT_MAX due to thrust
  size_t length = eloader_->edgeListLength();
  const int *Up = (const int *) eloader_->getUPtr()->data(); 
  const int *Vp = (const int *) eloader_->getVPtr()->data();
  thrust::device_vector<int> IDs(length * 2);
  thrust::copy_n(Up, length, IDs.begin());
  thrust::copy_n(Vp, length, IDs.begin() + length);
  thrust::sort(IDs.begin(), IDs.end());
  auto ed = thrust::unique(IDs.begin(), IDs.end());
  size_t numVertices = ed - IDs.begin();

  bytes = numVertices * sizeof(int);
  MappedFileSinkMemory FVM(cacheFVMPath(), bytes);
  MappedFileSinkMemory FVV(cacheFVVPath(), bytes);
  thrust::copy(IDs.begin(), IDs.begin() + numVertices, (int *) FVM.data());
  memcpy(FVV.data(), rands.data(), bytes);
}

HostGraphData GraphDataLoader::loadGraphData(size_t beg, size_t end) const {
  auto st = std::chrono::high_resolution_clock::now();
  auto ed = st;

  HostGraphData res;
  res.name = cacheGraphDirPath(beg, end).filename();
  res.beg = beg;
  res.end = end;
  auto cpath = cacheGraphDirPath(beg, end);
  std::filesystem::create_directory(cpath); 
  res.Eg_h.reset(new MappedFileSinkMemory(cpath / "Eg"));
  res.inEdgesV_h.reset(new MappedFileSinkMemory(cpath / "inEdgesV"));
  res.outEdgesV_h.reset(new MappedFileSinkMemory(cpath / "outEdgesV"));
  res.inEdgesR_h.reset(new MappedFileSinkMemory(cpath / "inEdgesR"));
  res.outEdgesR_h.reset(new MappedFileSinkMemory(cpath / "outEdgesR"));
  res.verticesFeatures.reset(new MappedFileSinkMemory(cpath / "verticesFeatures"));
  res.edgeFeatures.reset(new MappedFileSinkMemory(cpath / "edgeFeatures"));
  res.Euid_h.reset(new MappedFileSinkMemory(cpath / "Euid"));

  res.numEdges = res.Eg_h->size() / sizeof(int) / 3;
  res.numVertices = res.inEdgesR_h->size() / sizeof(int) - 1;
  res.totalBytes = calcTotalBytes(res.numEdges, res.numVertices);

  ed = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(ed - st).count();
  st = ed;
  fmt::print("# [GraphDataLoader] {} ms to load from cache for {} MB, range ({},{})\n", time, res.totalBytes / (1024 * 1024), beg, end);
  return res;
}

bool GraphDataLoader::isCached(size_t beg, size_t end) const {
  return std::filesystem::exists(cacheGraphDirPath(beg, end));
}

GraphDataLoader::GraphDataLoader(const EdgeListLoader *eloader, const FeatureLoader *floader) 
 : CacheDirectoryAccess(eloader->graphPath_), stem_(eloader->name_),
 UPtr_(eloader->getUPtr()), VPtr_(eloader->getVPtr()), 
 TPtr_(eloader->getTPtr()), FEPtr_(floader->getFEPtr()),
 FVVPtr_(floader->getFVVPtr()), FVMPtr_(floader->getFVMPtr()),
 EuidPtr_(eloader->getIDSPtr()),
 edgeListLength_(eloader->edgeListLength()), 
 verticesFeatureLength_(floader->verticesFeaturesLength()) {}

HostGraphData GraphDataLoader::createGraphData(size_t beg, size_t end, bool cache) const {
  if (end == 0) end = edgeListLength_;
  if (cache && isCached(beg, end)) return loadGraphData(beg, end);

  auto st = std::chrono::high_resolution_clock::now();
  auto ed = st;
  time_t time;
  // assume that alway cache == true first
  HostGraphData res;
  res.name = cacheGraphDirPath(beg, end).filename();
  res.beg = beg;
  res.end = end;
  res.numEdges = end - beg;

  thrust::device_vector<int> VerticesVec_d((end - beg) * 2);
  thrust::device_vector<int> OutIn_d;
  thrust::device_vector<int> OutR_d, InR_d;
  thrust::device_vector<int> VerticesFeatures_d;
  {
    // -------------------- Rename ----------------------
    int *Up = (int *) UPtr_->data(), *Vp = (int *) VPtr_->data();
    int *Ubeg = Up + beg, *Uend = Up + end;
    int *Vbeg = Vp + beg, *Vend = Vp + end;
    thrust::device_vector<int> OccurIdVec_d((end - beg) * 2);
    auto t = thrust::copy(Ubeg, Uend, VerticesVec_d.begin());
    thrust::copy(Vbeg, Vend, t);
    thrust::sequence(OccurIdVec_d.begin(), OccurIdVec_d.begin() + (end - beg), 0);
    thrust::sequence(OccurIdVec_d.begin() + (end - beg), OccurIdVec_d.end(),  0);
    
    // find first occurence id of a vertex
    thrust::stable_sort_by_key(VerticesVec_d.begin(), VerticesVec_d.end(), OccurIdVec_d.begin());
    thrust::device_vector<int> uniqueVertVec_d(end - beg + 1);
    auto VertFirstOccur_d = uniqueVertVec_d;
    auto P = thrust::reduce_by_key(
      VerticesVec_d.begin(), VerticesVec_d.end(),
      OccurIdVec_d.begin(), uniqueVertVec_d.begin(), VertFirstOccur_d.begin(),
      thrust::equal_to<int>(), thrust::minimum<int>()
    );

    res.numVertices = P.first - uniqueVertVec_d.begin();

    // assign each vertex a new index (create mapping)
    thrust::copy(Ubeg, Uend, VerticesVec_d.begin());
    thrust::sequence(OccurIdVec_d.begin(), OccurIdVec_d.begin() + (end - beg), 0);
    thrust::sort_by_key(
      VerticesVec_d.begin(), VerticesVec_d.begin() + (end - beg),
      OccurIdVec_d.begin() 
    );
    thrust::device_vector<bool> dstBool_d(res.numVertices);
    thrust::binary_search(
      VerticesVec_d.begin(), VerticesVec_d.begin() + (end - beg),
      uniqueVertVec_d.begin(), uniqueVertVec_d.begin() + (res.numVertices),
      dstBool_d.begin()
    );
    thrust::transform(dstBool_d.begin(), dstBool_d.end(), 
      dstBool_d.begin(), thrust::logical_not<bool>());

    thrust::device_vector<int> dstFirst_d(res.numVertices);
    thrust::lower_bound(
      VerticesVec_d.begin(), VerticesVec_d.begin() + (end - beg),
      uniqueVertVec_d.begin(), uniqueVertVec_d.begin() + (res.numVertices),
      dstFirst_d.begin()
    );
    thrust::gather(dstFirst_d.begin(), dstFirst_d.end(), OccurIdVec_d.begin(), dstFirst_d.begin());

    // first compare global min, then compare if exist in the source
    // then compare the position in the source
    auto zipItBeg = thrust::make_zip_iterator(
      thrust::make_tuple(VertFirstOccur_d.begin(), dstBool_d.begin(), dstFirst_d.begin())
    );
    auto zipItEnd = zipItBeg + res.numVertices;

    thrust::stable_sort_by_key(zipItBeg, zipItEnd, uniqueVertVec_d.begin());
    thrust::sequence(VertFirstOccur_d.begin(), VertFirstOccur_d.begin() + res.numVertices, 0);

    // (uniqueVertVec, VertFirstOccur) is a mapping from given ids to 0-indexed ids.
    // collect vertices features
    VerticesFeatures_d.resize(res.numVertices);
    {
      thrust::device_vector<int> tmpFIdx(res.numVertices); 
      const int *FVVp = (const int *) FVVPtr_->data();
      const int *FVMp= (const int *) FVMPtr_->data();
      thrust::device_vector<int> FVV(FVVp, FVVp + verticesFeatureLength_);
      thrust::device_vector<int> FVM(FVMp, FVMp + verticesFeatureLength_);
      // for each uniqueVert, map it to its feature
      thrust::lower_bound(
        FVM.begin(), FVM.end(), uniqueVertVec_d.begin(), uniqueVertVec_d.begin() + res.numVertices,
        tmpFIdx.begin()
      );
      thrust::gather(
        tmpFIdx.begin(), tmpFIdx.end(),
        FVV.begin(), VerticesFeatures_d.begin()
      );
    }

    // reorder
    thrust::stable_sort_by_key(
      uniqueVertVec_d.begin(), uniqueVertVec_d.begin() + res.numVertices, 
      VertFirstOccur_d.begin()
    );

    // do mapping
    t = thrust::copy(Ubeg, Uend, VerticesVec_d.begin());
    thrust::copy(Vbeg, Vend, t);
    thrust::lower_bound(
      uniqueVertVec_d.begin(), uniqueVertVec_d.begin() + res.numVertices, 
      VerticesVec_d.begin(), VerticesVec_d.end(),
      OccurIdVec_d.begin()
    );
    thrust::gather(
      OccurIdVec_d.begin(), OccurIdVec_d.end(), 
      VertFirstOccur_d.begin(), VerticesVec_d.begin()
    );
    // -------------------- out/in ----------------------
    auto dup = VerticesVec_d;
    thrust::sequence(OccurIdVec_d.begin(), OccurIdVec_d.begin() + (end - beg), 0);
    thrust::sequence(OccurIdVec_d.begin() + (end - beg), OccurIdVec_d.end(),  0);

    thrust::stable_sort_by_key(
      dup.begin(), dup.begin() + (end - beg), OccurIdVec_d.begin()
    );
    thrust::stable_sort_by_key(
      dup.begin() + (end - beg), dup.end(), OccurIdVec_d.begin() + (end - beg)
    );
    OutIn_d = OccurIdVec_d; // make copy
    // ------------------- OutR -------------------------
    auto &tmp1 = uniqueVertVec_d;
    auto &tmp2 = VertFirstOccur_d;
    auto &tmp3 = dstFirst_d;
    P = thrust::reduce_by_key(
      dup.begin(), dup.begin() + (end - beg),
      thrust::constant_iterator<int>(1), 
      tmp1.begin(), tmp2.begin()
    );
    OutR_d.resize(res.numVertices + 1);
    thrust::fill(tmp3.begin(), tmp3.end(), 0);
    thrust::scatter( tmp2.begin(), P.second, tmp1.begin(), tmp3.begin());
    thrust::inclusive_scan(tmp3.begin(), tmp3.end(), OutR_d.begin() + 1);
    // ------------------ InR ---------------------------
    P = thrust::reduce_by_key(
      dup.begin() + (end - beg), dup.end(),
      thrust::constant_iterator<int>(1), 
      tmp1.begin(), tmp2.begin()
    );
    InR_d.resize(res.numVertices + 1);
    thrust::fill(tmp3.begin(), tmp3.end(), 0);
    thrust::scatter( tmp2.begin(), P.second, tmp1.begin(), tmp3.begin());
    thrust::inclusive_scan(tmp3.begin(), tmp3.end(), InR_d.begin() + 1);
  }
  ed = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(ed - st).count();
  st = ed;
  fmt::print("# [GraphDataLoader] create on GPU {} ms\n", time);

  // allocate memory
  
  if (cache) {
    auto cpath = cacheGraphDirPath(beg, end);
    std::filesystem::create_directory(cpath); 
    res.Eg_h.reset(new MappedFileSinkMemory(cpath / "Eg", res.numEdges * sizeof(int) * 3));
    res.inEdgesV_h.reset(new MappedFileSinkMemory(cpath / "inEdgesV", res.numEdges * sizeof(int)));
    res.outEdgesV_h.reset(new MappedFileSinkMemory(cpath / "outEdgesV", res.numEdges * sizeof(int)));
    res.inEdgesR_h.reset(new MappedFileSinkMemory(cpath / "inEdgesR", (res.numVertices + 1) * sizeof(int)));
    res.outEdgesR_h.reset(new MappedFileSinkMemory(cpath / "outEdgesR", (res.numVertices + 1) * sizeof(int)));
    res.edgeFeatures.reset(new MappedFileSinkMemory(cpath / "edgeFeatures", res.numEdges * sizeof(int)));
    res.verticesFeatures.reset(new MappedFileSinkMemory(cpath / "verticesFeatures", res.numVertices * sizeof(int)));
    res.Euid_h.reset(new MappedFileSinkMemory(cpath / "Euid", res.numEdges * sizeof(int)));
  } else {
    res.Eg_h.reset(new VectorSinkMemory(res.numEdges * sizeof(int) * 3));
    res.inEdgesV_h.reset(new VectorSinkMemory(res.numEdges * sizeof(int)));
    res.outEdgesV_h.reset(new VectorSinkMemory(res.numEdges * sizeof(int)));
    res.inEdgesR_h.reset(new VectorSinkMemory((res.numVertices + 1) * sizeof(int)));
    res.outEdgesR_h.reset(new VectorSinkMemory((res.numVertices + 1) * sizeof(int)));
    res.edgeFeatures.reset(new VectorSinkMemory(res.numEdges * sizeof(int)));
    res.verticesFeatures.reset(new VectorSinkMemory(res.numVertices * sizeof(int)));
    res.Euid_h.reset(new VectorSinkMemory(res.numEdges * sizeof(int)));
  }

  res.totalBytes = calcTotalBytes(res.numEdges, res.numVertices);

  ed = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(ed - st).count();
  st = ed;
  fmt::print("# [GraphDataLoader] allocate on CPU {} ms for {} MB\n", time, res.totalBytes / (1024 * 1024));

  const int *Tbeg = (int *) TPtr_->data() + beg;
  const int *Tend = Tbeg + res.numEdges;
  thrust::device_vector<int> time_d(Tbeg, Tend);
  auto zipItBeg = thrust::make_zip_iterator(
    thrust::make_tuple(VerticesVec_d.begin(), VerticesVec_d.begin() + res.numEdges, time_d.begin())
  );
  auto zipItEnd = zipItBeg + res.numEdges; 
  thrust::device_vector<thrust::tuple<int, int, int>> Eg_d(zipItBeg, zipItEnd);
  thrust::tuple<int, int, int> *pr = thrust::raw_pointer_cast(Eg_d.data());

  gpuErrchk(cudaMemcpy(res.Eg_h->data(), pr, res.numEdges * sizeof(int) * 3, cudaMemcpyDeviceToHost));
  // use the one above is significantly slower.
  // thrust::copy(Eg_d.begin(), Eg_d.end(), (thrust::tuple<int, int, int> *) res.Eg_h->data());
  thrust::copy(OutIn_d.begin(), OutIn_d.begin() + res.numEdges, (int *) res.outEdgesV_h->data());
  thrust::copy(OutIn_d.begin() + res.numEdges, OutIn_d.end(), (int *) res.inEdgesV_h->data());
  thrust::copy(OutR_d.begin(), OutR_d.end(), (int *) res.outEdgesR_h->data());
  thrust::copy(InR_d.begin(), InR_d.end(), (int *) res.inEdgesR_h->data());
  thrust::copy(VerticesFeatures_d.begin(), VerticesFeatures_d.end(), (int *) res.verticesFeatures->data());

  int *Eg_h = (int *) res.Eg_h->data();


  ed = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(ed - st).count();
  st = ed;
  fmt::print("# [GraphDataLoader] back to host takes {} ms\n", time);

  const char *startFEbyte = (const char *) FEPtr_->data() + sizeof(int) * beg;
  memcpy(res.edgeFeatures->data(), startFEbyte, (end - beg) * sizeof(int));

  const char *startEuidbyte = (const char *) EuidPtr_->data() + sizeof(int) * beg;
  memcpy(res.Euid_h->data(), startEuidbyte, (end - beg) * sizeof(int));

  ed = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(ed - st).count();
  st = ed;
  fmt::print("# [GraphDataLoader] copy edge features takes {} ms\n", time);

  return res;
}

#define CudaCheckCode(code) { corelib::data::cudaThrowIfError((code), __FILE__, __LINE__); }

// Error Handling
char ErrorMsgMem[1024];

const char *getErrorMsg(cudaError_t code, const char *file, int line) {
  std::sprintf(ErrorMsgMem, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
  return ErrorMsgMem;
}

void cudaThrowIfError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) 
    throw std::runtime_error(corelib::data::getErrorMsg(code, file, line));
}

// Memory Management Wrapper
void DevicePtrDeleter::operator()(void *ptr) { CudaCheckCode(cudaFree(ptr)); }

DevicePtr_t DeviceMalloc(size_t size) {
  void *ptr;
  CudaCheckCode(cudaMalloc(&ptr, size));
  return DevicePtr_t(ptr);
}

DevicePtr_t DeviceMallocAsync(size_t size, cudaStream_t stream) {
  void *ptr;
  CudaCheckCode(cudaMallocAsync(&ptr, size, stream));
  return DevicePtr_t(ptr);
}

class GraphDataManagerImpl : public GraphDataManager {
  std::vector<cudaStream_t> streams_;
  mutable std::mutex mtx_;

  DeviceGraphData allocAsyncAt(int device, size_t E, size_t V) const {
    CudaCheckCode(cudaSetDevice(device));
    auto totalBytes = calcTotalBytes(E, V);
    DeviceGraphData t;

    t.Eg_d = DeviceMallocAsync(sizeof(TemporalEdge) * E, streams_[device]);
    t.Euid_d = DeviceMallocAsync(sizeof(int) * E, streams_[device]);

    t.inEdgesV_d = DeviceMallocAsync(sizeof(int) * E, streams_[device]);
    t.outEdgesV_d = DeviceMallocAsync(sizeof(int) * E, streams_[device]);
    t.inEdgesR_d = DeviceMallocAsync(sizeof(int) * (V + 1), streams_[device]);
    t.outEdgesR_d = DeviceMallocAsync(sizeof(int) * (V + 1), streams_[device]);
    t.edgeFeatures = DeviceMallocAsync(sizeof(int) * E, streams_[device]);
    t.verticesFeatures = DeviceMallocAsync(sizeof(int) * V, streams_[device]);

    t.totalBytes = totalBytes;
    t.numEdges = E;
    t.numVertices = V;
    t.device = device;
    return t;
  }

public:
  virtual std::vector<DeviceGraphData>
  allocBatch(int allDevices, const std::vector<const HostGraphData *> hostGraphs) const override {
    auto st = std::chrono::high_resolution_clock::now();
    auto ed = st;
    time_t time;

    std::vector<size_t> numEdgesVec, numVerticesVec;
    std::transform(hostGraphs.begin(), hostGraphs.end(), std::back_inserter(numEdgesVec), 
      [](auto &a) {return a->numEdges;}
    );
    std::transform(hostGraphs.begin(), hostGraphs.end(), std::back_inserter(numVerticesVec), 
      [](auto &a) {return a->numVertices;}
    );
    auto maxNumEdges = *std::max_element(numEdgesVec.begin(), numEdgesVec.end());
    auto maxNumVertices = *std::max_element(numVerticesVec.begin(), numVerticesVec.end());

    std::vector<DeviceGraphData> res;
    for (int i = 0; i < allDevices; i++) {
      res.push_back(allocAsyncAt(i, maxNumEdges, maxNumVertices));
    }
    for (int i = 0; i < allDevices; i++) {
      CudaCheckCode(cudaStreamSynchronize(streams_[i]));
    }

    ed = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(ed - st).count();
    st = ed;
    fmt::print("# [GraphDataManager] allocate on GPU takes {}ms\n", time);
    return res;
  }

  virtual DeviceGraphData
  alloc(int device, const HostGraphData *hostGraph) const override {
    auto t = allocAsyncAt(device, hostGraph->numEdges, hostGraph->numVertices);
    waitMove(device);
    return t;
  }

  virtual void
  moveAsync(DeviceGraphData &deviceGraph, const HostGraphData &hostGraph, int device) const override {
    std::lock_guard<std::mutex> lg(mtx_); 
    CudaCheckCode(cudaSetDevice(device));
    DeviceGraphData &s = deviceGraph;
    s.device = device;
    s.name = hostGraph.name;
    s.beg = hostGraph.beg;
    s.end = hostGraph.end;
    s.numEdges = hostGraph.numEdges;
    s.numVertices = hostGraph.numVertices;
    s.totalBytes = hostGraph.totalBytes;

    CudaCheckCode(cudaMemcpyAsync(s.Eg_d.get(), hostGraph.Eg_h->data(), hostGraph.Eg_h->size(), cudaMemcpyHostToDevice, streams_[device]));
    CudaCheckCode(cudaMemcpyAsync(s.Euid_d.get(), hostGraph.Euid_h->data(), hostGraph.Euid_h->size(), cudaMemcpyHostToDevice, streams_[device]));

    CudaCheckCode(cudaMemcpyAsync(s.inEdgesV_d.get(), hostGraph.inEdgesV_h->data(), hostGraph.inEdgesV_h->size(), cudaMemcpyHostToDevice, streams_[device]));
    CudaCheckCode(cudaMemcpyAsync(s.outEdgesV_d.get(), hostGraph.outEdgesV_h->data(), hostGraph.outEdgesV_h->size(), cudaMemcpyHostToDevice, streams_[device]));
    CudaCheckCode(cudaMemcpyAsync(s.inEdgesR_d.get(), hostGraph.inEdgesR_h->data(), hostGraph.inEdgesR_h->size(), cudaMemcpyHostToDevice, streams_[device]));
    CudaCheckCode(cudaMemcpyAsync(s.outEdgesR_d.get(), hostGraph.outEdgesR_h->data(), hostGraph.outEdgesR_h->size(), cudaMemcpyHostToDevice, streams_[device]));
    CudaCheckCode(cudaMemcpyAsync(s.edgeFeatures.get(), hostGraph.edgeFeatures->data(), hostGraph.edgeFeatures->size(), cudaMemcpyHostToDevice, streams_[device]));
    CudaCheckCode(cudaMemcpyAsync(s.verticesFeatures.get(), hostGraph.verticesFeatures->data(), hostGraph.verticesFeatures->size(), cudaMemcpyHostToDevice, streams_[device]));
  } 

  virtual void
  waitMove(int device) const override {
    CudaCheckCode(cudaStreamSynchronize(streams_[device]));
  }

  GraphDataManagerImpl(int totalDevice) : streams_(totalDevice) {
    for (int i = 0; i < totalDevice; i++) {
      CudaCheckCode(cudaSetDevice(i));
      CudaCheckCode(cudaStreamCreate(&streams_[i]));
    }
  }

  ~GraphDataManagerImpl() {}
};

std::unique_ptr<GraphDataManager> MakeGraphDataManager(int totalDevice) {
  return std::unique_ptr<GraphDataManager>(new GraphDataManagerImpl(totalDevice));
}

const TemporalEdge *DeviceJobData::Em_d() const {
  return (const TemporalEdge *) Em_d_.get();

}
const MotifEdgeInfoV1 *DeviceJobData::minfo() const {
  return (const MotifEdgeInfoV1 *) minfo_.get();
}

DeviceJobData::DeviceJobData(const DeviceGraphData &deviceGraph, const HostMotifData &hostMotif) {
  CudaCheckCode(cudaSetDevice(deviceGraph.device));
  device = deviceGraph.device;
  graphNumEdges = deviceGraph.numEdges;
  graphNumVertices = deviceGraph.numVertices;
  Eg_d = (const TemporalEdge *) deviceGraph.Eg_d.get();
  inEdgesV_d = (const int *) deviceGraph.inEdgesV_d.get();
  outEdgesV_d = (const int *) deviceGraph.outEdgesV_d.get();
  inEdgesR_d = (const int *) deviceGraph.inEdgesR_d.get();
  outEdgesR_d = (const int *) deviceGraph.outEdgesR_d.get();
  edgefeatures_d = (const int *) deviceGraph.edgeFeatures.get();
  nodefeatures_d = (const int *) deviceGraph.verticesFeatures.get();
  Euid_d = (const int *) deviceGraph.Euid_d.get();

  motifNumEdges = hostMotif.numEdges;
  motifNumVertices = hostMotif.numVertices;
  auto Em_bytes = sizeof(TemporalEdge) * motifNumEdges;
  Em_d_ = DeviceMalloc(Em_bytes);
  CudaCheckCode(cudaMemcpy(Em_d_.get(), hostMotif.edges.data(), Em_bytes, cudaMemcpyHostToDevice));

  minfol_ = hostMotif.minfo;
  for (auto &mi : minfol_) {
    if (mi.io == 1) {
      mi.arrR = outEdgesR_d;
      mi.arrV = outEdgesV_d; 
    } else {
      mi.arrR = inEdgesR_d;
      mi.arrV = inEdgesV_d; 
    }
  }
  auto minfo_bytes = sizeof(MotifEdgeInfoV1) * minfol_.size();
  minfo_ = DeviceMalloc(minfo_bytes);
  CudaCheckCode(cudaMemcpy(minfo_.get(), minfol_.data(), minfo_bytes, cudaMemcpyHostToDevice));
}


}

namespace corelib {

std::pair<int, int> GPUWorker::pause() {
  return {-1, -1};
}
  
std::unique_ptr<GPUWorker> GPUWorkerLibManager::construct(const std::string &lib, int gpu) {
  CudaCheckCode(cudaSetDevice(gpu));
  auto factory = table_.at(lib).getWorker;
  return std::unique_ptr<GPUWorker>(factory(gpu));
}

}