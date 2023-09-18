#include "data.h"
#include <numeric>
#include <iostream>
#include <cstring>
#include <chrono>
#include <range/v3/all.hpp>
#include <dlfcn.h>
#include <random>
#include <thread>
#include <functional>
#include <fmt/ranges.h>

#include "Graph.h" // for now

namespace corelib {

constexpr const int NUMTHREADS = 64;

}

namespace corelib::util {
  void generateRandToFile(size_t N, std::filesystem::path pathToFile) {
    std::vector<int> rands(N);

    std::vector<size_t> beg(NUMTHREADS), end(NUMTHREADS);
    size_t chunk = N / NUMTHREADS;
    for (size_t i = 0; i < NUMTHREADS; i++) {
      beg[i] = i * chunk;
      end[i] = (i + 1) * chunk;
    }
    end[NUMTHREADS - 1] = N;

    std::vector<std::thread> ts;
    for (size_t i = 0; i < NUMTHREADS; i++) {
      ts.push_back(std::thread([&, i]{
        std::mt19937 gen;
        std::random_device rd;
        gen.seed(rd());
        std::uniform_int_distribution<> dist(0, 1000000000);
        for (size_t j = beg[i]; j < end[i]; j++) {
          rands[j] = dist(gen);
        }
      }));
    }

    std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));

    fmt::print("[generateRandToFile] finish generation\n");
    fmt::print("[generateRandToFile] first 5 numbers: {} {} {} {} {}\n", rands[0], rands[1], rands[2], rands[3], rands[4]);

    data::MappedFileSinkMemory sink(pathToFile, sizeof(int) * N);
    memcpy(sink.data(), rands.data(), sizeof(int) * N);
  }

  void memcpyPar(void *dst, const void *src, size_t count) {
    const size_t chunk = count / NUMTHREADS;

    std::vector<std::thread> ts;
    for (int tid = 0; tid < NUMTHREADS; tid++) {
      ts.push_back(std::thread([&, tid]{
        size_t start = chunk * tid, end = chunk * (tid + 1);
        if (tid == NUMTHREADS - 1) {
          end = count;
        }

        void *dstl = ((char *) dst) + start;
        const void *srcl = ((const char *) src) + start;
        size_t countl = end - start;
        memcpy(dstl, srcl, countl);
      }));
    }
    std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));
  }
}

namespace corelib {

bool operator==(const TemporalEdge &lhs, const TemporalEdge &rhs) {
  return (lhs.u == rhs.u) && (lhs.v == rhs.v) && (lhs.t == rhs.t);
}

std::ostream &operator<<(std::ostream &o, const MotifEdgeInfoV1 &obj) {
  o << fmt::format("bN: {}, cN: {}, mN: {}, arrR: {}, arrV: {}, io: {}, newNode: {}",
    obj.baseNode, obj.constraintNode, obj.mappedNodes, fmt::ptr(obj.arrR), fmt::ptr(obj.arrV), obj.io, obj.newNode   
  );
  return o;
}


}

namespace corelib::data {

template<int N = 9>
struct Record {
  // parse N integers from str
  Record(const char *str) {
    auto beg = str;
    for (int i = 0; i < N - 1; i++) {
      while (*beg == ' ') beg++;
      fs[i] = std::atoi(beg);
      while (*beg != ' ') beg++;
    }
    fs[N - 1] = std::atoi(beg);
  }
  
  Record() {}

  // 9 features
  int fs[N] = {0};
};

// scan every line in a mapped_file file
class LineScanner {
  boost::iostreams::mapped_file &file_;
  int t_;

public:
  LineScanner(boost::iostreams::mapped_file &file, int t)
    : file_(file), t_(t) {}

  template <typename Func>
  void scan(Func func, size_t beg, size_t end) {
    auto f = file_.const_data() + beg;
    auto l = f + (end - beg);
    uint64_t num_bytes = (l - f) / t_;

    std::vector<std::thread> ts;
    for (int tid = 0; tid < t_; tid++) {
      ts.push_back(std::thread([&, tid]{
        auto start = tid * num_bytes + f;
        auto end = start + num_bytes;

        if (tid == t_ - 1) {
          end = l;
        } else {
          end = static_cast<const char *>(memchr(end, '\n', l - end));
        }
        
        if (tid == 0) {
          start = f;
        } else {
          start = static_cast<const char *>(memchr(start, '\n', l - start));
          start++;
        }

        do {
          func(tid, start);
          start = static_cast<const char *>(memchr(start, '\n', l-start));
          if (start) start++;
        } while (start && start < end);
      }));
    }
    std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));
  } 
};

CacheDirectoryAccess::CacheDirectoryAccess(std::filesystem::path pathToGraph)
  : name_(pathToGraph.stem()), graphPath_(pathToGraph),
  cacheDir_(pathToGraph.parent_path() / pathToGraph.stem())
   {
  if (!std::filesystem::exists(cacheDir_)) {
    std::filesystem::create_directory(cacheDir_);
  }
}

bool EdgeListLoader::shouldInit() const {
  bool existU = std::filesystem::exists(cacheUPath());
  bool existV = std::filesystem::exists(cacheVPath());
  bool existT = std::filesystem::exists(cacheTPath());
  bool existIDS = std::filesystem::exists(cacheIDSPath());
  return !(existU && existV && existT && existIDS);
}

void EdgeListLoader::createEdgeListCache() {
  auto start = std::chrono::high_resolution_clock::now();
  auto end = start;
  time_t time;
  boost::iostreams::mapped_file mfile(graphPath_, boost::iostreams::mapped_file::readonly);
  corelib::data::LineScanner lscan(mfile, thread_);

  std::vector<int> count(thread_); 
  std::vector<std::vector<int>> Us(thread_), Vs(thread_), Ts(thread_);
  lscan.scan([&](int tid, const char *start) { 
    Record<3> r(start);
    auto &U = Us[tid];
    auto &V = Vs[tid];
    auto &T = Ts[tid];
    U.push_back(r.fs[0]);
    V.push_back(r.fs[1]);
    T.push_back(r.fs[2]);
    count[tid]++;
  }, 0, mfile.size());

  for (int i = 1; i < thread_; i++) {
    count[i] += count[i - 1];
  }
  int total = count.back();

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(end - start).count();
  start = end;

  size_t arrSize = total * sizeof(int);
  std::vector<int> UVec(total);
  std::vector<int> VVec(total);
  std::vector<int> TVec(total);
  std::vector<int> IDS(total); // edge real index

  std::thread populateIDS([&] {
    for (int i = 0; i < total; i++) {
      IDS[i] = i;
    }
  });

  std::vector<std::thread> ts;
  for (int i = 0; i < thread_; i++) {
    ts.push_back(std::thread([&, i]{
      size_t offset = i ? count[i - 1] : 0;
      offset *= sizeof(int);
      auto Udst = ((char *) UVec.data()) + offset;
      auto Vdst = ((char *) VVec.data()) + offset;
      auto Tdst = ((char *) TVec.data()) + offset;

      auto Usrc = Us[i].data();
      auto Vsrc = Vs[i].data();
      auto Tsrc = Ts[i].data();

      auto bytes = Us[i].size() * sizeof(int);

      std::memcpy(Udst, Usrc, bytes);
      std::memcpy(Vdst, Vsrc, bytes);
      std::memcpy(Tdst, Tsrc, bytes);
    }));
  }

  std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));
  populateIDS.join();


  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(end - start).count();
  start = end;
  fmt::print("# [EdgeListLoader] Edge List Parsing (combining) for {}, takes {} ms for {} records\n", name_, time, total);

  auto zip_view = ranges::view::zip(UVec, VVec, TVec, IDS);
  ranges::stable_sort(zip_view, std::less<>{},
             [](const auto& t) { return std::get<2>(t);}); // Projection
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(end - start).count();
  start = end;
  // ~17 sec for 64M records
  fmt::print("# [EdgeListLoader] Edge List Parsing (serial stable sort) for {}, takes {} ms for {} records\n", name_, time, total);
  
  MappedFileSinkMemory Ufile(cacheUPath(), arrSize);
  MappedFileSinkMemory Vfile(cacheVPath(), arrSize);
  MappedFileSinkMemory Tfile(cacheTPath(), arrSize);
  MappedFileSinkMemory IDSfile(cacheIDSPath(), arrSize);

  ts.clear();
  for (int i = 0; i < thread_; i++) {
    ts.push_back(std::thread([&, i]{
      size_t offset = i ? count[i - 1] : 0;
      offset *= sizeof(int);
      auto Udst = (char *) Ufile.data() + offset;
      auto Vdst = (char *) Vfile.data() + offset;
      auto Tdst = (char *) Tfile.data() + offset;
      auto IDSdst = (char *) IDSfile.data() + offset;

      auto Usrc = (char *) UVec.data() + offset;
      auto Vsrc = (char *) VVec.data() + offset;
      auto Tsrc = (char *) TVec.data() + offset;
      auto IDSsrc = (char *) IDS.data() + offset;

      auto bytes = Us[i].size() * sizeof(int);

      std::memcpy(Udst, Usrc, bytes);
      std::memcpy(Vdst, Vsrc, bytes);
      std::memcpy(Tdst, Tsrc, bytes);
      std::memcpy(IDSdst, IDSsrc, bytes);
    }));
  }

  std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(end - start).count();
  start = end;
  fmt::print("# [EdgeListLoader] Edge List Parsing (dump to file) for {}, takes {} ms for {} records\n", name_, time, total);
}

void EdgeListLoader::loadEdgeListCache() {
  auto start = std::chrono::high_resolution_clock::now();
  auto end = start;
  U_.reset(new MappedFileSourceMemory(cacheUPath()));
  V_.reset(new MappedFileSourceMemory(cacheVPath()));
  T_.reset(new MappedFileSourceMemory(cacheTPath()));
  IDS_.reset(new MappedFileSourceMemory(cacheIDSPath()));
  end = std::chrono::high_resolution_clock::now();
  time_t time = std::chrono::duration_cast<std::chrono::milliseconds, time_t>(end - start).count();
  fmt::print("# [EdgeListLoader] Edge List Loading takes {} ms for {} records\n", time, edgeListLength());
}

EdgeListLoader::EdgeListLoader(std::filesystem::path pathToGraph)
  : CacheDirectoryAccess(pathToGraph) { 
  if (shouldInit()) {
    createEdgeListCache();
  }
  loadEdgeListCache();
}

size_t EdgeListLoader::edgeListLength() const {
  return U_->size() / sizeof(int);
}

bool FeatureLoader::shouldInit() const {
  bool existFE = std::filesystem::exists(cacheFEPath());
  bool existFVV = std::filesystem::exists(cacheFVVPath());
  bool existFVM = std::filesystem::exists(cacheFVMPath());
  return !(existFE && existFVV && existFVM);
}

void FeatureLoader::loadFeatureCache() {
  edgeFeatures_.reset(new MappedFileSourceMemory(cacheFEPath()));
  verticesFeatures_.reset(new MappedFileSourceMemory(cacheFVVPath()));
  verticesMap_.reset(new MappedFileSourceMemory(cacheFVMPath()));
}

FeatureLoader::FeatureLoader(const EdgeListLoader *eloader, std::filesystem::path pathToRands)
  : CacheDirectoryAccess(eloader->graphPath_), pathToRands_(pathToRands), eloader_(eloader) {
  if (shouldInit()) {
    createFeatureCache();
  }
  loadFeatureCache();
}

size_t FeatureLoader::edgeFeaturesLength() const {
  return edgeFeatures_->size() / sizeof(int);
}

size_t FeatureLoader::verticesFeaturesLength() const {
  return verticesFeatures_->size() / sizeof(int);
}

std::vector<SubPartition> divideList(const data::EdgeListLoader &eloader, int delta, int num) {
  const int *time = (const int *) eloader.getTPtr()->data();
  size_t size = eloader.edgeListLength();
  std::vector<SubPartition> res(num * 2 - 1);
  auto chunk = size / num;
  for (int i = 0; i < num; i++) {
    res[i].beg = i * chunk;
    res[i].end = (i + 1) * chunk;
  }
  res[num - 1].end = size;
  res[num - 1].n = res[num - 1].end - res[num - 1].beg;

  for (int i = 0; i < num - 1; i++) {
    auto p = res[i].end;
    auto t = time[p];
    auto lo = t - delta, hi = t + delta;
    auto loId = std::distance(time, std::lower_bound(time, time + size, lo)) - 1; 
    auto hiId = std::distance(time, std::lower_bound(time, time + size, hi)) + 1;
    res[i + num].beg = loId;
    res[i + num].end = hiId;
    res[i + num].n = p - loId;
    res[i].n = loId - res[i].beg;
  }
  return res;
}

void removeMinor(HostGraphDataList &complete) {
  size_t n = (complete.size() + 1) / 2;
  complete.resize(n);
}

void includeMinor(HostGraphDataList &major, HostGraphDataList &minor) {
  std::move(minor.begin(), minor.end(), std::back_inserter(major));
  minor.clear();
}

HostGraphDataList
GraphDataLoader::createPartitionsData(const SubPartitionList &partitions) const {
  HostGraphDataList major = createPartitionsDataMajor(partitions);
  HostGraphDataList minor = createPartitionsDataMinor(partitions);
  includeMinor(major, minor);
  return major;
}

HostGraphDataList
GraphDataLoader::createPartitionsDataMajor(const SubPartitionList &partitions) const {
  std::vector<HostGraphData> res;
  size_t major = (partitions.size() + 1) / 2;
  for (size_t i = 0; i < major; i++) {
    auto &p = partitions[i];
    res.push_back(createGraphData(p.beg, p.end, true));
  }
  return res;
}

HostGraphDataList
GraphDataLoader::createPartitionsDataMinor(const SubPartitionList &partitions) const {
  std::vector<HostGraphData> res;
  size_t major = (partitions.size() + 1) / 2;
  for (size_t i = major; i < partitions.size(); i++) {
    auto &p = partitions[i];
    res.push_back(createGraphData(p.beg, p.end, false));
  }
  return res;
}

HostMotifData::HostMotifData(const std::filesystem::path &pathToMotif) {
  // ** Temporally implement in this way
  auto M = TemporalGraph::ReadGraph(pathToMotif);
  auto me = M->edges();
  for (auto &e: me) {
    edges.push_back({e.u, e.v, e.t});
  }
  name = pathToMotif.stem();
  numEdges = M->num_edges();
  numVertices = M->num_nodes();

  int nodeMax = 1;
  for (size_t i = 1; i < numEdges; i++) {
    auto &e = edges[i];
    MotifEdgeInfoV1 mi;
    if (e.u > nodeMax) {
      mi.baseNode = e.v;
      mi.constraintNode = -e.u;
      mi.io = 0;
      mi.mappedNodes = nodeMax + 1;
      mi.newNode = true;
      nodeMax++;
    } else if (e.v > nodeMax) {
      mi.baseNode = e.u;
      mi.constraintNode = -e.v;
      mi.io = 1;
      mi.mappedNodes = nodeMax + 1;
      mi.newNode = true;
      nodeMax++;
    } else {
      mi.baseNode = e.v;
      mi.constraintNode = e.u;
      mi.io = -1;
      mi.mappedNodes = nodeMax + 1;
      mi.newNode = false;
    }
    minfo.push_back(mi);
  }
}


MineJob::MineJob(const DeviceJobData *_data, int _delta, int _beg, int _end) 
  : device(_data->device), beg(_beg), end(_end), delta(_delta), data(_data) {
  if (end == -1) {
    end = _data->graphNumEdges - _data->motifNumEdges + 1; 
  }
}

std::tuple<int, int> MineJob::getWork() const {
  return {beg, end};
}

void MineJob::setWork(int _beg, int _end) {
  beg = _beg;
  end = _end;
}

std::ostream &operator<<(std::ostream &o, const HostGraphData &obj) {
  o << fmt::format("name: {}\n", obj.name);
  o << fmt::format("beg: {}, end: {}\n", obj.beg, obj.end);
  o << fmt::format("numEdges: {}, numVertices: {}, totalBytes: {}", obj.numEdges, obj.numVertices, obj.totalBytes);
  return o;
}

std::ostream &operator<<(std::ostream &o, const DeviceGraphData &obj) {
  o << fmt::format("device: {}, name: {}\n", obj.device, obj.name);
  o << fmt::format("beg: {}, end: {}\n", obj.beg, obj.end);
  o << fmt::format("numEdges: {}, numVertices: {}, totalBytes: {}\n", obj.numEdges, obj.numVertices, obj.totalBytes);
  o << fmt::format("Eg_d: {}, inEdgesV_d: {}, outEdgesV_d: {}\n", 
    fmt::ptr(obj.Eg_d.get()), fmt::ptr(obj.inEdgesV_d.get()), fmt::ptr(obj.outEdgesV_d.get()));
  o << fmt::format("inEdgesR_d: {}, outEdgesR_d: {}\n", 
    fmt::ptr(obj.inEdgesR_d.get()), fmt::ptr(obj.outEdgesR_d.get()));
  o << fmt::format("edgeFeatures: {}, verticesFeatures: {}", 
    fmt::ptr(obj.edgeFeatures.get()), fmt::ptr(obj.verticesFeatures.get()));
  return o;
}

std::ostream &operator<<(std::ostream &o, const HostMotifData &obj) {
  o << fmt::format("name: {}, numEdges: {}, numVertices: {}\n", obj.name, obj.numEdges, obj.numVertices);
  o << fmt::format("Edges:\n");
  for (auto &e: obj.edges) {
    o << fmt::format("{} {} {}\n", e.u, e.v, e.t);
  }
  o << fmt::format("minfo:\n");
  std::string t;
  for (auto &mi: obj.minfo) {
    t += fmt::format("{}\n", mi);
  }
  t.pop_back();
  o << t;
  return o;
}

std::ostream &operator<<(std::ostream &o, const DeviceJobData &obj) {
  o << fmt::format("device: {}\n", obj.device);
  o << "[Graph]\n";
  o << fmt::format("numEdges: {}, numVertices: {}\n", obj.graphNumEdges, obj.graphNumVertices);
  o << fmt::format("Eg_d: {}, inEdgesV_d: {}, outEdgesV_d: {}\n", 
    fmt::ptr(obj.Eg_d), fmt::ptr(obj.inEdgesV_d), fmt::ptr(obj.outEdgesV_d));
  o << fmt::format("inEdgesR_d: {}, outEdgesR_d: {}\n", 
    fmt::ptr(obj.inEdgesR_d), fmt::ptr(obj.outEdgesR_d));
  o << fmt::format("edgeFeatures: {}, verticesFeatures: {}\n", 
    fmt::ptr(obj.edgefeatures_d), fmt::ptr(obj.nodefeatures_d));
  o << "[Motif]\n";
  o << fmt::format("numEdges: {}, numVertices: {}\n", obj.motifNumEdges, obj.motifNumVertices);
  std::string t;
  for (auto &mi: obj.minfol_) {
    t += fmt::format("{}\n", mi);
  }
  t.pop_back();
  o << t;
  return o;
}

}

namespace corelib {

GPUWorker::GPUWorker(std::string type, int gpu, int sizeBlock) : 
type_(type), gpu_(gpu), time_(0), sizeBlock_(sizeBlock) {}

GPUWorker::~GPUWorker() {}

unsigned long long GPUWorker::timed_run() {
  using namespace std::chrono;
  auto st = high_resolution_clock::now();
  auto res = run();
  auto ed = high_resolution_clock::now();
  auto stept = duration_cast<duration<unsigned long long, std::micro>>(ed - st).count();
  time_ += stept;
  return res;
}

void GPUWorker::take(data::MineJob &job) {
  job_ = &job;
}

unsigned long long GPUWorker::time() {
  return time_;
}

unsigned long long GPUWorker::count() {
  return count_;
}

void GPUWorker::clear_time() {
  time_ = 0;
}


int GPUWorker::numBlocks() {
  if (!job_) {
    return -1;
  } else {
    return (job_->end + sizeBlock_ - 1) / sizeBlock_;
  }
}

std::vector<int> GPUWorker::printEnum(int) { return {}; }

LibHandle::LibHandle(void *h, FactoryType f) : handle(h), getWorker(f) {};

LibHandle::LibHandle(const std::string &libpath) {
  handle = dlopen(libpath.c_str(), RTLD_LAZY);
  if (!handle) {
    std::cerr << "Load Lib Error: " << dlerror() << std::endl;
    exit(-1);
  }

  getWorker = (LibHandle::FactoryType) dlsym(handle, "getWorker");
  if (!getWorker) {
    dlclose(handle);
    std::cerr << "query func Error: " << dlerror() << std::endl;
    exit(-1);
  }
}

LibHandle::LibHandle(LibHandle &&rhs) {
  (*this) = std::move(rhs);
}

LibHandle &LibHandle::operator=(LibHandle &&rhs) {
  if (this != &rhs) {
    handle = rhs.handle;
    getWorker = rhs.getWorker;
    rhs.handle = nullptr;
    rhs.getWorker = nullptr;
  }
  return rhs;
}

std::string LibHandle::toString() const {
  std::stringstream s;
  s << "handle: " << (void *) (handle)
    << ", factory: " << (void *) (getWorker) ;
  return s.str();
}

LibHandle::~LibHandle() { dlclose(handle); }

void GPUWorkerLibManager::openLib(const std::string &libpath) {
  std::string name;
  std::size_t pos = libpath.find_last_of("/");
  if (pos != std::string::npos) {
      name = libpath.substr(pos + 4, libpath.length() - pos - 7); // Extract libpathrary name from path with directory information
  } else {
      name = libpath.substr(3, libpath.length() - 6); // Extract libpathrary name from path without directory information
  }

  if (table_.find(name) != table_.end()) {
    unloadLib(name);
  } 
  table_.emplace(name, libpath);
  std::cout << "#" << name << " is opened" << std::endl;
}

void GPUWorkerLibManager::unloadLib(const std::string &lib) {
  table_.erase(lib);
  std::cout << "#" << lib << " is unloaded" << std::endl;
}

std::string GPUWorkerLibManager::toString() const {
  std::stringstream s;
  for (auto &e: table_) {
    s << e.first << ": " << e.second.toString() << std::endl;
  }
  auto str = s.str();
  if (!table_.empty()) str.pop_back();
  return str;
}

SingleGPUExecution::SingleGPUExecution(const data::GraphDataManager *dataManager, int device)
  : device_(device), dataManager_(dataManager) {}


void SingleGPUExecution::formJobData() {
  if (hostGraph_ && hostMotif_) {
    jobData_ = data::DeviceJobData(deviceGraph_, *hostMotif_);
  }
}


void SingleGPUExecution::setHostGraph(const data::HostGraphData *hostGraph, int n) {
  n_ = n;
  hostGraph_ = hostGraph;
  std::vector<const data::HostGraphData *> targ = {hostGraph_};
  deviceGraph_ = dataManager_->alloc(device_, hostGraph_);
  // deviceGraph_ = std::move(dataManager_->allocBatch(1, targ).front());
  dataManager_->moveAsync(deviceGraph_, *hostGraph_, device_);
  dataManager_->waitMove(device_);
  formJobData();
}

void SingleGPUExecution::setHostMotif(const data::HostMotifData *hostMotif) {
  hostMotif_ = hostMotif;
  formJobData();
}

void SingleGPUExecution::setWorker(GPUWorker *w) {
  w_ = w;
}

unsigned long long SingleGPUExecution::run(int delta) {
  job_ = data::MineJob(&jobData_, delta, 0, n_);
  w_->take(job_);
  return w_->timed_run();
}

SingleGPUExecutionDiv::SingleGPUExecutionDiv(const data::GraphDataManager *dataManager, int device, int div)
  : SingleGPUExecution(dataManager, device), div_(div) {}

void SingleGPUExecutionDiv::formJobData() {
  if (hostGraph_ && hostMotif_) {
    jobData_ = data::DeviceJobData(deviceGraph_, *hostMotif_);

    int chunk = n_ / div_;
    for (int i = 0; i < div_; i++) {
      queue_.push_back(Task{i * chunk, (i + 1) * chunk});
    }
    queue_.back().end = n_;
  }
}

unsigned long long SingleGPUExecutionDiv::run(int delta) {
  fmt::print("[SingleGPUExecutionDiv] div: {}\n", queue_.size());
  unsigned long long res = 0;
  for (auto &t : queue_) {
    job_ = data::MineJob(&jobData_, delta, t.beg, t.end);
    w_->take(job_);
    res += w_->timed_run();
  }
  return res;
}

void SingleGPURunBatch(const data::GraphDataManager *dataManager, data::HostGraphDataList &hostGraphs, 
  const data::HostMotifData *hostMotif, GPUWorker *w, int delta) {
  for (auto &hostGraph: hostGraphs) {
    SingleGPUExecution exec(dataManager);
    exec.setHostGraph(&hostGraph);
    exec.setHostMotif(hostMotif);
    exec.setWorker(w);
    exec.run(delta);
  }
}

MultiGPUExecution::MultiGPUExecution(const data::GraphDataManager *dataManager, int deviceNum)
  : deviceNum_(deviceNum), dataManager_(dataManager) {}

void MultiGPUExecution::loadGraph(int start, int n) {
  for (int d = 0; d < n; d++) {
    int t = start + d;
    dataManager_->moveAsync(deviceGraphs_[d], *hostGraphs_[t], d);
    deviceNs_[d] = ns_[t];
  }
  for (int d = 0; d < n; d++) {
    dataManager_->waitMove(d);
  }
}

void MultiGPUExecution::formJobData() {
  if (!deviceGraphs_.empty() && hostMotif_) {
    jobsData_.resize(deviceNum_);
    for (int d = 0; d < deviceNum_; d++) {
      jobsData_[d] = data::DeviceJobData(deviceGraphs_[d], *hostMotif_);
    }
  }
}

void MultiGPUExecution::setHostGraphs(const data::HostGraphDataList &hostGraphs, 
  const data::SubPartitionList &ps) {
  hostGraphs_.clear();
  std::transform(hostGraphs.begin(), hostGraphs.end(), std::back_inserter(hostGraphs_),
    [] (const data::HostGraphData &hg) { return &hg; });
  ns_.clear();
  std::transform(ps.begin(), ps.end(), std::back_inserter(ns_), 
    [] (const data::SubPartition &p) { return p.n; });
  
  deviceGraphs_ = dataManager_->allocBatch(deviceNum_, hostGraphs_); // alloc & transfer the first deviceNum_ graph
  deviceNs_.resize(deviceNum_);
  loadGraph(0, deviceNum_);
  formJobData();
}

void MultiGPUExecution::setHostMotif(const data::HostMotifData *hostMotif) {
  hostMotif_ = hostMotif;
  formJobData();
}

void MultiGPUExecution::setWorkers(std::vector<GPUWorker *> &ws) {
  ws_ = ws;
}


int MultiGPUExecution::numDevice() const {
  return deviceNum_;
}

void MultiGPUExecution::lanuch(int n, int delta) {
  for (int d = 0; d < n; d++) {
    jobs_[d] = data::MineJob(&jobsData_[d], delta, 0, deviceNs_[d]);
  }

  std::vector<std::thread> ts;
  for (int d = 0; d < n; d++) {
    ts.push_back(std::thread([d, delta, this]{
      auto &w = ws_[d];
      auto &j = jobs_[d];
      w->take(j);
      w->timed_run();
    }));
  }
  std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));
}

MultiGPUExecutionNaive::MultiGPUExecutionNaive(const data::GraphDataManager *dataManager, int deviceNum)
  : MultiGPUExecution(dataManager, deviceNum) {}

void MultiGPUExecutionNaive::run(int delta) {
  jobs_.resize(deviceNum_);
  lanuch(deviceNum_, delta);
  loadGraph(deviceNum_, deviceNum_ - 1);
  formJobData();
  lanuch(deviceNum_ - 1, delta);
}

MultiGPUExecutionDyn::MultiGPUExecutionDyn(const data::GraphDataManager *dataManager, int deviceNum)
  : MultiGPUExecution(dataManager, deviceNum) {}

void MultiGPUExecutionDyn::formQueue() {
  if (!deviceGraphs_.empty() && hostMotif_) {
    taskQueues_.clear();
    for (int d = 0; d < deviceNum_; d++) {
      TaskQueue q(N);
      size_t chunk = deviceNs_[d] / N;
      for (int i = 0; i < N - 1; i++) {
        int beg = i * chunk;
        int end = (i + 1) * chunk;
        q[i] = Task{beg, end};
      }
      q[N - 1] = Task{(int) ((N - 1) * chunk), ns_[d]};
      taskQueues_.push_back(q);
    }
  }
  qMtxes_ = std::vector<std::mutex>(deviceNum_);
  atQueue_.reserve(deviceNum_);
  for (int d = 0; d < deviceNum_; d++) {
    atQueue_[d] = d;
  }
}

int MultiGPUExecutionDyn::whichLeft() {
  int which = -1;
  size_t maxSize = 0;
  for (int d = 0; d < deviceNum_; d++) {
    std::lock_guard<std::mutex> lg(qMtxes_[d]);
    if (taskQueues_[d].size() > maxSize) {
      which = d;
    }
  }
  return which;
}

auto MultiGPUExecutionDyn::getTask(int d) -> Task {
  int q = atQueue_[d];
  std::lock_guard<std::mutex> lg(qMtxes_[q]);
  if (taskQueues_[q].empty()) {
    return Task{-1, -1};
  } else {
    Task res = taskQueues_[q].back();
    taskQueues_[q].pop_back();
    return res;
  }
}

void MultiGPUExecutionDyn::threadFn(int d, int delta) {
  auto &w = ws_[d];
  while (true) {
    auto task = getTask(d);
    if (task.beg >= 0) {
      jobs_[d] = data::MineJob(&jobsData_[d], delta, task.beg, task.end);
      // fmt::print("# [Dyn exec] start task {}\n", d);
      w->take(jobs_[d]);
      // fmt::print("# [Dyn exec] start run {}\n", d);
      w->timed_run();
      // fmt::print("# [Dyn exec] finish run {}\n", d);
    } else {
      // fmt::print("# [Dyn exec] {} start switching\n", d);
      int which = this->whichLeft();
      if (which != -1) {
        // [NOTE] a bug if we use lanuchDyn twice
        fmt::print("# [Dyn exec] swtich {} -> {}\n", d, which);
        dataManager_->moveAsync(deviceGraphs_[d], *hostGraphs_[which], d);
        dataManager_->waitMove(d);
        jobsData_[d] = data::DeviceJobData(deviceGraphs_[d], *hostMotif_);
        atQueue_[d] = which;
        fmt::print("# [Dyn exec] swtich {} -> {} over\n", d, which);
      } else {
        fmt::print("# [Dyn exec] {} return\n", d);
        return;
      }
    }
  }
}

void MultiGPUExecutionDyn::launchDyn(int n, int delta) {
  formQueue();

  std::vector<std::thread> ts;
  for (int d = 0; d < n; d++) {
    ts.push_back(std::thread(
      std::bind(&MultiGPUExecutionDyn::threadFn, this, d, delta)
    ));
  }
  std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));
}

void MultiGPUExecutionDyn::run(int delta) {
  jobs_.resize(deviceNum_);
  launchDyn(deviceNum_, delta);
  loadGraph(deviceNum_, deviceNum_ - 1);
  formJobData();
  lanuch(deviceNum_ - 1, delta);
}

// [Reserved] should not be used

SingleGPUExecutionPause::SingleGPUExecutionPause(const data::GraphDataManager *dataManager, int device)
  : SingleGPUExecution(dataManager, device) {}

unsigned long long SingleGPUExecutionPause::run(int delta) {
  using namespace std::literals::chrono_literals;
  std::pair<int, int> j = {0, n_};
  unsigned long long res = 0;
  do {
    fmt::print("[SingleGPUExecutionPause] {} {}\n", j.first, j.second);
    job_ = data::MineJob(&jobData_, delta, j.first, j.second);
    std::thread t([&] { 
      w_->take(job_);
      res += w_->timed_run();
    });
    std::this_thread::sleep_for(500ms);
    j = w_->pause();
    t.join();
  } while (j.first < j.second); 
  
  return res;
  
}


MultiGPUExecutionDynPause::MultiGPUExecutionDynPause(const data::GraphDataManager *dataManager, int deviceNum) 
  : MultiGPUExecutionDyn(dataManager, deviceNum) {}

void MultiGPUExecutionDynPause::preempt() {
  fmt::print("preempt called\n");
  std::lock_guard<std::mutex> lg(pmtx_);
  for (int d = 0; d < deviceNum_; d++) {
    fmt::print("preempt {}\n", d);
    if (!done_[d]) {
      fmt::print("preempt {} called\n", d);
      auto r = ws_[d]->pause();
      fmt::print("preempt {} {}\n", r.first, r.second);
    }
  }
}

void MultiGPUExecutionDynPause::threadFnSignal(int d, int delta) {
  auto &w = ws_[d];
  while (true) {
    auto task = getTask(d);
    if (task.beg >= 0) {
      jobs_[d] = data::MineJob(&jobsData_[d], delta, task.beg, task.end);
      w->take(jobs_[d]);
      w->timed_run();
    } else {
      int which = this->whichLeft();
      if (which != -1) {
        fmt::print("# [Dyn exec] swtich {} -> {}\n", d, which);
        dataManager_->moveAsync(deviceGraphs_[d], *hostGraphs_[which], d);
        dataManager_->waitMove(d);
        jobsData_[d] = data::DeviceJobData(deviceGraphs_[d], *hostMotif_);
        atQueue_[d] = which;
        fmt::print("# [Dyn exec] swtich {} -> {} over\n", d, which);
      } else {
        fmt::print("# [Dyn exec] {} return\n", d);
        { // signal and exit
          std::lock_guard<std::mutex> lg(pmtx_);
          done_[d] = true;
        }
        pcv_.notify_all();
        return;
      }
    }
  }

}

void MultiGPUExecutionDynPause::launchDynPause(int n, int delta) {
  done_ = std::vector<bool>(deviceNum_, false);
  formQueue();
  std::vector<std::thread> ts;
  for (int d = 0; d < n; d++) {
    ts.push_back(std::thread(
      std::bind(&MultiGPUExecutionDynPause::threadFnSignal, this, d, delta)
    ));
  }

  {
    std::unique_lock uq(pmtx_);
    pcv_.wait(uq, [this]{
      for (auto b : done_) {
        if (b) return true;
      }
      return false;
    });
  }
  preempt();
  fmt::print("preempt finishes\n");

  std::for_each(ts.begin(), ts.end(), std::mem_fn(&std::thread::join));
}

void MultiGPUExecutionDynPause::run(int delta) {
  jobs_.resize(deviceNum_);
  launchDynPause(deviceNum_, delta);
  loadGraph(deviceNum_, deviceNum_ - 1);
  formJobData();
  lanuch(deviceNum_ - 1, delta);
}

unsigned long long allCount(std::vector<GPUWorker *> &ws) {
  unsigned long long res = 0;
  for (auto &w: ws) {
    res += w->count();
  }
  return res;
}

std::ostream &operator<<(std::ostream &o, const GPUWorker &w) {
  fmt::print(o, "type: {}, gpu: {}, job:{}\n", w.type_, w.gpu_, fmt::ptr(w.job_));
  return o;
}

std::ostream &operator<<(std::ostream &o, const LibHandle &obj) {
  o << obj.toString();
  return o;
}

std::ostream &operator<<(std::ostream &o, const GPUWorkerLibManager &obj) {
  o << obj.toString();
  return o;
}
}