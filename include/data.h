#ifndef __OLD_TMOFIT_LIGHT_INCLUDE_DATA_H_
#define __OLD_TMOFIT_LIGHT_INCLUDE_DATA_H_

#include <boost/iostreams/device/mapped_file.hpp>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <fmt/format.h>
#include <future>
#include <functional>
#include <condition_variable>
#ifndef __NVCC__
#include <fmt/ostream.h>
#endif

namespace corelib::util {
  void generateRandToFile(size_t N, std::filesystem::path pathToFile);

  void memcpyPar(void *dst, const void *src, size_t count);
}

namespace corelib {

struct TemporalEdge {
  int u, v, t;
};

bool operator==(const TemporalEdge &lhs, const TemporalEdge &rhs);

struct MotifEdgeInfoV1 {
  int baseNode, constraintNode;
  int mappedNodes; // current mapped nodes, used to refer to the map data structure
  const int *arrR;
  const int *arrV;
  int io; // degnerated version
  bool newNode;
};

std::ostream &operator<<(std::ostream &o, const MotifEdgeInfoV1 &obj);

}

namespace corelib::data {

/*
 * A piece of read-only memory
 */
struct SourceMemory {
  virtual const void *data() const = 0;
  virtual size_t size() const = 0;
  virtual ~SourceMemory() {}
};

class MappedFileSourceMemory : public SourceMemory {
  boost::iostreams::mapped_file_source mfile_;
public:
  MappedFileSourceMemory(std::filesystem::path pathToFile);

  const void *data() const override { return mfile_.data(); }
  size_t size() const override { return mfile_.size(); }
  ~MappedFileSourceMemory(); 
};

/*
 * A piece of read-write memory
 */
struct SinkMemory {
  virtual void *data() const = 0;
  virtual size_t size() const = 0;
  virtual ~SinkMemory() {}
};

class MappedFileSinkMemory : public SinkMemory {
  boost::iostreams::mapped_file_sink mfile_;
public:
  MappedFileSinkMemory(std::filesystem::path pathToFile, size_t size); 
  MappedFileSinkMemory(std::filesystem::path pathToFile);
  void *data() const override { return mfile_.data(); }
  size_t size() const override { return mfile_.size(); }
  ~MappedFileSinkMemory();
};

class VectorSinkMemory : public SinkMemory {
  std::vector<uint8_t> mem_;
public:
  VectorSinkMemory(size_t size) : mem_(size) {}
  void *data() const override { return (void *) mem_.data(); }
  size_t size() const override {return mem_.size(); }
  ~VectorSinkMemory() override {}
};


// [NOTE] should have a more natural design
/*
 * A class providing methods to accesses caching conventions
 */
struct CacheDirectoryAccess {
  const std::string name_;
  const std::filesystem::path graphPath_, cacheDir_;
  CacheDirectoryAccess(std::filesystem::path pathToGraph);

  std::filesystem::path cacheUPath() const { return cacheDir_ / "U.cache"; }
  std::filesystem::path cacheVPath() const { return cacheDir_ / "V.cache"; }
  std::filesystem::path cacheTPath() const { return cacheDir_ / "T.cache"; }
  std::filesystem::path cacheIDSPath() const { return cacheDir_ / "IDS.cache"; } // ids of reordered edges
  std::filesystem::path cacheFEPath() const { return cacheDir_ / "EdgesFeatures"; }
  std::filesystem::path cacheFVVPath() const { return cacheDir_ / "VerticesFeaturesValue"; }
  std::filesystem::path cacheFVMPath() const { return cacheDir_ / "VerticesFeaturesMap"; }
  std::filesystem::path cacheGraphDirPath(size_t beg, size_t end) const {
    return cacheDir_ / fmt::format("{}-{}-{}", name_, beg, end);
  }
  std::string name() const {return name_;}
};

/*
 * A dataloader that loads edge list.
 *
 * The class maintains caches for edge lists to enjoy speedup after initial
 * setup.
 */
class EdgeListLoader : public CacheDirectoryAccess {
  std::unique_ptr<SourceMemory> U_, V_, T_, IDS_;
  const int thread_ = 64;

  bool shouldInit() const;
  void createEdgeListCache();
  void loadEdgeListCache();

public:
  // pathToGraph is the path to the overall edgelist
  EdgeListLoader(std::filesystem::path pathToGraph);

  // Create graph data for edge list in [beg, end)
  // HostGraphData createGraphData(size_t beg = 0, size_t end = -1) const;

  size_t edgeListLength() const;
  const SourceMemory *getUPtr() const { return U_.get(); }
  const SourceMemory *getVPtr() const { return V_.get(); }
  const SourceMemory *getTPtr() const { return T_.get(); }
  const SourceMemory *getIDSPtr() const { return IDS_.get(); }
};

/*
 * A feature Loader that loads random features to the graph
 */
class FeatureLoader : public CacheDirectoryAccess {
  std::unique_ptr<SourceMemory> edgeFeatures_, verticesFeatures_, verticesMap_;
  std::filesystem::path pathToRands_;
  const EdgeListLoader *eloader_;

  bool shouldInit() const;
  void createFeatureCache();
  void loadFeatureCache();

public:
  FeatureLoader(const EdgeListLoader *eloader, std::filesystem::path pathToRands);

  size_t edgeFeaturesLength() const;
  size_t verticesFeaturesLength() const;

  const SourceMemory *getFEPtr() const { return edgeFeatures_.get(); }
  const SourceMemory *getFVVPtr() const { return verticesFeatures_.get(); }
  const SourceMemory *getFVMPtr() const { return verticesMap_.get(); }
};

/*
 * The graph data for a subset of edge list
 *
 * It is page locked such that memory transfer is faster
 */
struct HostGraphData {
  std::string name;
  size_t beg{0}, end{0};
  size_t numEdges{0}, numVertices{0}, totalBytes{0};
  std::unique_ptr<SinkMemory> Eg_h, inEdgesV_h, outEdgesV_h;
  std::unique_ptr<SinkMemory> inEdgesR_h, outEdgesR_h;
  std::unique_ptr<SinkMemory> edgeFeatures, verticesFeatures;
  std::unique_ptr<SinkMemory> Euid_h;

  HostGraphData() = default;
  HostGraphData(HostGraphData &&) = default;
  HostGraphData &operator=(HostGraphData &&) = default;
};

std::ostream &operator<<(std::ostream &o, const HostGraphData &obj);


struct SubPartition {
  int beg, end, n;
};

typedef std::vector<SubPartition> SubPartitionList;
SubPartitionList divideList(const data::EdgeListLoader &eloader, int delta, int num);

typedef std::vector<HostGraphData> HostGraphDataList;
void removeMinor(HostGraphDataList &complete);
void includeMinor(HostGraphDataList &major, HostGraphDataList &minor);
/*
 * A dataloader that loads graph data
 *
 * For major paritions, it will create the graph data if the given partition is
 * met for the first time. After the initial setup,
 * The grap data is loaded from cache.
 * 
 * Smaller paritions should be created on-the-fly.
 */
class GraphDataLoader : public CacheDirectoryAccess {
  std::string stem_;
  const SourceMemory *UPtr_, *VPtr_, *TPtr_;
  const SourceMemory *FEPtr_, *FVVPtr_, *FVMPtr_;
  const SourceMemory *EuidPtr_;
  size_t edgeListLength_, verticesFeatureLength_;

  HostGraphData loadGraphData(size_t beg, size_t end) const;
  bool isCached(size_t beg, size_t end) const;
public:
  GraphDataLoader(const EdgeListLoader *eloader, const FeatureLoader *floader);
  std::string name() const {return stem_; }

  HostGraphData createGraphData(size_t beg = 0, size_t end = 0, bool cache = true) const;

  HostGraphDataList createPartitionsData(const SubPartitionList &partitions) const;
  HostGraphDataList createPartitionsDataMajor(const SubPartitionList &partitions) const;
  HostGraphDataList createPartitionsDataMinor(const SubPartitionList &partitions) const;
};

// Memory Management Wrapper
struct DevicePtrDeleter { void operator()(void *ptr); };
using DevicePtr_t = std::unique_ptr<void, DevicePtrDeleter>;

/*
 * The graph data on a device
 *
 * The objects of this class owns data on devices
 */
struct DeviceGraphData {
  int device{0};
  std::string name;
  size_t beg{0}, end{0};
  size_t numEdges{0}, numVertices{0}, totalBytes{0};
  DevicePtr_t Eg_d, inEdgesV_d, outEdgesV_d, inEdgesR_d, outEdgesR_d;
  DevicePtr_t edgeFeatures, verticesFeatures;
  DevicePtr_t Euid_d;
};


std::ostream &operator<<(std::ostream &o, const DeviceGraphData &obj);


/*
 * A class that move data asynchronously over multigpus
 */
struct GraphDataManager {
  /*
   * allocate DeviceGraphData for multiple GPUs
   *
   * It takes a list of HostGraph and allocate memory on
   * devices that can store any of them
   */
  virtual std::vector<DeviceGraphData> 
  allocBatch(int allDevices, const std::vector<const HostGraphData *> hostGraphs) const = 0;

  virtual DeviceGraphData
  alloc(int device, const HostGraphData *hostGraph) const = 0;

  // transfer a host graph to device async (thread-safe)
  virtual void
  moveAsync(DeviceGraphData &deviceGraph, const HostGraphData &hostGraph, int device) const = 0;

  // wait until the data movement on (thread-safe)
  virtual void waitMove(int device) const = 0;

  virtual ~GraphDataManager() {};
};

std::unique_ptr<GraphDataManager> MakeGraphDataManager(int totalDevice);

/*
 * Motif Data on the host
 */
struct HostMotifData {
  std::string name;
  size_t numEdges{0}, numVertices{0};
  std::vector<TemporalEdge> edges;
  std::vector<MotifEdgeInfoV1> minfo;

  HostMotifData(const std::filesystem::path &pathToMotif);
  HostMotifData() = default; 
};

std::ostream &operator<<(std::ostream &o, const HostMotifData &obj);

/*
 * Job Data on the Device
 *
 * It owns the motif data on a device and keeps 
 * reference to graph data on a device. Only the data
 * that is related to computation is kept. The Job
 * data will be on the same device of the graph data
 */
struct DeviceJobData {
  int device{0};
  size_t graphNumEdges{0}, graphNumVertices{0};
  const TemporalEdge *Eg_d;
  const int *inEdgesV_d, *outEdgesV_d, *inEdgesR_d, *outEdgesR_d;
  const int *edgefeatures_d, *nodefeatures_d;
  const int *Euid_d;

  size_t motifNumEdges{0}, motifNumVertices{0};
  DevicePtr_t Em_d_, minfo_;
  std::vector<MotifEdgeInfoV1> minfol_;
  const TemporalEdge *Em_d() const;
  const MotifEdgeInfoV1 *minfo() const;

  DeviceJobData(const DeviceGraphData &deviceGraph, const HostMotifData &hostMotif);
  DeviceJobData() = default;
};

std::ostream &operator<<(std::ostream &o, const DeviceJobData &obj);

/*
 * A Job, contains everything to let a worker to mine
 */
struct MineJob {
  int device{-1};
  int beg{-1}, end{-1}, delta{-1};
  const DeviceJobData *data{nullptr};

  MineJob(const DeviceJobData *_data, int _delta, int _beg = 0, int _end = -1);
  MineJob() = default;
  std::tuple<int, int> getWork() const;
  void setWork(int _beg, int _end);
};

}

namespace corelib {

struct GPUWorker {
  std::string type_;
  int gpu_;
  unsigned long long time_;
  int sizeBlock_;
  unsigned long long count_;

  data::MineJob *job_ = nullptr;

  GPUWorker(std::string type, int gpu = 0, int sizeBlock = 96);
  virtual ~GPUWorker();
  virtual unsigned long long run() = 0;
  virtual void take(data::MineJob &job);
  virtual void update_job() = 0;
  int numBlocks();

  unsigned long long timed_run();
  unsigned long long time();
  unsigned long long count();
  void clear_time();

  virtual std::vector<int> printEnum(int n);

  virtual std::pair<int, int> pause();
};

std::ostream &operator<<(std::ostream &o, const GPUWorker &w);

// The class to manage GPUWorker library
struct LibHandle {
  typedef GPUWorker *(*FactoryType)(int);

  void *handle;
  FactoryType getWorker;
  LibHandle(void *h, FactoryType f);
  LibHandle(const std::string &libpath);

  LibHandle(LibHandle &&rhs);
  LibHandle &operator=(LibHandle &&rhs);
  ~LibHandle(); 

  std::string toString() const;
};

std::ostream &operator<<(std::ostream &o, const LibHandle &obj);

class GPUWorkerLibManager {
  std::unordered_map<std::string, LibHandle> table_;

  public:
  void openLib(const std::string &libpath);
  void unloadLib(const std::string &lib);
  std::unique_ptr<GPUWorker> construct(const std::string &lib, int gpu);  
  std::string toString() const;
};

std::ostream &operator<<(std::ostream &o, const GPUWorkerLibManager &obj);


/*
 * Execution owns memory on devices, and utilize GPUWorker(s)
 * to finish a mining task.
 * 
 * SingleGPUExecution uses a single GPU to do the mining
 */
class SingleGPUExecution {
protected:
  int device_{0};
  GPUWorker *w_{nullptr};
  const data::HostGraphData *hostGraph_{nullptr};
  int n_{0}; // only the first n edges will be used as seeds
  const data::HostMotifData *hostMotif_{nullptr};
  const data::GraphDataManager *dataManager_{nullptr};

  data::DeviceJobData jobData_;
  data::MineJob job_;
  data::DeviceGraphData deviceGraph_;

  virtual void formJobData();
public:
  SingleGPUExecution(const data::GraphDataManager *dataManager, int device = 0);
  void setHostGraph(const data::HostGraphData *hostGraph, int n = -1);
  void setHostMotif(const data::HostMotifData *hostMotif);
  void setWorker(GPUWorker *w);
  virtual unsigned long long run(int delta);
};

/*
 * Divide the job into pieces and run it using a single GPU
 */
class SingleGPUExecutionDiv : public SingleGPUExecution {
protected:
  virtual void formJobData() override;
  int div_;
  struct Task { int beg, end; };
  std::vector<Task> queue_;
public:
  SingleGPUExecutionDiv(const data::GraphDataManager *dataManager, int device = 0, int div = 64);
  virtual unsigned long long run(int delta) override;
};

void SingleGPURunBatch(const data::GraphDataManager *dataManager, data::HostGraphDataList &hostGraphs, 
  const data::HostMotifData *hostMotif, GPUWorker *w, int delta);

/*
 * This class use multiple GPUs to do the mining and can 
 * accept a batch of graphs
 */
class MultiGPUExecution {
protected:
  int deviceNum_{0};
  std::vector<GPUWorker *> ws_;
  std::vector<const data::HostGraphData *> hostGraphs_;
  std::vector<int> ns_;
  const data::HostMotifData *hostMotif_{nullptr};
  const data::GraphDataManager *dataManager_{nullptr};

  std::vector<data::DeviceJobData> jobsData_;
  std::vector<data::MineJob> jobs_;
  std::vector<data::DeviceGraphData> deviceGraphs_;
  std::vector<int> deviceNs_;

  void loadGraph(int start, int n);
  void formJobData();
  void lanuch(int n, int delta);
public:
  MultiGPUExecution(const data::GraphDataManager *dataManager, int deviceNum);
  void setHostGraphs(const data::HostGraphDataList &hostGraphs, 
    const data::SubPartitionList &ps);
  void setHostMotif(const data::HostMotifData *hostMotif);
  void setWorkers(std::vector<GPUWorker *> &ws);
  virtual void run(int delta) = 0;
  int numDevice() const;
};

/*
 * A naive way to utilize multiple GPUs (without load balancing)
 */
class MultiGPUExecutionNaive : public MultiGPUExecution {
public:
  MultiGPUExecutionNaive(const data::GraphDataManager *dataManager, int deviceNum);
  virtual void run(int delta) override;
};

/*
 * multi GPU execution with dynamic load balancing
 */
class MultiGPUExecutionDyn : public MultiGPUExecution {
protected:
  const int N = 16;
  const int SZ = 1024 * 1024;
  struct Task {int beg, end; };
  typedef std::vector<Task> TaskQueue;

  std::vector<TaskQueue> taskQueues_;

  std::vector<std::mutex> qMtxes_;
  std::vector<int> atQueue_;

  void formQueue();
  int whichLeft();
  Task getTask(int d);
  void threadFn(int d, int delta);
  void launchDyn(int n, int delta);
public:
  MultiGPUExecutionDyn(const data::GraphDataManager *dataManager, int deviceNum);
  virtual void run(int delta) override;
};

unsigned long long allCount(std::vector<GPUWorker *> &ws);

/*
 * dynamic scheduling and pasue work to prevent work imbalance
 */
class MultiGPUExecutionDynPause : public MultiGPUExecutionDyn {
  std::vector<bool> done_;

  std::condition_variable pcv_;
  std::mutex pmtx_;

  void preempt();
  void threadFnSignal(int d, int delta);
  void launchDynPause(int n, int delta);

public:
  MultiGPUExecutionDynPause(const data::GraphDataManager *dataManager, int deviceNum);
  virtual void run(int delta) override;
};

/*
 * Run the job with 1 second pasue
 */
class SingleGPUExecutionPause : public SingleGPUExecution {
public:
  SingleGPUExecutionPause(const data::GraphDataManager *dataManager, int device = 0);
  virtual unsigned long long run(int delta) override;
};


}

#ifndef __NVCC__
// NVCC will issue warning
template <> struct fmt::formatter<corelib::MotifEdgeInfoV1> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::data::HostGraphData> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::data::DeviceGraphData> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::data::HostMotifData> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::data::DeviceJobData> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::GPUWorker> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::GPUWorkerLibManager> : fmt::ostream_formatter {};
template <> struct fmt::formatter<corelib::LibHandle> : fmt::ostream_formatter {};
#endif

#endif // __OLD_TMOFIT_LIGHT_INCLUDE_DATA_H_