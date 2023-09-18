#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>

#include <thread>
#include <functional>

namespace py = pybind11;

#include "data.h"

PYBIND11_MAKE_OPAQUE(corelib::data::HostGraphDataList);
PYBIND11_MAKE_OPAQUE(corelib::data::SubPartitionList);

PYBIND11_MODULE(corelib, corelib) {
  corelib.doc() = "Libraries for the temporal pattern mining system";

  auto data = corelib.def_submodule("data");
  py::bind_vector<corelib::data::HostGraphDataList>(data, "HostGraphDataList");
  py::bind_vector<corelib::data::SubPartitionList>(data, "SubPartitionList");

  py::class_<corelib::data::EdgeListLoader>(data, "EdgeListLoader", "A dataloader that loads edge list")
  .def(py::init<std::filesystem::path>())
  .def("name", &corelib::data::EdgeListLoader::name)
  .def("edgeListLength", &corelib::data::EdgeListLoader::edgeListLength)
  .def("__repr__", [](const corelib::data::EdgeListLoader &e)
    { return fmt::format("EdgeListLoader Object <{}>", e.name());});

  py::class_<corelib::data::FeatureLoader>(data, "FeatureLoader", "A feature Loader that loads random features to the graph")
  .def(py::init<const corelib::data::EdgeListLoader *, std::filesystem::path>())
  .def("name", &corelib::data::FeatureLoader::name)
  .def("edgeFeaturesLength", &corelib::data::FeatureLoader::edgeFeaturesLength)
  .def("verticesFeaturesLength", &corelib::data::FeatureLoader::verticesFeaturesLength)
  .def("__repr__", [](const corelib::data::FeatureLoader &e)
    { return fmt::format("FeatureLoader Object <{}>", e.name());});

  py::class_<corelib::data::HostGraphData>(data, "HostGraphData", "The graph data for a subset of edge list")
  .def("__repr__", [](const corelib::data::HostGraphData &e) {
    return fmt::format("<HostGraphData>\n{}", e);
  });

  py::class_<corelib::data::SubPartition>(data, "SubPartition", "A sub-partition of edge list")
  .def("__repr__", [](const corelib::data::SubPartition &e) 
    { return fmt::format("<SubPartition: {} {} {}>", e.beg, e.end, e.n);
  });

  data.def("divideList", &corelib::data::divideList);
  data.def("removeMinor", &corelib::data::removeMinor);
  data.def("includeMinor", &corelib::data::includeMinor);

  py::class_<corelib::data::GraphDataLoader>(data, "GraphDataLoader", "A dataloader that loads graph data")
  .def(py::init<const corelib::data::EdgeListLoader *, const corelib::data::FeatureLoader *>())
  .def("createGraphData", &corelib::data::GraphDataLoader::createGraphData, 
    py::arg("beg") = 0, py::arg("end") = 0, py::arg("cache") = true)
  .def("createPartitionsData", &corelib::data::GraphDataLoader::createPartitionsData)
  .def("createPartitionsDataMajor", &corelib::data::GraphDataLoader::createPartitionsDataMajor)
  .def("createPartitionsDataMinor", &corelib::data::GraphDataLoader::createPartitionsDataMinor)
  .def("__repr__", [](const corelib::data::GraphDataLoader &e) {
    return fmt::format("<GraphDataLoader: {}>", e.name());
  });

  py::class_<corelib::data::DeviceGraphData>(data, "DeviceGraphData", "The graph data on a device")
  .def("__repr__", [](const corelib::data::DeviceGraphData &e) {
    return fmt::format("<DeviceGraphData>\n{}", e);
  });

  py::class_<corelib::data::GraphDataManager>(data, "GraphDataManager", "A class that move data asynchronously over multigpus")
  .def("allocBatch", &corelib::data::GraphDataManager::allocBatch)
  .def("alloc", &corelib::data::GraphDataManager::alloc)
  .def("moveAsync", &corelib::data::GraphDataManager::moveAsync)
  .def("waitMove", &corelib::data::GraphDataManager::waitMove)
  .def("__repr__", [](const corelib::data::GraphDataManager &) {return std::string("<GraphDataLoader>");});

  data.def("MakeGraphDataManager", corelib::data::MakeGraphDataManager);

  py::class_<corelib::data::HostMotifData>(data, "HostMotifData", "Motif Data on the host")
  .def(py::init<const std::filesystem::path &>())
  .def("__repr__", [] (const corelib::data::HostMotifData &e) {
    return fmt::format("<HostMotifData>\n{}", e);
  });

  py::class_<corelib::data::DeviceJobData>(data, "DeviceJobData", "Job Data on the Device")
  .def(py::init<const corelib::data::DeviceGraphData &, const corelib::data::HostMotifData &>())
  .def("__repr__", [] (const corelib::data::DeviceJobData &e) {
    return fmt::format("<DeviceJobData>\n{}", e);
  });

  py::class_<corelib::data::MineJob>(data, "MineJob", "A Job, contains everything to let a worker to mine")
  .def(py::init<const corelib::data::DeviceJobData *, int, int, int>(),
    py::arg("_data"), py::arg("_delta"), py::arg("_beg") = 0, py::arg("_end") = -1)
  .def("getWork", &corelib::data::MineJob::getWork)
  .def("setWork", &corelib::data::MineJob::setWork)
  .def("__repr__", [] (const corelib::data::MineJob &e) {
    return fmt::format("<MineJob: {}, {}, {}, {}>", e.device, e.delta, e.beg, e.end);
  });

  py::class_<corelib::GPUWorker>(corelib, "GPUWorker", "GPU worker class")
  .def("run", &corelib::GPUWorker::run)
  .def("take", &corelib::GPUWorker::take)
  .def("update_job", &corelib::GPUWorker::update_job)
  .def("timed_run", &corelib::GPUWorker::timed_run)
  .def("time", &corelib::GPUWorker::time)
  .def("count", &corelib::GPUWorker::count)
  .def("clear_time", &corelib::GPUWorker::clear_time)
  .def("pause", &corelib::GPUWorker::pause)
  .def("printEnum", &corelib::GPUWorker::printEnum)
  .def("__repr__", [] (const corelib::GPUWorker &w){
    return fmt::format("<GPUWorker>\n{}", w);
  });

  py::class_<corelib::GPUWorkerLibManager>(corelib, "GPUWorkerLibManager", "to manager GPU worker libraries")
  .def(py::init<>())
  .def("openLib", &corelib::GPUWorkerLibManager::openLib)
  .def("unloadLib", &corelib::GPUWorkerLibManager::unloadLib)
  .def("construct", &corelib::GPUWorkerLibManager::construct)
  .def("__repr__", [] (const corelib::GPUWorkerLibManager &e) {
    return fmt::format("<GPUWorkerLibManager>\n{}", e);
  });

  py::class_<corelib::SingleGPUExecution>(corelib, "SingleGPUExecution", "SingleGPUExecution uses a single GPU to do the mining")
  .def(py::init<const corelib::data::GraphDataManager *, int>(), py::arg("dataManager"), py::arg("device") = 0)
  .def("setHostGraph", &corelib::SingleGPUExecution::setHostGraph, py::arg("hostGraph"), py::arg("n") = -1)
  .def("setHostMotif", &corelib::SingleGPUExecution::setHostMotif)
  .def("setWorker", &corelib::SingleGPUExecution::setWorker)
  .def("run", &corelib::SingleGPUExecution::run)
  .def("__repr__", [] (const corelib::SingleGPUExecution &) {
    return std::string("<SingleGPUExeuction>");
  });

  py::class_<corelib::SingleGPUExecutionDiv>(corelib, "SingleGPUExecutionDiv", "SingleGPUExecutionDiv uses a single GPU to do the mining")
  .def(py::init<const corelib::data::GraphDataManager *, int, int>(), py::arg("dataManager"), py::arg("device") = 0, py::arg("div") = 64)
  .def("setHostGraph", &corelib::SingleGPUExecutionDiv::setHostGraph, py::arg("hostGraph"), py::arg("n") = -1)
  .def("setHostMotif", &corelib::SingleGPUExecutionDiv::setHostMotif)
  .def("setWorker", &corelib::SingleGPUExecutionDiv::setWorker)
  .def("run", &corelib::SingleGPUExecutionDiv::run)
  .def("__repr__", [] (const corelib::SingleGPUExecutionDiv &) {
    return std::string("<SingleGPUExeuction>");
  });

  py::class_<corelib::SingleGPUExecutionPause>(corelib, "SingleGPUExecutionPause", "SingleGPUExecutionPause uses a single GPU to do the mining")
  .def(py::init<const corelib::data::GraphDataManager *, int>(), py::arg("dataManager"), py::arg("device") = 0)
  .def("setHostGraph", &corelib::SingleGPUExecutionPause::setHostGraph, py::arg("hostGraph"), py::arg("n") = -1)
  .def("setHostMotif", &corelib::SingleGPUExecutionPause::setHostMotif)
  .def("setWorker", &corelib::SingleGPUExecutionPause::setWorker)
  .def("run", &corelib::SingleGPUExecutionPause::run)
  .def("__repr__", [] (const corelib::SingleGPUExecutionPause &) {
    return std::string("<SingleGPUExeuction>");
  });

  corelib.def("SingleGPURunBatch", &corelib::SingleGPURunBatch);

  py::class_<corelib::MultiGPUExecutionNaive>(corelib, "MultiGPUExecutionNaive", "A naive way to utilize multiple GPUs (without load balancing)")
  .def(py::init<const corelib::data::GraphDataManager *, int>())
  .def("setHostGraphs", &corelib::MultiGPUExecutionNaive::setHostGraphs)
  .def("setHostMotif", &corelib::MultiGPUExecutionNaive::setHostMotif)
  .def("setWorkers", &corelib::MultiGPUExecutionNaive::setWorkers)
  .def("run", &corelib::MultiGPUExecutionNaive::run)
  .def("numDevice", &corelib::MultiGPUExecutionNaive::numDevice)
  .def("__repr__", [] (const corelib::MultiGPUExecutionNaive &e) { 
    return fmt::format("<MultiGPUExecutionNaive: {}>", e.numDevice());
  });

  py::class_<corelib::MultiGPUExecutionDyn>(corelib, "MultiGPUExecutionDyn", "multi GPU execution with dynamic load balancing")
  .def(py::init<const corelib::data::GraphDataManager *, int>())
  .def("setHostGraphs", &corelib::MultiGPUExecutionDyn::setHostGraphs)
  .def("setHostMotif", &corelib::MultiGPUExecutionDyn::setHostMotif)
  .def("setWorkers", &corelib::MultiGPUExecutionDyn::setWorkers)
  .def("run", &corelib::MultiGPUExecutionDyn::run)
  .def("numDevice", &corelib::MultiGPUExecutionDyn::numDevice)
  .def("__repr__", [] (const corelib::MultiGPUExecutionDyn &e) { 
    return fmt::format("<MultiGPUExecutionDyn: {}>", e.numDevice());
  });

  py::class_<corelib::MultiGPUExecutionDynPause>(corelib, "MultiGPUExecutionDynPause", "multi GPU execution with dynamic load balancing")
  .def(py::init<const corelib::data::GraphDataManager *, int>())
  .def("setHostGraphs", &corelib::MultiGPUExecutionDynPause::setHostGraphs)
  .def("setHostMotif", &corelib::MultiGPUExecutionDynPause::setHostMotif)
  .def("setWorkers", &corelib::MultiGPUExecutionDynPause::setWorkers)
  .def("run", &corelib::MultiGPUExecutionDynPause::run)
  .def("numDevice", &corelib::MultiGPUExecutionDynPause::numDevice)
  .def("__repr__", [] (const corelib::MultiGPUExecutionDynPause &e) { 
    return fmt::format("<MultiGPUExecutionDynPause: {}>", e.numDevice());
  });


  corelib.def("allCount", &corelib::allCount);

  
}