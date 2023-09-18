import logging

from .utils import *
from .KernelGen import *

from . import locations, Query, utils, KernelGen, core

from timeit import default_timer as timer
import numpy as np

class Result:
  def __init__(self, line) -> None:
    self.count, self.microsec = [int(s) for s in line.split(',')]

  def __str__(self) -> str:
    return f'count: {self.count}, microsec: {self.microsec}'

  def __repr__(self) -> str:
    return str(self)

class Pipeline:
  def __init__(self, pathToQuery) -> None:
    self.query = Query.Query(pathToQuery)
    
  def run(self):
    # constraints generate templates
    self.query.cons.genConsTemplates()
    codeGen = KernelGen.KernelGen(self.query.pattern(), self.query)
    code = codeGen.generate()
    createDirIfNotExist(locations.dynLibSourceRoot())
    sourcePath = locations.dynLibSource() 
    with open(sourcePath, "w") as file:
        file.write(code)

  def compile(self):
    compilePlugin()
    
class HeavyRes:
  def __init__(self, graph: str, gpu: int) -> None:
    self.graph = graph
    self.eloader = core.data.EdgeListLoader(self.graph)
    self.floader = core.data.FeatureLoader(self.eloader, locations.randFilePath())
    self.gloader = core.data.GraphDataLoader(self.eloader, self.floader)
    self.gpu = gpu 
    if gpu > 1:
      subpartition = core.data.divideList(self.eloader, 3600, self.gpu)
      self.major = self.gloader.createPartitionsDataMajor(subpartition)
    else:
      self.hostGraph = self.gloader.createGraphData()
      

class Mining:
  def __init__(self, pathToGraph, gpu):
    self.heavy = HeavyRes(pathToGraph, gpu)
    self.libManager = core.GPUWorkerLibManager()
    self.dataManager = core.data.MakeGraphDataManager(gpu)
  
  def makePlugin(self, query):
    # constraints generate templates
    query.cons.genConsTemplates()
    codeGen = KernelGen.KernelGen(query.pattern(), query)
    code = codeGen.generate()
    createDirIfNotExist(locations.dynLibSourceRoot())
    sourcePath = locations.dynLibSource() 
    with open(sourcePath, "w") as file:
        file.write(code)
    compilePlugin()

  def processSingle(self, query: Query.Query):
    hostMotif = core.data.HostMotifData(query.pattern())
    w = self.libManager.construct(utils.libFileToLibName(locations.dynLibPath()), 0)
    exec = core.SingleGPUExecution(self.dataManager)
    exec.setHostGraph(self.heavy.hostGraph)
    exec.setHostMotif(hostMotif)
    exec.setWorker(w)

    start = timer()
    exec.run(query.constraints().delta())
    end = timer()
    cn = w.count()
    tm = w.time()

    print("@@@ [output starts]", flush=True)
    if not query.options().enumerate():
      print(f"{w.count()}, {w.time()}, {end - start}")
    else:
      e = w.printEnum(query.options().enumNum())
      ea = np.array(e)
      _, _, allE = InspectMotif(query.pattern())
      ea = ea.reshape(-1, allE)
      for k in ea:
        print(' '.join(map(str, k)))
    print("@@@ [output ends]")

    del w
    return (cn, tm, (end - start))

  def processMulti(self, query: Query.Query):
    hostMotif = core.data.HostMotifData(query.pattern())
    ws = [self.libManager.construct(utils.libFileToLibName(locations.dynLibPath()), d) for d in range(self.heavy.gpu)]

    subpartition = core.data.divideList(self.heavy.eloader, query.constraints().delta(), self.heavy.gpu)
    minor = self.heavy.gloader.createPartitionsDataMinor(subpartition)
    core.data.includeMinor(self.heavy.major, minor)
    exec = core.MultiGPUExecutionDyn(self.dataManager, self.heavy.gpu)
    exec.setHostGraphs(self.heavy.major, subpartition)
    exec.setHostMotif(hostMotif)
    exec.setWorkers(ws)

    start = timer()
    exec.run(query.constraints().delta())
    end = timer()

    r = core.allCount(ws)
    times = [w.time() for w in ws]

    print("@@@ [output starts]", flush=True)
    if not query.options().enumerate():
      print(f'{r}, {max(times)}, {end - start}')
    else:
      _, _, allE = InspectMotif(query.pattern())
      out = [np.array(w.printEnum(query.options().enumNum())) for w in ws]
      out = [ea.reshape(-1, allE) for ea in out]

      for ea in out:
        for k in ea:
          print(' '.join(map(str, k)))
          
    print("@@@ [output ends]")

    core.data.removeMinor(self.heavy.major)
    del ws
    return (r, max(times), (end - start))
  
  def processQuery(self, query: Query.Query):
    if query.graph() != self.heavy.graph or query.options().gpu() != self.heavy.gpu:
      print(f"cannot process query {query}")
    self.makePlugin(query)
    self.libManager.openLib(locations.dynLibPath())
    if self.heavy.gpu > 1:
      return self.processMulti(query)
    else:
      return self.processSingle(query)

  def run(self, pathToQuery):
    query = Query.Query(pathToQuery)
    return self.processQuery(query)
      

class MiningSingle(Mining):
  def __init__(self, pathToQuery):
    query = Query.Query(pathToQuery)
    super().__init__(query.graph(), query.options().gpu())
    self.query = query

  def run(self):
    return self.processQuery(self.query)



class MiningMultiAsIfSingle(Mining):
  def __init__(self, pathToQuery):
    query = Query.Query(pathToQuery)
    super().__init__(query.graph(), query.options().gpu())
    self.query = query

  def processQuery(self, query: Query):
    self.makePlugin(query)
    self.libManager.openLib(locations.dynLibPath())
    hostMotif = core.data.HostMotifData(query.pattern())
    w = self.libManager.construct(utils.libFileToLibName(locations.dynLibPath()), 0)

    subpartition = core.data.divideList(self.heavy.eloader, query.constraints().delta(), self.heavy.gpu)
    minor = self.heavy.gloader.createPartitionsDataMinor(subpartition)
    core.data.includeMinor(self.heavy.major, minor)

    start = timer()
    core.SingleGPURunBatch(self.dataManager, self.heavy.major, hostMotif, w, query.constraints().delta())
    end = timer()

    print(f"{w.count()}, {w.time()}, {end - start}")
    del w
    return (w.count(), w.time(), (end - start))
  
  def run(self):
    return self.processQuery(self.query)
    

class MiningMultiNaive(Mining):
  def __init__(self, pathToQuery):
    query = Query.Query(pathToQuery)
    super().__init__(query.graph(), query.options().gpu())
    self.query = query

  def processQuery(self, query: Query):
    self.makePlugin(query)
    self.libManager.openLib(locations.dynLibPath())
    hostMotif = core.data.HostMotifData(query.pattern())
    ws = [self.libManager.construct(utils.libFileToLibName(locations.dynLibPath()), d) for d in range(self.heavy.gpu)]

    subpartition = core.data.divideList(self.heavy.eloader, query.constraints().delta(), self.heavy.gpu)
    minor = self.heavy.gloader.createPartitionsDataMinor(subpartition)
    core.data.includeMinor(self.heavy.major, minor)
    exec = core.MultiGPUExecutionNaive(self.dataManager, self.heavy.gpu)
    exec.setHostGraphs(self.heavy.major, subpartition)
    exec.setHostMotif(hostMotif)
    exec.setWorkers(ws)

    start = timer()
    exec.run(query.constraints().delta())
    end = timer()

    r = core.allCount(ws)
    times = [w.time() for w in ws]
    print(f'{r}, {max(times)}, {end - start}')

    core.data.removeMinor(self.heavy.major)
    del ws
    return (r, max(times), (end - start))

  def run(self):
    return self.processQuery(self.query)

