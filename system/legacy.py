from tmine import core, locations, utils
import sys

# run lagacy code

file = sys.argv[1]
motifFile = sys.argv[2] 
delta = int(sys.argv[3])
l = sys.argv[4]

randFile = "/home/yichaoy/data1/random/5GRand_0_1B"
libFile = locations.buildDir() + f"/plugins/lib{l}.so"
libName = utils.libFileToLibName(libFile)

eloader = core.data.EdgeListLoader(file)
floader = core.data.FeatureLoader(eloader, randFile)
gloader = core.data.GraphDataLoader(eloader, floader)

hostGraph = gloader.createGraphData()

hostMotif = core.data.HostMotifData(motifFile)

dataManager = core.data.MakeGraphDataManager(4)

libManager = core.GPUWorkerLibManager()

libManager.openLib(libFile)
w = libManager.construct(libName, 0)
t = w.time()

exec = core.SingleGPUExecution(dataManager)

exec.setHostGraph(hostGraph)
exec.setHostMotif(hostMotif)
exec.setWorker(w)

r = exec.run(delta)

print(f"finish, count: {w.count()}, time: {w.time()}")

del w # all workers need to be released before libManager is released