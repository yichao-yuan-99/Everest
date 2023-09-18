import os

def projRoot():
  t = os.path.abspath(__file__)  
  t = os.path.dirname(t)
  t = os.path.dirname(t)
  t = os.path.dirname(t)
  return t

def dataRoot():
  return projRoot() + '/data'

def graphRoot():
  return dataRoot() + '/target-graphs'

def patternRoot():
  return dataRoot() + '/4p-vertices'

def buildDir():
  return projRoot() + '/build'

def thisPkgDir():
  return projRoot() + '/libs'

def systemRoot():
  return projRoot() + '/system'

def dynLibSourceRoot():
  return systemRoot() + '/.tmp'

def dynLibSource():
  return dynLibSourceRoot() + '/GPUWorkerDyn.cu'

def dynLibRoot():
  return systemRoot() + '/plugins'

def dynLibPath():
  return dynLibRoot() + '/libDYN.so'
  
def templateRoot():
  return systemRoot() + '/templates'

def consRoot():
  return templateRoot() + '/constraints'

def randFilePath():
  return '/home/yichaoy/data1/random/5GRand_0_1B'

def queryRoot():
  return systemRoot() + '/queries'

def genQueryPath():
  return queryRoot() + '/generatedQuery.yaml'

def inputMotifsRoot():
  return projRoot() + '/inputs/motifs'

def inputGraphRoot():
  return projRoot() + '/inputs/graphs'