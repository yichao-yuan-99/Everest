import yaml
import re
import sys
from . import locations

class Options:
  def __init__(self, opts: dict = {
      'gpu' : 1,
      'warpCentric' : True,
      'warpBalancing' : True,
      'tailExpansion' : True,
    }) -> None:

    self.opts = opts.copy()
    if self.multigpu():
      self.opts['multigpu'] = True

  def __str__(self) -> str:
    return f'Options:\n {self.opts}'

  def __repr__(self) -> str:
    return str(self)

  def gpu(self):
    return self.opts['gpu']
  
  def multigpu(self):
    return self.opts['gpu'] > 1

  def enumerate(self):
    return 'enumerate' in self.opts
  
  def enumNum(self):
    return int(self.opts['enumerate'])

  def update(self, rhs):
    return self.opts.update(rhs.opts)

class AntiEdge:
  def __init__(self, antiEdge) -> None:
    self.anti = antiEdge

  def array(self):
    s = [str(i).lower() for i in self.anti]
    s = s[1:]
    nums = ', '.join(s)
    n = len(s)
    return f'__device__ static const bool antiEdge[{n}] = {{ {nums} }};\n'

  def offset(self):
    o = []
    for i in range(1, len(self.anti)):
      c = 0
      for j in reversed(range(0, i)):
        if self.anti[j]:
          c = c + 1
        else:
          break
      o.append(c)
    oo = [str(i) for i in o]
    nums = ', '.join(oo)
    return f'__device__ static const int antiEdgeOffset[{len(o)}] = {{ {nums} }};\n'
  
  def prev(self):
    o = []
    for i in range(1, len(self.anti)):
      c = 0
      for j in reversed(range(0, i)):
        if not self.anti[j]:
          o.append(j)
          break
    oo = [str(i) for i in o] 
    nums = ', '.join(oo)
    return f'__device__ static const int antiEdgePrev[{len(o)}] = {{ {nums} }};\n'
  
  def isAntiEdge(self, i):
    return self.anti[i]

class TemporalConstraints:
  def __init__(self, temporal) -> None:
    self.tem = temporal 
  
  def array(self):
    s = [str(i) for i in self.tem]
    nums = ', '.join(s)
    n = len(self.tem)
    return f'__device__ static const int temporalArr[{n}] = {{ {nums} }};\n'

class FilterConstraints:
  def __init__(self, cond) -> None:
    self.cond = cond
    self.cond = self.replace_e_with_edgeFeature(self.cond)

  def replace_e_with_edgeFeature(self, input_string):
    def replace_match(match_obj):
      return f'edgeFeature[eid{match_obj.group(1)}]'
    return re.sub(r'e(\d+)', replace_match, input_string)
  
  def replace_n_with_nodeFeature(self, input_string):
    def replace_match(match_obj):
      return f'nodeFeature[m{match_obj.group(1)}]'
    return re.sub(r'n(\d+)', replace_match, input_string)

class Constraints:
  def __init__(self, constraints: dict) -> None:
    self.cons = constraints
    if 'temporal' in constraints:
      self.temporal = TemporalConstraints(constraints['temporal'])
    if 'antiEdge' in constraints:
      self.antiEdge = AntiEdge(constraints['antiEdge'])
    if 'filter' in constraints:
      self.filter = FilterConstraints(constraints['filter'])
    

  def __str__(self) -> str:
    return f'Constraints:\n f{self.cons}'

  def __repr__(self) -> str:
    return str(self)

  def isDelta(self):
    return 'delta' in self.cons
  
  def isTemporal(self):
    return 'temporal' in self.cons
  
  def isAnti(self):
    return 'antiEdge' in self.cons

  def isFilter(self):
    return 'filter' in self.cons

  def delta(self):
    return self.cons['delta']

  def genConsTemplates(self):
    # generate vertexConsFunc
    vertexConsFuncs = ''
    vertexConsCallFirst = ''
    vertexConsCallLoop = ''
    if self.hasVertexCons():
      vertexConsFuncs = self.renderVertexConsFunc()
      vertexConsCallFirst = self.renderUniformVertexCheckFirst() # !!
      vertexConsCallLoop = self.renderUniformVertexCheckLoop() # !!
    with open(locations.consRoot() + '/vertexConsFuncs.snp', 'w') as file:
      file.write(vertexConsFuncs)
    with open(locations.consRoot() + '/vertexConsCallFirst.snp', 'w') as file:
      file.write(vertexConsCallFirst)
    with open(locations.consRoot() + '/vertexConsCallLoop.snp', 'w') as file:
      file.write(vertexConsCallLoop)

    # generate edgeConsFunc
    edgeConsFuncs = ''
    edgeConsCallFirst = ''
    edgeConsCallLoop = ''
    if self.hasEdgeCons():
      edgeConsFuncs = self.renderEdgeConsFunc()
      edgeConsCallFirst = self.renderUniformEdgeCheckFirst() # !!
      edgeConsCallLoop = self.renderUniformEdgeCheckLoop() # !!
    with open(locations.consRoot() + '/edgeConsFuncs.snp', 'w') as file:
      file.write(edgeConsFuncs)
    with open(locations.consRoot() + '/edgeConsCallFirst.snp', 'w') as file:
      file.write(edgeConsCallFirst)
    with open(locations.consRoot() + '/edgeConsCallLoop.snp', 'w') as file:
      file.write(edgeConsCallLoop)


  def hasVertexCons(self):
    c = 0
    for k, v in self.cons.items():
      if re.search('node*', k):
        c = c + 1

    return c > 0

  def hasEdgeCons(self):
    c = 0
    for k, v in self.cons.items():
      if re.search('edge*', k):
        c = c + 1

    return c > 0
    
  def uniVertexCons(self):
    c = 0
    h = 0
    for k, v in self.cons.items():
      if re.search('node*', k):
        c = c + 1
      if k == 'node':
        h = 1
    
    return c == 1 and h == 1

  def uniEdgeCons(self):
    c = 0
    h = 0
    for k, v in self.cons.items():
      if re.search('edge*', k):
        c = c + 1
      if k == 'edge':
        h = 1
    
    return c == 1 and h == 1

  def renderVertex(self, name, cond):
    out = f'__device__ static bool {name}(int id, const int *nodeFeature) {{\n'
    out += '  auto f = nodeFeature[id];\n'
    out += '  return ' + cond + ';\n'
    out += '}\n'
    return out
    

  def renderUniformVertex(self):
    return self.renderVertex('nodeConstraintUni', self.cons['node'])
  
  def renderDifferentVertex(self):
    c = 0
    for k, v in self.cons.items():
      if re.search('node*', k):
        c = c + 1
    assert(c > 1)

    out = ''
    for i in range(0, c):
      out += self.renderVertex(f'nodeConstraint{i}', self.cons[f'node{i}']) 

    out += 'typedef bool (*NODECONS_FUNC)(int, const int*);\n'
    out += '__device__ NODECONS_FUNC nodeConstraints[] = {'
    for i in range(0, c):
      out += f' nodeConstraint{i},'
    out += '};\n'
    return out
    

  def renderUniformVertexCheckFirst(self):
    # only consider uniform case
    if self.uniVertexCons():
      out = 'fcheck = (fcheck && nodeConstraintUni(Eg[tid].u, nodeFeature));\n'
      out += 'fcheck = (fcheck && nodeConstraintUni(Eg[tid].v, nodeFeature));\n'
    else:
      out = 'fcheck = (fcheck && nodeConstraints[0](Eg[tid].u, nodeFeature));\n'
      out += 'fcheck = (fcheck && nodeConstraints[1](Eg[tid].v, nodeFeature));\n'
    return out 

  def renderUniformVertexCheckLoop(self):
    # only consider uniform case
    if self.uniVertexCons():
      out = 'checked &= nodeConstraintUni(node, nodeFeature);\n'
    else:
      out = 'checked &= nodeConstraints[mi.mappedNodes](node, nodeFeature);\n'
    return out 


  def renderEdge(self, name, cond):
    out = f'__device__ static bool {name}(int id, const int *edgeFeature) {{\n'
    out += '  auto f = edgeFeature[id];\n'
    out += '  return ' + cond + ';\n'
    out += '}\n'
    return out

  def renderUniformEdge(self):
    return self.renderEdge("edgeConstraintUni", self.cons['edge'])

  def renderDifferentEdge(self):
    c = 0
    for k, v in self.cons.items():
      if re.search('edge*', k):
        c = c + 1
    assert(c > 1)
    
    out = ''
    for i in range(0, c):
      out += self.renderEdge(f'edgeConstraint{i}', self.cons[f'edge{i}']) 

    out += 'typedef bool (*EDGECONS_FUNC)(int, const int*);\n'
    out += '__device__ EDGECONS_FUNC edgeConstraints[] = {'
    for i in range(1, c):
      out += f' edgeConstraint{i},'
    out += '};\n'
    return out
    

  def renderUniformEdgeCheckFirst(self):
    if self.uniEdgeCons():
      out = 'fcheck &= edgeConstraintUni(tid, edgeFeature);\n'
    else:
      out = 'fcheck &= edgeConstraints[level](tid, edgeFeature);\n'
    return out 

  def renderUniformEdgeCheckLoop(self):
    if self.uniEdgeCons():
      out = 'checked &= edgeConstraintUni(idx, edgeFeature);\n'
    else:
      out = 'checked &= edgeConstraint0(idx, edgeFeature);\n'
    return out 

  def renderVertexConsFunc(self):
    if self.uniVertexCons():
      return self.renderUniformVertex()
    else:
      return self.renderDifferentVertex() # only consider uniform case for now

  def renderEdgeConsFunc(self):
    if self.uniEdgeCons():
      return self.renderUniformEdge()
    else:
      return self.renderDifferentEdge()


class Query:
  def __init__(self, pathToQuery: str) -> None:
    with open(pathToQuery, 'r') as file:
      self.queryFile = yaml.safe_load(file)
      self.name = pathToQuery
      self.cons = Constraints(self.queryFile['constraints'])
      defaultOpts = Options()
      defaultOpts.update(Options(self.queryFile['options']))
      self.opts = defaultOpts
      self.queryFile['pattern'] = locations.inputMotifsRoot() + f'/{self.pattern()}'
      self.queryFile['graph'] = locations.inputGraphRoot() + f'/{self.graph()}'

  def __str__(self) -> str:
    return f'Query {self.name}:\n {self.queryFile}'

  def __repr__(self) -> str:
    return str(self)

  def pattern(self) -> str:
    return self.queryFile['pattern']

  def graph(self) -> str:
    return self.queryFile['graph']

  def constraints(self):
    return self.cons

  def options(self):
    return self.opts