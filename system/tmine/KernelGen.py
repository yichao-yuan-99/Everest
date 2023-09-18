import re
import os
from . import locations
from .Query import *

def reduce_blank_lines(text):
  return re.sub(r'\n\s*\n', '\n\n', text)

# numLevel = valid Stack entry
# maxNumLevel = numem - 2
# max valid stack entry = numem - 2
# mnode_p[i] means the node that is mapped at level i
def InspectMotif(pathToMotif):
  """stack size = M - 1; map size = mapped node before last edge"""
  mnode = []
  mnode_p = []
  l = 0
  allE = 0
  with open(pathToMotif, 'r') as f:
    for line in f:
      allE = allE + 1
      l = l + 1
      u, v, t = line.split()
      mnode_p.append(len(mnode))
      if not u in mnode:
        mnode.append(u)
      if not v in mnode:
        mnode.append(v)
  return mnode_p[1:], (l - 2), allE # mapped size and max stack size

class KernelGen:
  def __init__(self, pathToMotif, query : Query) -> None:
    options = query.options()
    cons = query.constraints()
    self.cons = cons
    self.options = options
    self.query = query
    # the threshold for sharable levels are 3 (i.e. 5 edges)
    self.MAXSHARED = 99 # do not consider very large motif for now 

    # motif dependent parameters
    Mapped, maxLevel, allE = InspectMotif(pathToMotif)
    self.Mapped = Mapped
    self.maxLevel = maxLevel
    self.allE = allE

    self.sharedLevel = min(self.MAXSHARED, self.maxLevel)
    self.sharedMap = self.Mapped[self.sharedLevel]
    self.ownLevel = max(0, self.sharedLevel - self.MAXSHARED)

    self.regMap = self.sharedMap
    self.localMap = Mapped[-1] - self.regMap
    self.regStack = self.sharedLevel
    self.localStack = self.ownLevel

    self.true = True
    self.warpCentric = options.opts['warpCentric']
    self.warpBalancing = options.opts['warpBalancing']
    self.tailExpansion = options.opts['tailExpansion']

    self.keepEid = False
    self.keepPrevEid = False
    self.fineGrained = False
    self.antiEdge = False
    self.filter = False
    if cons.isTemporal():
      self.keepEid = True
      self.keepPrevEid = True
      self.fineGrained = True
    
    if cons.isAnti():
      self.keepEid = True
      self.keepPrevEid = True
      self.antiEdge = True

    if cons.isFilter():
      self.keepEid = True
      self.keepPrevEid = True
      self.filter = True
    
    self.enumerate = False
    self.enum_NUM = 0
    if options.enumerate():
      self.enumerate = True
      self.keepEid = True
      self.enum_NUM = options.enumNum()

  def __str__(self):
    return (
      f"MAXSHARED: {self.MAXSHARED}\n"
      f"Mapped: {self.Mapped}\n"
      f"maxLevel: {self.maxLevel}\n"
      f"sharedLevel: {self.sharedLevel}\n"
      f"sharedMap: {self.sharedMap}\n"
      f"ownLevel: {self.ownLevel}\n"
      f"regMap: {self.regMap}\n"
      f"localMap: {self.localMap}\n"
      f"regStack: {self.regStack}\n"
      f"localStack: {self.localStack}\n"
      f"true: {self.true}\n"
      f"warpCentric: {self.warpCentric}\n"
      f"warpBalancing: {self.warpBalancing}\n"
      f"tailExpansion: {self.tailExpansion}\n"
    )

  def __repr__(self):
    return str(self)

  def attrLookup(self, name):
    return getattr(self, name)

  def generate(self):
    line = r'// @@@{expandTemplate top.snp}'
    out = self.expandLine(line)
    out = reduce_blank_lines(out)
    return out 

  def expandLine(self, line):
    pad = line.find('//')
    command = re.search(r".*\@\@\@\{(.*)\}", line).group(1)
    tokens = command.split()
    for i, t in enumerate(tokens):
      if t[0] == '$':
        tokens[i] = self.attrLookup(t[1:])
    out = self.processTokens(tokens)
    lines = out.split('\n')
    ll = [' ' * pad + l for l in lines]
    out = '\n'.join(ll)
    out = reduce_blank_lines(out)
    return out

  def processTokens(self, tokens):
    out = self.attrLookup(tokens[0])(*tokens[1:])
    return out

  def expandTemplate(self, template : str):
    filename = locations.templateRoot() + '/' + template
    out = []
    if os.path.exists(filename):
      with open(filename, "r") as file:
          for line in file:
            r = re.search(r".*\@\@\@\{(.*)\}", line)
            if r is not None:
              oo = self.expandLine(line)
              out += oo.split('\n')
            else:
              out.append(line.rstrip())
    else:
      print(f"Error: File '{filename}' not found.")
    out = '\n'.join(out)
    return out

  def enableIf(self, f, s, *tokens):
    tf = True
    if f == 'not':
      tf = not s
    else:
      tf = f 
      tokens = [s] + [i for i in tokens]

    if tf:
      return self.processTokens(tokens)
    else:
      return ''

  ### Code to generate text ###

  def showText(self, *tokens):
    return ' '.join(tokens)
  
  def includeHeader(self, header):
    headerDir = locations.projRoot() + '/include'
    return f'#include "{headerDir}/{header}"'


  def LevelDepStatesDeclare(self):
    regMap = self.regMap
    localMap = self.localMap
    regStack = self.regStack
    localStack = self.localStack

    begs = [f'stackbeg{i} = 0' for i in range(regStack)]
    ends = [f'stackend{i} = 0' for i in range(regStack)]
    ms = [f'm{i} = 0' for i in range(regMap)]
    out = ''
    out += 'int ' + ', '.join(begs) + ';\n'
    out += 'int ' + ', '.join(ends) + ';\n'
    out += 'int ' + ', '.join(ms) + ';\n'
    if localMap > 0:
      out += f'int localM[{localMap}];\n'
    if localStack > 0:
      out += f'int localBeg[{localStack}];\n'
      out += f'int localEnd[{localStack}];\n'

    if self.keepEid:
      eids = [f'eid{i} = 0' for i in range(self.allE - 1)]
      tmp = ', '.join(eids)
      out += f'int {tmp};\n'
    return out

  def LevelDepStatesArgPass(self):
    regMap = self.regMap
    localMap = self.localMap
    regStack = self.regStack
    localStack = self.localStack

    begs = [f'stackbeg{i},' for i in range(regStack)]
    ends = [f'stackend{i},' for i in range(regStack)]
    tls = [f'tl{i},' for i in range(regStack)]
    ms = [f'm{i},' for i in range(regMap)]
    out = ''
    out += ' '.join(begs) + '\n'
    out += ' '.join(ends) + '\n'
    out += ' '.join(ms) + '\n'
    if localMap > 0:
      out += 'localM,\n'
    if localStack > 0:
      out += 'localBeg, localEnd,\n'

    if self.keepEid:
      eids = [f'eid{i},' for i in range(self.allE - 1)]
      out += ' '.join(eids) + '\n'
      
    return out

  def LevelDepStatesArgList(self):
    regMap = self.regMap
    localMap = self.localMap
    regStack = self.regStack
    localStack = self.localStack

    begs = [f'int stackbeg{i},' for i in range(regStack)]
    ends = [f'int stackend{i},' for i in range(regStack)]
    ms = [f'int m{i},' for i in range(regMap)]
    out = ''
    out += ' '.join(begs) + '\n'
    out += ' '.join(ends) + '\n'
    out += ' '.join(ms) + '\n'
    if localMap > 0:
      out += f'int *localM,\n'
    if localStack > 0:
      out += f'int *localBeg,\n'
      out += f'int *localEnd,\n'

    if self.keepEid:
      eids = [f'int eid{i},' for i in range(self.allE - 1)]
      out += ' '.join(eids) + '\n'
    return out

  def LevelDepStatesDump(self):
    regMap = self.regMap
    regStack = self.regStack

    out = ''
    for i in range(regStack):
      out += f'offload[offpos].stackbeg{i} = stackbeg{i};\n'
      out += f'offload[offpos].stackend{i} = stackend{i};\n'

    for i in range(regMap):
      out += f'offload[offpos].m{i} = m{i};\n'

    if self.keepEid:
      for i in range(self.allE - 1):
        out += f'offload[offpos].eid{i} = eid{i};\n'

    return out

  def LevelDepStatesLoad(self):
    regMap = self.regMap
    regStack = self.regStack

    out = ''
    for i in range(regStack):
      out += f'stackbeg{i} = offload[loc].stackbeg{i} + offset;\n'

    for i in range(regStack):
      out += f'stackend{i} = min(stackbeg{i} + 1, offload[loc].stackend{i});\n'

    for i in range(regMap):
      out += f'm{i} = offload[loc].m{i};\n'

    if self.keepEid:
      for i in range(self.allE - 1):
        out += f'eid{i} = offload[loc].eid{i};\n'

    return out

  def LevelDepStatesShare(self):
    regMap = self.regMap
    regStack = self.regStack
    localStack = self.localStack

    def isAnti(i):
      return self.antiEdge and self.cons.antiEdge.isAntiEdge(i)

    # !!! not consider local states and time limits
    out = ''
    for i in range(regStack):
      if not isAnti(i + 1):
        out += f'stackbeg{i} = __shfl_sync(0xffffffff, stackbeg{i}, src);\n'

    for i in range(regStack):
      if not isAnti(i + 1):
        out += f'stackend{i} = __shfl_sync(0xffffffff, stackend{i}, src);\n'

    for i in range(regMap):
      out += f'm{i} = __shfl_sync(0xffffffff, m{i}, src);\n'

    if self.keepEid:
      for i in range(self.allE - 1):
        out += f'eid{i} = __shfl_sync(0xffffffff, eid{i}, src);\n'

    if localStack == 0:
      if self.antiEdge:
        out += 'if (!antiEdge[level]) '
      out += 'beg += r0e;\n'

    for i in range(regStack):
      if not isAnti(i + 1):
        out += f'stackbeg{i} += r0e;\n'

    out += 'if (src != laneid) {\n'
    if localStack == 0:
      if self.antiEdge:
        out += '  if (!antiEdge[level])'
      out += '  end = min(end, beg + 1);\n'

    for i in range(regStack):
      if not isAnti(i + 1):
        out += f'  stackend{i} = min(stackend{i}, stackbeg{i} + 1);\n'
    
    if self.antiEdge:
      out += f'  if (antiEdge[level]) bypass = true;\n'

    out += '}\n'
    return out
      

  def shrinkEnd(self, beg, end, tl):
    return f"""    stackiob >>= 1;
    i_S = stackiob & 1 ? outEdgesV : inEdgesV;
    probeEnd(i_S, {beg}, {end}, {tl}, Eg);
    width = max(width, {end} - {beg});\n"""

  def shrinkStack(self):
    regStack = self.regStack
    localStack = self.localStack

    out = 'switch(level) {\n'
    for i in reversed(range(localStack)):
      out += f'  case {i + regStack + 1}:\n'
      tl = 'tl'
      if self.fineGrained:
        tl = f'Eg[eid{i}].t + temporalArr[{i}]'
      out += self.shrinkEnd(f'localBeg[{i}]', f'localEnd[{i}]', tl)

    for i in reversed(range(regStack)):
      out += f'  case {i + 1}:\n'
      tl = 'tl'
      if self.fineGrained:
        tl = f'Eg[eid{i}].t + temporalArr[{i}]'
      out += self.shrinkEnd(f'stackbeg{i}', f'stackend{i}', tl)

    out += '}\n'
    return out


  # The node map is described by <regMap, localMap>
  def MappingLookUp(self, key: str, to: str):
    regMap = self.regMap
    localMap = self.localMap

    out = f'switch ({key}) {{\n'
    for i in range(regMap):
      out += f'  case {i}: {to} = m{i};'
      if i != regMap - 1 or localMap != 0:
        out += ' break;\n'
      else:
        out += '\n'
    if localMap != 0:
      out += f'  default: {to} = localM[{key} - {regMap}];\n'
    out += f'}}\n'
    return out

  def PushMapping(self):
    regMap = self.regMap
    localMap = self.localMap

    out = 'switch (mi.mappedNodes) {\n'
    for i in range(2, regMap):
      out += f'  case {i}: m{i} = node;\n'

    if localMap != 0:
      out += f'  default: localM[mi.mappedNodes - {regMap}] = node;\n'
    out += '}\n'
    return out

  def PushStack(self):
    regStack = self.regStack
    localStack = self.localStack

    out = 'switch (level) {\n'
    for i in range(regStack):
      out += f'  case {i}: stackbeg{i} = beg; stackend{i} = end;\n'

    if localStack != 0:
      out += f'  default: localBeg[level - {regStack}] = beg; localEnd[level - {regStack}] = end;\n'
      
    out += '}\n'
    return out

  def PopStack(self):
    regStack = self.regStack
    localStack = self.localStack

    out = 'switch (level) {\n'
    for i in range(regStack):
      out += f'  case {i}: beg = stackbeg{i}; end = stackend{i}; break;\n'

    if localStack != 0:
      out += f'  default: beg = localBeg[level - {regStack}]; end = localEnd[level - {regStack}];\n'
      
    out += '}\n'
    return out

  def MappingCheckingSame(self):
    regMap = self.regMap
    localMap = self.localMap

    out = 'switch (mi.constraintNode) {\n'
    for i in range(regMap):
      out += f'  case {i}: checked = (m{i} == node); break;\n'

    if localMap != 0:
      out += f'  default: checked = (localM[mi.constraintNode - {regMap}] == node);\n'
    out += '}\n'
    return out

  def MappingCheckingNotSame(self):
    regMap = self.regMap
    localMap = self.localMap

    out = 'switch (mi.mappedNodes) {\n'
    if regMap > 2:
      out += f'  case {regMap}: checked = (m{regMap-1} != node);\n'
    for i in reversed(range(3, regMap)):
      out += f'  case {i}: checked = checked && (m{i - 1} != node);\n'
    out += f'  case 2: checked = checked && (m1 != node) && (m0 != node);\n'
    out += '}\n'

    if localMap > 0:
      out += 'if (checked) {\n'
      out += f'  for (int i = 0; i < {localMap}; i++) {{\n'
      out += '    checked = checked && (localM[i] != node);\n'
      out += '  }\n'
      out += '}\n'

    return out

  def StoreIGtoEID(self):
    out = 'switch (level) {\n'
    for i in range(self.maxLevel):
      out += f'  case {i + 1} : eid{i + 1} = i_g; break;\n'
    out += '}\n'
    return out
  
  def fineGrained_arr(self):
    out = f'constexpr int DELTA = {self.cons.delta()};\n'
    out += self.cons.temporal.array()
    return out

  def fineGrained_firstTl(self):
    return f'tl = Eg[tid].t + temporalArr[0];\n'

  def fineGrained_calcTl(self):
    return f'tl = min(temporalArr[level] + Eg[i_g].t, Eg[eid0].t + DELTA);\n'


  def keepPrevEid_bt(self):
    if not self.antiEdge:
      out = 'switch (level) {\n'
    else:
      out = 'switch (antiEdgePrev[level]) {\n'

    for i in range(self.maxLevel):
      out += f'  case {i} : i_g = eid{i}; break;\n'
    out += '}\n'
    return out

  def keepPrevEid_next(self):
    out = '' 
    if self.antiEdge:
      out = 'switch (antiEdgePrev[level]) {\n'
      for i in range(1, self.allE - 1):
        out += f'  case {i}: i_g = eid{i}; break;\n'
      out += '}\n'
    return out

  def antiEdge_arr(self):
    out = self.cons.antiEdge.array() 
    out += self.cons.antiEdge.offset()
    out += self.cons.antiEdge.prev()
    return out

  def enumNum(self):
    out = f'constexpr int ENUMNUM = {self.enum_NUM};\n'
    return out

  def enumOut(self):
    out = f'int w = atomicAdd(gcount, 1);\n'
    out += 'if (w >= ENUMNUM - 1) {\n'
    out += '  atomicExch(source, work);\n'
    out += '}\n'
    out += 'if (w < ENUMNUM) {\n'
    # tt = '%d '
    # eids = [f'eid{i}' for i in range(self.allE - 1)]
    # teids = ', '.join(eids) 
    # out += f'  printf("{tt * self.allE}\\n", {teids}, i_g);\n'
    # out += '}\n'
    out += 'int pp = w * numem;\n'
    for i in range(self.allE - 1):
      out += f'  list_d[pp + {i}] = Euid_d[eid{i}];\n'
    out += f'  list_d[pp + {self.allE - 1}] = Euid_d[i_g];\n'
    out += '}\n'
    return out

  def allocEnum(self):
    return f'gpuErrchk(cudaMalloc(&list_d, sizeof(int) * ENUMNUM * {self.allE}));'

  def filterBeg(self):
    return f'if ({self.cons.filter.cond}) {{'

  def filterEnd(self):
    return f'}}'

    
    

