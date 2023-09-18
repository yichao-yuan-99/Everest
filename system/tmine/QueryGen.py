from . import locations

import yaml

def basicWithNodeTHCons(pattern, graph, delta, TH, warpCentric, warpBalancing, tailExpansion):
  query = {}
  query['pattern'] = pattern
  query['graph'] = graph
  cons = {}
  cons['node'] = f'f > {TH};'
  cons['delta'] = delta
  query['constraints'] = cons
  options = {}
  options['gpu'] = 1
  options['warpCentric'] = warpCentric
  options['warpBalancing'] = warpBalancing
  options['tailExpansion'] = tailExpansion
  query['options'] = options

  with open(locations.genQueryPath(), 'w') as file:
    yaml.dump(query, file, default_flow_style=False)

def basic(pattern, graph, delta, warpCentric, warpBalancing, tailExpansion, gpu):
  query = {}
  query['pattern'] = pattern
  query['graph'] = graph
  cons = {}
  cons['delta'] = delta
  query['constraints'] = cons
  options = {}
  options['gpu'] = gpu 
  options['warpCentric'] = warpCentric
  options['warpBalancing'] = warpBalancing
  options['tailExpansion'] = tailExpansion
  query['options'] = options

  with open(locations.genQueryPath(), 'w') as file:
    yaml.dump(query, file, default_flow_style=False)

def basicDst(pattern, graph, delta, warpCentric, warpBalancing, tailExpansion, gpu, dst):
  query = {}
  query['pattern'] = pattern
  query['graph'] = graph
  cons = {}
  cons['delta'] = delta
  query['constraints'] = cons
  options = {}
  options['gpu'] = gpu 
  options['warpCentric'] = warpCentric
  options['warpBalancing'] = warpBalancing
  options['tailExpansion'] = tailExpansion
  query['options'] = options

  with open(dst, 'w') as file:
    yaml.dump(query, file, default_flow_style=False)