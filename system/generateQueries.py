from tmine import QueryGen, locations
from pathlib import Path

Graphs = ['wiki-talk-temporal', 'sx-stackoverflow', 'temporal-reddit-reply', 'ethereum']

Delta = {}
Delta['wiki-talk-temporal'] = 86400
Delta['sx-stackoverflow'] = 86400
Delta['temporal-reddit-reply'] = 36000
Delta['ethereum'] = 3600

OptLevel = {}
OptLevel['all'] = (True, True, True)
OptLevel['tailWarpOnly'] = (True, False, True)
OptLevel['warpStealOnly'] = (True, True, False)

for opt, vals in OptLevel.items():
  for g in Graphs:
    Path(locations.queryRoot() + f'/basics/{opt}/{g}').mkdir(exist_ok=True)
    for ii in range(0, 13):
      i = ii + 1
      m = f'M{i}.txt'
      q = f'M{i}.yaml'
      dst = locations.queryRoot() + f'/basics/{opt}/{g}/{q}'
      QueryGen.basicDst(m, f'{g}.txt', Delta[g], *vals, 1, dst)

