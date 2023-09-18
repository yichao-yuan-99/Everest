import sys
from tmine import Query, locations, frontend, utils

def runQueryOnce(query : str):
  p = frontend.MiningSingle(query)
  p.run()
  
if __name__ == '__main__':
  query = sys.argv[1]
  runQueryOnce(query)
  