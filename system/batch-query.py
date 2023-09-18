from tmine import frontend, locations
import sys, os

# if multiple queries share the same graph and number of gpus,
# they can be run in a batch

graph = sys.argv[1]
gpu = sys.argv[2]
queryDirectory = sys.argv[3]

queries = os.listdir(queryDirectory)
queries.sort()

m = frontend.Mining(f'{locations.inputGraphRoot()}/{graph}', int(gpu))
for q in queries:
  qpath = f'{queryDirectory}{q}'
  print("@@@ [output starts]")
  print(f"# process {q}")
  print("@@@ [output ends]")
  m.run(qpath)