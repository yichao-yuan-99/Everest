The system finds motif patterns in `motifs` directory, and find the graphs in `graphs` directory.

# Motifs
The `motifs` directory includes experimented motifs in the paper. 

`M4-A.txt` is M4 plus an anti edge (t:3). The anti edge is attached to edge 2.

# Graphs
The `wiki-talk` graph can be downloaded from https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz
The `stackoverflow` graph can be downloaded from https://snap.stanford.edu/data/sx-stackoverflow.txt.gz 
The `reddit-reply` graph can be downloaded from https://www.cs.cornell.edu/~arb/data/temporal-reddit-reply/
The `ethereum` graph can be downloaded from https://zenodo.org/record/4543269#.ZFW64XbMIQ_. 
The preprocessing program for `ethereum` can be downloaded from https://github.com/dkondor/patest_new 
Or, run the `eth_download.sh` and `eth_preprocess.sh` scripts directly in the above repository.
It first downloads the graph, then preprocesses the graph.

The acceptable format for the system is an edge list, each row contains `<SRC> <DST> <TIMESTAMP>`
All comments, like `#...`, should be removed.

These graphs should be placed in `graphs` directory.
An example layout is shown below
```
inputs
├── graphs
│   ├── wiki-talk-temporal.txt
│   ├── sx-stackoverflow.txt
│   ├── temporal-reddit-reply.txt
│   └── ethereum.txt
├── motifs
│   ├── M10.txt
│   ├── M11.txt
│   ├── M12.txt
│   ├── M13.txt
│   ├── M1.txt
│   ├── M2.txt
│   ├── M3.txt
│   ├── M4-A.txt
│   ├── M4.txt
│   ├── M5.txt
│   ├── M6.txt
│   ├── M7.txt
│   ├── M8.txt
│   └── M9.txt
└── README.md
```

# Labels
Because most of the public temporal graphs are non-attributed, we use random numbers as the labels of vertices and edges.

Assume `$PROJ_ROOT` is the root directory of this repo.
After you build the project, generate the random numbers as following
```
cd $PROJ_ROOT/build
./src/generateRand 1000000000 <pathToFile>
```
This generates a file that contains 1 billion random numbers, whose values range from 0 to 1 billion.

Then, change `$PROJ_ROOT/system/tmine/locations.py:randFilePath():46:47`.
Replace that path to your generated random numbers
```
def randFilePath():
  return <pathToFile>
```