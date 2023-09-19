Copyright ©2023; The Regents of the University of Michigan
# Overview
Everest is a system that mines temporal motifs using GPU.

If you find this repo useful, please cite the following paper:
```
@inproceedings{yuan2023everest,
  author    = {Yichao Yuan and
               Haojie Ye and
               Sanketh Vedula and
               Wynn Kaza and
               Nishil Talati 
               },
  title     = {{Everest: GPU-Accelerated System For Mining Temporal Motifs}},
  booktitle = {{50th International Conference on Very Large Databases (VLDB 2024)}},
  pages     = {},
  publisher = {ACM},
  year      = {2024}
}
```

# Dependencies
Everest requires the following software and libraries be installed in the system:
- C++17 compatible compiler (go to Troubleshooting Section for Compilation Problem)
- CUDA (11.4 is tested, but a lower version may also be acceptable)
- Boost (1.71 is tested, but a lower version may also be acceptable)
- Python (>= 3.8.1, go to Troubleshooting Section for Python Problems)
- pyyaml (Python package. Used to parse YAML files)
- CMake (>= 3.18)

Besides, the following libraries are also used. They are downloaded and linked to Everest at compile time.
- range-v3
- fmt
- pybind11

To get profiling results, it also requires `nsight compute` installed.

## We also provide a conda environment file
Use `enivronment.yml` if you are facing difficult to make a compatible environemnt

Install the above using conda env create -f environment.yml

​
Post installation:

```
conda env config vars set CPATH="${CONDA_PREFIX}/include:${CPATH}"
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"​
```

Then reactivate the environment

# Build
First change line 5-7 in `CMakeLists.txt`, such that they are pointing to your compilers.
If there a problem during compilation, see Troubleshooting section.

Then use the following commands to build the system
```
mkdir build && cd build
cmake ..
make -j
```

The code assumes the GPUs support `SM86`.
If they do not, change all the places with `SM86`, `sm_86` and any other related things to the supported architecture number.
- line 28, 29, 82 in `CMakeLists.txt`
- line 24 in `system/tmin/utils.py`

Look up the number here: https://developer.nvidia.com/cuda-gpus#compute

# Prepare Inputs
Please follow the README in `inputs` directory 

# Program Structure

Everest uses C++/CUDA for data loading, preprocessing, shared library management and performing execution.
These components are exposed to Python via pybind11.
The overall system uses Python to manage the query processing flow, generate optimized libraries based on queries, and perform other miscellaneous jobs.

## Repository Structure
The structure of the repository is shown below:

```
├── CMakeLists.txt
├── include         # header files
├── inputs          # where the systems looks for inputs
│   ├── graphs
│   ├── motifs
│   └── README.md
├── plugins         # legacy code, including the baseline GPU implementation
├── README.md
├── src             # C++/CUDA source code
│   ├── CMakeLists.txt
│   ├── core        # core library
│   ├── mods        # pybind files
│   └── tools       # some helper tools
└── system          # the overall system. cd to this directory to run everything
    ├── batch-query.py
    ├── filter.py
    ├── generateQueries.py
    ├── legacy.py
    ├── process-batch-query.sh
    ├── process-single-query.sh
    ├── run-legacy.sh          # all the python files and bash scripts above
    ├── single-query.py        # provide simple user interfaces
    ├── queries                # lots of example queries
    ├── scripts                # some helper scripts 
    ├── templates              # the template files used by the code generator
    ├── tmine                  # the Python source code 
    └── tools                  # some helper tools
```


# Usages 

## A Baisc Query Example
The following is a simple query example (`system/queries/examples/simpleQuery.yaml`)
```
pattern: M3.txt
graph: wiki-talk-temporal.txt
constraints:
  delta: 86400
options: 
  gpu: 1
```
The query is in YAML format.
This query describes a task that mines motif `M3` from `wiki-talk-temporal`, with a time window of `86400`.
This query will use one GPU in the system.
By default, this query counts the number of matches.
`M3` and `wiki-talk-temporal` should be prepared in `inputs` directory during the previous step.

To process this query, do the following
```
cd system
./process-single-query.sh queries/examples/simpleQuery.yaml
```
The system will start to process this query, after a while (this includes the overhead of compiling generated code, loading graphs etc.), something like the following will be printed
```
# <Number of Matches>, <pure processing time in micro sec>, <~ time in second>
904054, 22608, 0.022635252913460135 
```

## Process a Batch of Queries
When multiple queries use the same graph and same number of GPUs, these queries can be processed together to amortize the overhead of loading graphs.
A set of quries that meet this property can be found in `system/queries/basics/all/sx-stackoverflow/`.
These queries mine motif `M1-13` in `sx-stackoverflow` with a 1 day time window, using 1 GPU.

To process them in a batch, do the following
```
cd system
./process-batch-query.sh sx-stackoverflow 1 system/queries/basics/all/sx-stackoverflow/
```
Its output will be something like the following
```
# process M1.yaml
229122198, 100571, 0.1006035809405148
# process M10.yaml
...
```

## Enumerate
If `enumerate:<max enumeration>` is specified in the `options` section of the query, the matched motifs will be enumerated.
The examples can be found in `system/queries/enumerate` directory.
The following query
```
pattern: M10.txt
graph: wiki-talk-temporal.txt
constraints:
  delta: 86400
options: 
  gpu: 4
  enumerate: 1000000
```
Enumerates all the `M10` motifs in `wiki-talk`.
The output will be a list of motifs, each is represented by a row of edge index.
```
3561659 2352506 6090185 4049667 2352516 # edge id of edge 1, edge id of edge 2...
524745 3754925 368558 3721982 3754931
524747 3754925 368558 3721982 3754931
524747 3754925 368558 3721982 3754933
...
```
The index here is the 0-indexed position of the edge in the `.txt` edge list.

## Vertex and Edge Labeling 
The requirements for vertices' and edges' labels can be specified via a set of constraints sections.
The snippet below specifies the first motif node's label should be greater than 5e8.
Similar requirements are specified for the second, third, and the last nodes.
```
...
constraints:
  node0: f > 500000000
  node1: f < 500000000
  node2: f > 500000000
  node3: f < 500000000
...
```
An example query is `system/queries/additional-features/ETH-M4-v.yaml`

## Fine-Grained Temporal Constraints
Fined grained temporal constraints can be specified as below in `constraints` section:
```
...
constraints:
  temporal: [900, 1800, 900]
  delta: 3600
...
```
This requires the second edge happens within 900 seconds after the first edge happens.
An example query is `system/queries/additional-features/ETH-M4-t.yaml`

## Anti-edge
Anti edges are specified within `constraints` section and an edge in the motif pattern
```
pattern: M4-A.txt
constraints:
  antiEdge: [False, False, True, False, False]
...
```
This specifies `M4` pattern with an anti-edge attached to the second edge.
The anti-edge attached to a regular edge should be declared immediatly following that regular edge.
The anti-edge here is thus placed in temporal order 3.

Below shows how to specify fine-grained temporal constraints together with anti-edges
```
...
constraints:
  antiEdge: [False, False, True, False, False]
  temporal: [900, 1800, 1800, 900]
  delta: 3600
...
```
The anti-edge's fine-grained temporal constrain is specified in the second element of array `temporal`.

An example query is `system/queries/additional-features/ETH-M4-all.yaml`

## Multi-GPU Support
To use multiple GPUs, set `gpu` section in `options` to the number of available gpus.
For example, the query below will use 4 GPUs.
```
...
options:
  gpu: 4
...
```
An example query is `system/queries/multiGPU/ETH-M7-4.yaml`

## Turn on/off load balancing features
Mechanisms that perform load balancing can be turned off.
The following turns off intra-warp work stealing.
```
...
options:
  tailExpansion: true
  warpBalancing: false
  warpCentric: true
...
```

The following turns off tail warp work redistribution
```
...
options:
  tailExpansion: false
  warpBalancing: true
  warpCentric: true
...
```

Or, turn off all of these. This version only includes candidate edges caching and motif-specific code generation.
```
...
options:
  tailExpansion: false
  warpBalancing: false
  warpCentric: false
...
```

Examples can be found in `system/queries/basics`

The multi-GPU support is valid to use only when all optimizations are enabled, i.e. in the default settings.

## Run Baseline
The baseline code is not compatible with the query interface as it can only do delta-window motif mining.
The baseline can be run with the following commands:
```
cd system
./run-legacy.sh <pathToGraph> <pathToMotif> <delta> <Type>
```
Type can be either (a) Baseline, (b) U0, which only includes caching candidate edges, or (c) U1B, which includes candidate edges caching and motif-specific code (data structure).

For example:
```
cd system
./run-legacy.sh ../inputs/graphs/sx-stackoverflow.txt ../inputs/motifs/M1.txt 86400 Baseline
```

# Reproducing results

## Compare the effect of different optimization combinations
To reproduce the results for `wso`, `two` and `a`, use the queries in `queries/basics/warpStealOnly`, `queries/basics/tailWarpOnly`, and `queries/basics/all`.
To reproduce results for baseline, `c`, and `g`, use the `./run-legacy.sh` script, with type Baseline, U0, and U1B, respectively.
These correspond to data in Figure 10.

## Multi-GPU scaling
The queries in `queries/multiGPU` demonstrates the effect of Multiple GPUs.
Each query corresponds to one data point in Figure 11.

## additional constraints
The queries in `queries/additional-features` demonstrates the effect of additional features.
It can be used to generate results in Table 4.

## Profiling Results
in `system`, use the following to generate profiling results
```
source scripts/prof.sh
```
Each corresponds to one data point in Figure 11.



# Troubleshooting

## During Compilation of `.cu` file, it shows `fatal error: filesystem: No such file or directory`

This is because the name `gcc` is pointing to a version of gcc that does not support C++17.
`nvcc` uses `gcc` as host compiler, so `nvcc` will produce error to compile C++17 code.
### Fix
In `CMakeLists.txt:29`, change `set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -arch=sm_86")` to
`set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -arch=sm_86 -ccbin <your c++17 compiler>")`

In `system/tmin/utils.py:24`, change `command = f'nvcc -O3 --std=c++17 -arch=sm_86 -I../include -Xcompiler -fPIC -shared -o {out} {file}'` to ` ommand = f'nvcc -O3 --std=c++17 -arch=sm_86  -ccbin <your c++17 compiler> -I../include -Xcompiler -fPIC -shared -o {out} {file}'`

## Errors when loading compiled `corelib` into python

This is likely due to inconsistent versions of python for compilation and execution.
Make sure that the python libraries used during compilation (i.e. searched in `LD_LIBRARY_PATH`) is compatible with the one you are using to start the system.

If you cannot fix your environment, you can compile python from source.
Assume that `$HOME/opt/python-3.8.1` is a safe directory to install python, then you can compile python3.8.1 using the following commands
```
wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tar.xz
tar -xvf Python-3.8.1.tar.xz 
cd Python-3.8.1
./configure --prefix=$HOME/opt/python-3.8.1
make -j
make install
```

Then, add this Python to your `.bashrc`
```
export $HOME/opt/python-3.8.1/bin:/usr/local/cuda-11.2/bin/:$PATH
export LD_LIBRARY_PATH=$HOME/opt/python-3.8.1/lib/
```

Then, delete all the things you have built so far and restart the building process again.
