#include "Graph.h"

#include <fstream>
#include <sstream>

#include <iostream>
#include <algorithm>

TemporalEdge TemporalGraph::edge(unsigned idx) {
    return _edges[idx];
}

std::vector<TemporalEdge> &TemporalGraph::edges() {
    return _edges;
}

TemporalEdge *TemporalGraph::edges_ptr() {
    return _edges.data();
}

unsigned TemporalGraph::num_edges() const {
    return _edges.size();
}

unsigned TemporalGraph::num_nodes() const {
    return _idx2n.size();
}

unsigned TemporalGraph::num_sedges() const {
    return _sedges.size();
}

int TemporalGraph::mapNode(int idx) const {
    return _idx2n[idx];
}

int TemporalGraph::mapEdge(int idx) const {
    return _idx2e[idx];
}

void TemporalGraph::populateCSR() {
  if (_incsr.rsize() != num_nodes() + 1) {
      _incsr = CSR(_in);
  }
  if (_outcsr.rsize() != num_nodes() + 1) {
      _outcsr = CSR(_out);
  }
}

CSR &TemporalGraph::inEdges() {
    if (_incsr.rsize() != num_nodes() + 1) {
        _incsr = CSR(_in);
    }
    return _incsr;
}

std::vector<int> &TemporalGraph::inEdges(int ei) {
    return _in[ei];
}

CSR &TemporalGraph::outEdges() {
    if (_outcsr.rsize() != num_nodes() + 1) {
        _outcsr = CSR(_out);
    }
    return _outcsr;
}

std::vector<int> &TemporalGraph::outEdges(int ei) {
    return _out[ei];
}

void TemporalGraph::insert(int u, int v, int t) {
    // map the node to internal index
    auto mapnode = [this](int &n) {
        if (_n2idx.find(n) == _n2idx.end()) {
            int i = _idx2n.size();
            _n2idx[n] = i; // record the bijection
            _idx2n.push_back(n);
            _in.push_back({}); // reserve position for idx
            _out.push_back({});
            n = i;
        } else {
            n = _n2idx[n];
        }
    };
    mapnode(u);
    mapnode(v);

    _in[v].push_back(_edges.size());
    _out[u].push_back(_edges.size());
    _edges.push_back({u, v, t});
    _u.push_back(u);
    _v.push_back(v);
    _t.push_back(t);
    _sedges.insert({u, v});
}

std::string TemporalGraph::summary() const {
    std::string r;
    r += "total nodes: ";
    r += std::to_string(num_nodes());
    r += "; total temporal edges: ";
    r += std::to_string(num_edges());
    r += "; total static edges: ";
    r += std::to_string(num_sedges());
    return r;
}

std::string TemporalGraph::name() const {
  return _name;
}

void TemporalGraph::content(std::ostream &out, bool internal, int l) {
    // map edges back
    std::vector<TemporalEdge> eRordered(num_edges(), {0, 0, 0});
    for (unsigned int i = 0; i < num_edges(); i++) {
        eRordered[mapEdge(i)] = _edges[i]; 
    }

    // for each edge, map and output
    l = l == -1 ? num_edges() : l;
    for (int i = 0; i < l; i++) {
        auto &e = internal ? _edges[i] : eRordered[i];
        auto u = internal ? e.u : mapNode(e.u);
        auto v = internal ? e.v : mapNode(e.v);
        out << u << " " << v << " " << e.t << std::endl;
    }
}

void TemporalGraph::ioEdgesContent(std::ostream &out, bool in, bool internal, int l) {
    auto &edgeM = in ? _in : _out;
    l = l == -1 ? num_nodes() : l;
    // loop in original's nodes sorting order
    for (auto &p : _n2idx) {
        int n = internal ? p.second : p.first;
        out << n;
        for (auto e : edgeM[p.second]) {
            int ei = internal ? e : mapEdge(e);
            out << " " << ei; 
        }
        out << std::endl;
        if (--l == 0) break;
    }
}

void TemporalGraph::saveImportant(std::ostream &ofs) {
  populateCSR();
  int numEdges = num_edges();
  int numNodes = num_nodes();
  int lenVin = _incsr.vsize();
  int lenCin = _incsr.csize();
  int lenRin = _incsr.rsize();
  int lenVout = _outcsr.vsize();
  int lenCout = _outcsr.csize();
  int lenRout = _outcsr.rsize();
  std::cout << "#" << numNodes << " " << numEdges << std::endl;
  std::cout << "#" << lenVin << " " << lenCin << " " << lenRin << std::endl;
  std::cout << "#" << lenVout << " " << lenCout << " " << lenRout << std::endl;
  ofs.write( (const char *) &numEdges, sizeof(int));
  ofs.write( (const char *) &numNodes, sizeof(int));
  ofs.write( (const char *) &lenVin, sizeof(int));
  ofs.write( (const char *) &lenCin, sizeof(int));
  ofs.write( (const char *) &lenRin, sizeof(int));
  ofs.write( (const char *) &lenVout, sizeof(int));
  ofs.write( (const char *) &lenCout, sizeof(int));
  ofs.write( (const char *) &lenRout, sizeof(int));
  ofs.write( (const char *) _edges.data(), sizeof(TemporalEdge) * numEdges);
  ofs.write( (const char *) _idx2n.data(), sizeof(int) * numNodes);
  _incsr.saveImportant(ofs);
  _outcsr.saveImportant(ofs);
}

void TemporalGraph::loadImportant(std::istream &ifs) {
  int numEdges, numNodes;
  int lenVin, lenCin, lenRin, lenVout, lenCout, lenRout;
  ifs.read((char *) &numEdges, sizeof(int));
  ifs.read((char *) &numNodes, sizeof(int));
  ifs.read((char *) &lenVin, sizeof(int));
  ifs.read((char *) &lenCin, sizeof(int));
  ifs.read((char *) &lenRin, sizeof(int));
  ifs.read((char *) &lenVout, sizeof(int));
  ifs.read((char *) &lenCout, sizeof(int));
  ifs.read((char *) &lenRout, sizeof(int));
  std::cout << "#" << numNodes << " " << numEdges << std::endl;
  std::cout << "#" << lenVin << " " << lenCin << " " << lenRin << std::endl;
  std::cout << "#" << lenVout << " " << lenCout << " " << lenRout << std::endl;
  _edges.resize(numEdges);
  _idx2n.resize(numNodes);
  ifs.read( (char *) _edges.data(), sizeof(TemporalEdge) * numEdges);
  ifs.read( (char *) _idx2n.data(), sizeof(int) * numNodes);
  
  _incsr.loadImportant(ifs, lenVin, lenRin);
  _outcsr.loadImportant(ifs, lenVout, lenRout);
  // std::cout << _incsr.vsize() << " " << _incsr.rsize() << " " << _incsr.rarray()[_incsr.rsize() - 1] << std::endl;
}


std::unique_ptr<TemporalGraph> TemporalGraph::ReadGraph(std::string file, bool useCache) {
    auto fileG = file.substr(file.find_last_of("/") + 1);
    auto filePrefix = file.substr(0, file.find_last_of("/"));
    auto name = fileG.substr(0, fileG.find_first_of("."));
    fileG = fileG.substr(0, fileG.find_first_of(".")) + ".cache";
    fileG = filePrefix + "/" + fileG;
    std::ifstream cache(fileG, std::ifstream::in);
    bool hasCache = cache.good();

    if (hasCache && useCache) {
      auto tg = new TemporalGraph;
      tg->_name = name;
      tg->loadImportant(cache);
      std::cout << "#load from cache" << std::endl;
      return std::unique_ptr<TemporalGraph>(tg);
    } else {
      std::ifstream ifile(file);
      if (!ifile) throw std::runtime_error("cannot open file: " + file);
      std::string line;
      struct Tedge {
          int u, v, t, i;
      };
      auto ecmp = [](const Tedge &a, const Tedge &b) -> bool {
          return a.t < b.t;
      };

      // read and sort the input graph
      std::vector<Tedge> readtmp;
      int i = 0;
      while (std::getline(ifile, line)) {
          int u, v, t;
          std::stringstream sl(line);
          sl >> u >> v >> t;
          readtmp.push_back({u, v, t, i++});
          if (i % 1000000 == 0) std::cout << "#Read: " << i << std::endl;
      } 
      std::stable_sort(readtmp.begin(), readtmp.end(), ecmp); 
      // the sorting algorithm influence the order of edges that have same timestamp
      // resulting in about 1% deviation. In theory, only matches that are strictly
      // orderred should be considered. However, current algorithms & implementation
      // often accept matching with = ordering (like BT and SNAP). The BT algorithm 
      // does a stable sort on the input graph based on time. Here we also use stable
      // sort to keep consistency.

      // insert the edges in sorted order, track the mapping between edges
      auto tg = new TemporalGraph;
      tg->_name = name;
      int j = 0;
      for (auto &te : readtmp) {
          tg->_idx2e.push_back(te.i);
          tg->insert(te.u, te.v, te.t);
          j++;
          if (j % 1000000 == 0) std::cout << "#Inserted: " << j << std::endl;
      }
      if (useCache) {
        std::ofstream newcache(fileG);
        std::cout << "#Create Cache" << std::endl;
        tg->saveImportant(newcache);
      }
      return std::unique_ptr<TemporalGraph>(tg);
    }
}

std::ostream &operator<<(std::ostream &o, const TemporalGraph &g) {
  o << g.summary();
  return o;
}