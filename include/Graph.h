#ifndef _GRAPH_
#define _GRAPH_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <unordered_map>
#include <iostream>

// (source, destination, timestamp)
struct TemporalEdge {
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & u;
    ar & v;
    ar & t;
  }
  int u, v, t;
};

struct Edge {
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & u;
    ar & v;
  }
  int u, v;
};

struct CSR
{
    // constructor for in/out matrix, CIDX set to empty
    CSR(const std::vector<std::vector<int>> &ioedges) {
        for (auto &r : ioedges) {
            RIDX.push_back(V.size());
            for (auto &c: r) {
                V.push_back(c);
            }
        }
        RIDX.push_back(V.size());
    }
    CSR() = default;
    int *varray() { return V.data(); }
    unsigned int vsize() { return V.size(); }
    int *carray() { return CIDX.data(); } // actually not used
    unsigned int csize() { return CIDX.size(); }
    int *rarray() { return RIDX.data(); }
    unsigned int rsize() { return RIDX.size(); }
    void content(std::ostream &out) {
        out << "V:";
        for (auto i: V) out << " " << i;
        out << std::endl << "CIDX:"; 
        for (auto i: CIDX) out << " " << i;
        out << std::endl << "RIDX:"; 
        for (auto i: RIDX) out << " " << i;
        out << std::endl; 
    }

    void saveImportant(std::ostream &ofs) {
      ofs.write((const char *) V.data(), V.size() * sizeof(int));
      ofs.write((const char *) RIDX.data(), RIDX.size() * sizeof(int));
    }

    void loadImportant(std::istream &ifs, int lenV, int lenR) {
      V.resize(lenV);
      RIDX.resize(lenR);
      ifs.read((char *) V.data(), lenV * sizeof(int));
      ifs.read((char *) RIDX.data(), lenR * sizeof(int));
    }

    private:
        std::vector<int> V, CIDX, RIDX;
};

class TemporalGraph {
    friend std::ostream &operator<<(std::ostream &o, const TemporalGraph &g);
    std::string _name;
    std::vector<TemporalEdge> _edges;
    std::vector<int> _u, _v, _t;
    std::vector<std::vector<int>> _in, _out; // for a node, who are its in/out edge idx
    CSR _incsr, _outcsr; // csr version for above two 
    std::set<std::pair<int, int>> _sedges; // static edges
    // mappings 
    std::vector<int> _idx2n;
    std::map<int, int> _n2idx; // map the input node to an index
    std::vector<int> _idx2e;

    // have to insert in time order
    void insert(int s, int d, int ts);

  public:
    void saveImportant(std::ostream &ofs);
    void loadImportant(std::istream &ifs);
    TemporalEdge edge(unsigned idx); 
    std::vector<TemporalEdge> &edges();
    TemporalEdge *edges_ptr();

    void populateCSR();
    CSR &inEdges();
    CSR &outEdges();
    std::vector<int> &inEdges(int ei);
    std::vector<int> &outEdges(int ei);

    const int *uarray() const {return _u.data();}
    const int *varray() const {return _v.data();}
    const int *tarray() const {return _t.data();}

    int mapNode(int idx) const; // map idx to node
    int mapEdge(int idx) const; // map idx to edge

    unsigned num_edges() const;
    unsigned num_nodes() const;
    unsigned num_sedges() const; // static edges
    std::string summary() const;

    std::string name() const;

    // output content to out. if (internal) use internal rep, else map to original
    // l is the number of edge to display (from head)
    void content(std::ostream &out, bool internal = true, int l = -1);
    void ioEdgesContent(std::ostream &out, bool in, bool internal = true, int l = -1);

    // static function to read a graph
    static std::unique_ptr<TemporalGraph> ReadGraph(std::string file, bool useCache = false);
};

std::ostream &operator<<(std::ostream &o, const TemporalGraph &g);

#endif // !_GRAPH_