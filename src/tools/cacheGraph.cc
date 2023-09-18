/*
 * make cache for edge list. Add random features to its edges and vertices
 *
 * Provide the path to Graph and the path to a random file. Random file should
 * have enough random numbers
 */

#include "data.h"
#include "Graph.h"
#include <cassert>

int main(int argc, char **argv) {
  if (argc != 3) {
    fmt::print("usage: ./cacheGraph <pathToGraph> <pathToRandom>\n");
    if (argc == 2) {
      auto r = TemporalGraph::ReadGraph(argv[1], true);
    }
  } else {
    std::filesystem::path pathToGraph(argv[1]);
    std::filesystem::path pathToRandom(argv[2]);

    corelib::data::EdgeListLoader eloader(pathToGraph);
    corelib::data::FeatureLoader floader(&eloader, pathToRandom);

    assert(eloader.edgeListLength() == floader.edgeFeaturesLength());
    fmt::print("[cacheGraph] created cache. N: {}, M: {}\n", floader.verticesFeaturesLength(), eloader.edgeListLength());
  }
}