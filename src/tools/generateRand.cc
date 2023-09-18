/*
 * Generate random numbers to a file
 */

#include <iostream>
#include "data.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    fmt::print("usage: ./generateRand <numOfRand> <pathToFile>\n");
  } else {
    size_t numOfRand = std::atoll(argv[1]);
    std::filesystem::path pathToFile(argv[2]); 
    corelib::util::generateRandToFile(numOfRand, pathToFile);
  }
}