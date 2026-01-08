#ifndef SCNN_COMMON_H_
#define SCNN_COMMON_H_

#include <iomanip>
#include <deque>
#include <vector>
#include <tuple>
#include <string>
#include <set>
#include <map>
#include <random>   
#include <assert.h>
#include <tuple>
#include <fstream>



namespace Scnn {
    using namespace std;

// Hardware Configuration
struct Config{
    static const int NUM_PE = 64;
    static const int NUM_MULTIPLIERS = 16;
    static const int OUTPUT_PORT = 16;
};

}

#endif