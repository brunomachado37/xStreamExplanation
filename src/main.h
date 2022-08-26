#include <cassert>
#include "chain.h"
#include <chrono>
#include "docopt.h"
#include "hash.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include "param.h"
#include <random>
#include "streamhash.h"
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include "util.h"
#include <vector>

#include <fstream>

using namespace std;

tuple<vector<vector<float>>,vector<vector<float>>> compute_deltamax(vector<vector<float>>& window, uint c, uint k, mt19937_64& prng);

class xStream{
    private:
        xStream(uint pk, uint pc, uint pd, uint pinit_sample_size);

        uint k; uint c; uint d; bool fixed; bool cosine; int nwindows; uint init_sample_size; uint scoring_batch_size;

        mt19937_64 prng;

        vector<vector<float>> deltamax;
        vector<vector<float>> shift;

        vector<vector<unordered_map<vector<int>,int>>> cmsketches;                                   
        vector<vector<uint>> fs;

        vector<uint64_t> h;
        float density_constant;

        vector<vector<float>> window;
        vector<float> anomalyscores;
        
        uint row_idx;
        uint window_size;

        vector<vector<tuple<uint, uint, float>>> minDensities;

        vector<vector<bool>> feature_projection_map;

    public:
        vector<float> getScores(void){ 
            return anomalyscores; 
        }

        vector<vector<tuple<uint, uint, float>>> getMinDensity(void){ 
            return minDensities; 
        }

        vector<vector<bool>> getFeatureProjectionMap(void){ 
            return feature_projection_map; 
        }

        int fit(vector<string> x);

        static xStream init(uint pk, uint pc, uint pd, uint pinit_sample_size){
            return xStream(pk, pc, pd, pinit_sample_size);
        }
};