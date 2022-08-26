#include "main.h"

using namespace std;


xStream::xStream(uint pk, uint pc, uint pd, uint pinit_sample_size){

  this->k = pk;
  this->c = pc;
  this->d = pd;
  this->fixed = true;
  this->cosine = false;
  this->nwindows = 1;
  this->init_sample_size = pinit_sample_size;
  this->scoring_batch_size = 100000;

  #ifdef SEED
      mt19937_64 temp(SEED);
  #else
      random_device rd;
      mt19937_64 temp(rd());
  #endif

  this->prng = temp;

  for (uint i = 0; i < c; i++){
    this->deltamax.push_back(vector<float>(k, 0.0));
    this->shift.push_back(vector<float>(k, 0.0));
    this->cmsketches.push_back(vector<unordered_map<vector<int>, int>>(d));
    this->fs.push_back(vector<uint>(d, 0));
  }

  // initialize streamhash functions
  for (uint i = 0; i < k; i++){
    this->h.push_back(0);
  }
  this->density_constant = streamhash_compute_constant(DENSITY, k);

  streamhash_init_seeds(this->h, this->prng);
  chains_init_features(this->fs, this->k, this->prng);

  // current window of projected tuples, part of the model
  for (uint i = 0; i < init_sample_size; i++){
    this->window.push_back(vector<float>(k));
  }

  this->anomalyscores.reserve(1000000);
  this->minDensities.reserve(10000000);

  this->row_idx = 1;
  this->window_size = 0;

}

int xStream::fit(vector<string> x) { 
  vector<float> xp;

  if (row_idx == 1) {
    for (uint i = 0; i < k; i++){
      this->feature_projection_map.push_back(vector<bool>(x.size()));
    }
    xp = streamhash_project(x, h, DENSITY, density_constant, feature_projection_map);
  } 
  else {
    xp = streamhash_project(x, h, DENSITY, density_constant);
  }

  // if the initial sample has not been seen yet, continue
  if (row_idx < init_sample_size) {
    window[window_size] = xp;
    window_size++;
    row_idx++;
    return 0; 
  }

  // check if the initial sample just arrived
  if (row_idx == init_sample_size) {
    window[window_size] = xp;
    window_size++;

    if (!cosine) {
      // compute deltmax/shift from initial sample
      tie(deltamax, shift) = compute_deltamax(window, c, k, prng);
    }

    // add initial sample tuples to chains
    for (auto x : window) {
      if (cosine) {
        chains_add_cosine(x, cmsketches, fs, true);
      } else {
        chains_add(x, deltamax, shift, cmsketches, fs, true);
      }
    }

    // score initial sample tuples
    for (auto x : window) {
      float anomalyscore;
      vector<tuple<uint, uint, float>> minDensity(c);

      if (cosine) {
        anomalyscore = chains_add_cosine(x, cmsketches, fs, false);
      } 
      else {
        anomalyscore = chains_add(x, deltamax, shift, cmsketches, fs, false, minDensity);
      }

      anomalyscores.push_back(anomalyscore);
      minDensities.push_back(minDensity);

    }

    window_size = 0;
    row_idx++;
    return 1;
  }

  // row_idx > init_sample_size
  if (nwindows <= 0) { // non-windowed mode

    float anomalyscore;
    vector<tuple<uint, uint, float>> minDensity(c);

    if (cosine) {
      anomalyscore = chains_add_cosine(xp, cmsketches, fs, true);
    } else {
      anomalyscore = chains_add(xp, deltamax, shift, cmsketches, fs, true, minDensity);
    }
    anomalyscores.push_back(anomalyscore);
    minDensities.push_back(minDensity);

  }

  else if (nwindows > 0) { // windowed mode
    window[window_size] = xp;
    window_size++;

    float anomalyscore;
    vector<tuple<uint, uint, float>> minDensity(c);

    if (cosine) {
      anomalyscore = chains_add_cosine(xp, cmsketches, fs, false);
    } else {
      anomalyscore = chains_add(xp, deltamax, shift, cmsketches, fs, false, minDensity);
    }
    anomalyscores.push_back(anomalyscore);
    minDensities.push_back(minDensity);

    // if the batch limit is reached, construct new chains
    // while different from the paper, this is more cache-efficient
    if (window_size == static_cast<uint>(init_sample_size)) {
      // uncomment this to compute a new deltamax, shift from the new window points
      //tie(deltamax, shift) = compute_deltamax(window, c, k, prng);

      // clear old bincounts
      for (uint chain = 0; chain < c; chain++) {
        for (uint depth = 0; depth < d; depth++) {
          cmsketches[chain][depth].clear();
        }
      }

      // add current window tuples to chains
      for (auto x : window) {
        if (cosine) {
          chains_add_cosine(x, cmsketches, fs, true);
        } else {
          chains_add(x, deltamax, shift, cmsketches, fs, true);
        }
      }

      window_size = 0;
    }
  }

  if ((row_idx > init_sample_size) && (row_idx % scoring_batch_size == 0)) {
    cerr << "\tscoring at tuple: " << row_idx << endl;
    cout << row_idx << "\t";
    for (uint i = 0; i < anomalyscores.size(); i++) {
      float anomalyscore = anomalyscores[i];
      cout << setprecision(12) << anomalyscore << " ";
    }
    cout << endl;
  }

  row_idx++;

  return 2;
}


tuple<vector<vector<float>>,vector<vector<float>>>
compute_deltamax(vector<vector<float>>& window, uint c, uint k, mt19937_64& prng) {

  vector<vector<float>> deltamax(c, vector<float>(k, 0.0));
  vector<vector<float>> shift(c, vector<float>(k, 0.0));

  vector<float> dim_min(k, numeric_limits<float>::max());
  vector<float> dim_max(k, numeric_limits<float>::min());

  for (auto x : window) {
    for (uint j = 0; j < k; j++) {
      if (x[j] > dim_max[j]) { dim_max[j] = x[j]; }
      if (x[j] < dim_min[j]) { dim_min[j] = x[j]; }
    }
  }

  // initialize deltamax to half the projection range, shift ~ U(0, dmax)
  for (uint i = 0; i < c; i++) {
    for (uint j = 0; j < k; j++) {
      deltamax[i][j] = (dim_max[j] - dim_min[j])/2.0;
      if (abs(deltamax[i][j]) <= EPSILON) {
        deltamax[i][j] = 1.0;
      }
      uniform_real_distribution<> dis(0, deltamax[i][j]);
      shift[i][j] = dis(prng);
    }
  }

  return make_tuple(deltamax, shift);
}
