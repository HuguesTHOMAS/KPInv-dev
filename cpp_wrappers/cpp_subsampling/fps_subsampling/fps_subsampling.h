

#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

void fps_subsampling(vector<PointXYZ>& original_points,
                      vector<int>& subsampled_inds,
                      int new_n,
                      float min_d,
                      int verbose);
