
#include "fps_subsampling.h"


void fps_subsampling(vector<PointXYZ>& original_points,
                      vector<int>& subsampled_inds,
                      int new_n,
                      float min_d,
                      int verbose)
{

	// Initialize variables
	// ********************

	// New max number of points
	if (new_n < 0)
		new_n = original_points.size();
	
	// square radius
	float min_d2 = 0;
	if (min_d > 0)
		min_d2 = min_d * min_d;

	// Number of points in the cloud
	size_t N = original_points.size();

	// Variables
	vector<bool> remaining_mask(original_points.size(), true);
	int best_j = 0;
	float best_d2 = 0;

	// Fisrt select a random point
	subsampled_inds.reserve(new_n);
	subsampled_inds.push_back(0);
	PointXYZ p0 = original_points[0];
	remaining_mask[0] = false;


	// Init remaining distances
	// ************************

	vector<float> all_d2(original_points.size());
	int j = 0;
	for (auto& p : original_points)
	{
		// Get distance to first point
		float d2 = (p0 - p).sq_norm();
		all_d2[j] = d2;

		// Update max dist
		if (d2 > best_d2)
		{
			best_j = j;
			best_d2 = d2;
		}
		j++;
	}


	// Start FPS
	// *********

	for (int i = 1; i < new_n; i++)
	{
		// Stop if we reach minimum distance
		if (best_d2 < min_d2)
			break;

		// Add the fursthest point
		subsampled_inds.push_back(best_j);
		p0 = original_points[best_j];

		// Remove from remaining
		remaining_mask[best_j] = false;

		// Loop on all remaining points to get the distances
		j = 0;
		best_d2 = 0;
		for (auto& p : original_points)
		{
			if (remaining_mask[j])
			{
				// Update all the minimum distances
				float d2 = (p0 - p).sq_norm();
				if (d2 < all_d2[j])
					all_d2[j] = d2;
				else
					d2 = all_d2[j];

				// Update max of the min distances
				if (d2 > best_d2)
				{
					best_j = j;
					best_d2 = d2;
				}
			}
			j++;
		}
	}

	return;
}



// void fps_subsampling_2(vector<PointXYZ>& original_points,
//                       vector<int>& subsampled_inds,
//                       int new_n,
//                       float sampleDl,
//                       int verbose)
// {

// 	// Initialize variables
// 	// ******************
	
// 	// square radius
// 	float min_d2 = sampleDl * sampleDl;

// 	// Number of points in the cloud
// 	size_t N = original_points.size();

// 	// Variables
// 	vector<bool> remaining_mask(original_points.size(), true); 


// 	// Variables
// 	vector<size_t> remaining_i(original_points.size()); 
// 	std::iota(remaining_i.begin(), remaining_i.end(), 0);
// 	int best_j = 0;
// 	float best_d2 = 0;

// 	// Fisrt select a random point
// 	subsampled_inds.reserve(new_n);
// 	subsampled_inds.push_back(0);
// 	PointXYZ p0 = original_points[0];
// 	remaining_i.erase(remaining_i.begin());


// 	// Init remaining distances
// 	// ************************

// 	vector<float> all_d2(original_points.size());
// 	int j = 0;
// 	for (auto& p : original_points)
// 	{
// 		// Get distance to first point
// 		float d2 = (p0 - p).sq_norm();
// 		all_d2[j] = d2;

// 		// Update max dist
// 		if (d2 > best_d2)
// 		{
// 			best_j = j;
// 			best_d2 = d2;
// 		}
// 		j++;
// 	}


// 	// Start FPS
// 	// *********

// 	for (int i = 1; i < new_n; i++)
// 	{
// 		// Add the fursthest point
// 		subsampled_inds.push_back(best_j);

// 		// Remove from remaining
// 		remaining_i.erase(remaining_i.begin() + best_j);
// 		remaining_d2.erase(remaining_i.begin() + best_j);



// 		// Loop on all reamining points to get the distances
// 		j = 0;
// 		for (auto& p : remaining_i)
// 		{
// 			float d2 = (p0 - p).sq_norm();
// 			if (d2 < remaining_d2[j])
// 				remaining_d2[j] = d2;
// 			j++;
// 		}










// 	}

























// 	// Limits of the cloud
// 	PointXYZ minCorner = min_point(original_points);
// 	PointXYZ maxCorner = max_point(original_points);
// 	PointXYZ originCorner = floor(minCorner * (1/sampleDl)) * sampleDl;

// 	// Dimensions of the grid
// 	size_t sampleNX = (size_t)floor((maxCorner.x - originCorner.x) / sampleDl) + 1;
// 	size_t sampleNY = (size_t)floor((maxCorner.y - originCorner.y) / sampleDl) + 1;
// 	//size_t sampleNZ = (size_t)floor((maxCorner.z - originCorner.z) / sampleDl) + 1;



// 	// Create the sampled map
// 	// **********************

// 	// Verbose parameters
// 	int i = 0;
// 	int nDisp = N / 100;

// 	// Initialize variables
// 	size_t iX, iY, iZ, mapIdx;
// 	unordered_map<size_t, SampledData> data;

// 	for (auto& p : original_points)
// 	{
// 		// Position of point in sample map
// 		iX = (size_t)floor((p.x - originCorner.x) / sampleDl);
// 		iY = (size_t)floor((p.y - originCorner.y) / sampleDl);
// 		iZ = (size_t)floor((p.z - originCorner.z) / sampleDl);
// 		mapIdx = iX + sampleNX*iY + sampleNX*sampleNY*iZ;

// 		// If not already created, create key
// 		if (data.count(mapIdx) < 1)
// 			data.emplace(mapIdx, SampledData(fdim, ldim));

// 		// Fill the sample map
// 		if (use_feature && use_classes)
// 			data[mapIdx].update_all(p, original_features.begin() + i * fdim, original_classes.begin() + i * ldim);
// 		else if (use_feature)
// 			data[mapIdx].update_features(p, original_features.begin() + i * fdim);
// 		else if (use_classes)
// 			data[mapIdx].update_classes(p, original_classes.begin() + i * ldim);
// 		else
// 			data[mapIdx].update_points(p);

// 		// Display
// 		i++;
// 		if (verbose > 1 && i%nDisp == 0)
// 			std::cout << "\rSampled Map : " << std::setw(3) << i / nDisp << "%";

// 	}

// 	// Divide for barycentre and transfer to a vector
// 	subsampled_points.reserve(data.size());
// 	if (use_feature)
// 		subsampled_features.reserve(data.size() * fdim);
// 	if (use_classes)
// 		subsampled_classes.reserve(data.size() * ldim);
// 	for (auto& v : data)
// 	{
// 		subsampled_points.push_back(v.second.point * (1.0 / v.second.count));
// 		if (use_feature)
// 		{
// 		    float count = (float)v.second.count;
// 		    transform(v.second.features.begin(),
//                       v.second.features.end(),
//                       v.second.features.begin(),
//                       [count](float f) { return f / count;});
//             subsampled_features.insert(subsampled_features.end(),v.second.features.begin(),v.second.features.end());
// 		}
// 		if (use_classes)
// 		{
// 		    for (int i = 0; i < ldim; i++)
// 		        subsampled_classes.push_back(max_element(v.second.labels[i].begin(), v.second.labels[i].end(),
// 		        [](const pair<int, int>&a, const pair<int, int>&b){return a.second < b.second;})->first);
// 		}
// 	}

// 	return;
// }

