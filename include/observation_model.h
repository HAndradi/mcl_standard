#ifndef ObservationModel_H
#define ObservationModel_H

#include "particle.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>

typedef pcl::PointNormal PointNormal;
typedef pcl::PointCloud<PointNormal> PointCloudNormal;
typedef pcl::octree::OctreePointCloudSearch<PointNormal> OctreeNormal;

struct Correspondence {
    PointNormal scan_pt;
    PointNormal scan_pt_transformed;
    PointNormal map_pt;
};

class ObservationModel {
    public:
        ObservationModel(PointCloudNormal::Ptr map_cloud, int num_observations, float observation_std_dev, float max_nn_dist, float max_nn_normal_ang_diff);
        void setInputCloud(PointCloudNormal::Ptr cloud);
        float getLogLikelihood(Particle &particle);
    private:
        int num_observations_;
        float observation_std_dev_;
        float max_nn_sqr_dist_;
        float min_cos_nn_normal_angle_diff_;
        OctreeNormal map_octree_;
        PointCloudNormal::Ptr input_cloud_;
        PointCloudNormal::Ptr map_cloud_;
        void initializeMapClouds(PointCloudNormal::Ptr map_cloud);
};

#endif
