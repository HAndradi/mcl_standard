#include "observation_model.h"
#include <pcl/common/transforms.h>
#include <Eigen/SVD>
#include <cmath>
#include <algorithm>

ObservationModel::ObservationModel(PointCloudNormal::Ptr map_cloud, int num_observations, float observation_std_dev, float max_nn_dist
                                   , float max_nn_normal_ang_diff, float obs_alpha): map_octree_(0.1) {
    num_observations_ = num_observations;
    observation_std_dev_ = observation_std_dev;
    max_nn_sqr_dist_ = pow(max_nn_dist,2);
    min_cos_nn_normal_angle_diff_ = cos(max_nn_normal_ang_diff);
    initializeMapClouds(map_cloud);
    obs_alpha_ = obs_alpha; 
}

void ObservationModel::setInputCloud(PointCloudNormal::Ptr cloud) {
    input_cloud_.reset(new PointCloudNormal ());
    for (size_t i=0; i < cloud->points.size(); i++) {
        PointNormal pt = cloud->points[i];
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z) && std::isfinite(pt.normal_x) 
            && std::isfinite(pt.normal_y) && std::isfinite(pt.normal_z)) {
            input_cloud_->points.push_back(cloud->points[i]);
        }
    }
    if (input_cloud_->points.size() > num_observations_) {
        std::random_shuffle(input_cloud_->points.begin(), input_cloud_->points.end());
        input_cloud_->points.resize(num_observations_);
    } else {
        std::cout << "Insufficient number of scan points" << std::endl;
    }
}

float ObservationModel::getLogLikelihood(Particle &particle) {
    Eigen::Affine3f transform = Eigen::Affine3f::Identity() * particle.getQuat();
    transform.translation() =  particle.getPos();
    PointCloudNormal::Ptr transformed_cloud (new PointCloudNormal ());
    pcl::transformPointCloudWithNormals (*input_cloud_, *transformed_cloud, transform);

    int num_observations = transformed_cloud->points.size();
    float nn_sqr_dist_sum = 0;
    float log_likelihood_sum = 0;
    for (size_t i = 0; i < transformed_cloud->points.size(); i++){
        PointNormal& scan_pt = transformed_cloud->points[i];
        int nn_pt_id;
        float nn_sqr_dist;
        map_octree_.approxNearestSearch(scan_pt, nn_pt_id, nn_sqr_dist);
        float nn_sqr_dist_capped = (nn_sqr_dist > max_nn_sqr_dist_) ? max_nn_sqr_dist_ : nn_sqr_dist;
        ///////////////////////////////////////// P(rand) inclusion ///////////////////////////////////////////////////////////////
        double p_hit = 1/(observation_std_dev_*sqrt(2*M_PI)) * exp(- 0.5 * nn_sqr_dist_capped / pow(observation_std_dev_,2));
        double p_rand = 1/sqrt(max_nn_sqr_dist_);
        float log_likelihood_pt = log(obs_alpha_ * p_rand + (1-obs_alpha_) * p_hit);
        log_likelihood_sum += log_likelihood_pt;
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        nn_sqr_dist_sum += nn_sqr_dist_capped;
    }
    float log_likelihood = num_observations * log(1/(observation_std_dev_*sqrt(2*M_PI))) - 0.5 * nn_sqr_dist_sum / pow(observation_std_dev_,2);
    return log_likelihood_sum;
    //return log_likelihood;
}

void ObservationModel::initializeMapClouds(PointCloudNormal::Ptr map_cloud) {
    map_cloud_.reset(new PointCloudNormal ());
    *map_cloud_ = *map_cloud;
    std::cout << "map_cloud_: " << map_cloud_->points.size() << std::endl;
    map_octree_.setInputCloud(map_cloud_);
    map_octree_.addPointsFromInputCloud();
}

