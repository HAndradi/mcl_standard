#ifndef NormalEstimation_H_
#define NormalEstimation_H_

#include <pcl/point_types.h>
#include "pcl/point_cloud.h"
#include <Eigen/Core>

typedef pcl::PointXYZ PointT;
typedef pcl::PointNormal PointNormal;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointNormal> PointCloudNormal;

class NormalEstimation{
    public:
        NormalEstimation(int nn_height, int min_nn_width, int max_nn_width, float max_row_nn_dist,
                         float min_nn_major_axes_var, float max_nn_minor_axis_var, float max_pt_to_normal_ang);
        void computeSurfaceNormals (const PointCloud::Ptr cloud_in, PointCloudNormal::Ptr cloud_normals);
    private:
        int nn_half_height_;
        int min_nn_half_width_;
        int max_nn_half_width_;
        float max_row_nn_dist_sqrd_;
        float max_nn_minor_axis_var_;
        float min_nn_major_axes_var_;
        float max_pt_to_normal_ang_;
        float max_valid_nn_normal_dist_;
        
        PointNormal createNaNXYZNormalPoint ();
        bool getNearestNeighborCloud(const PointCloud::Ptr cloud_in, int row, int col, PointCloud::Ptr cloud_nn);
        void performPCA (PointCloud::Ptr cloud_in, Eigen::Matrix3f& eigen_vecs, Eigen::Vector3f& eigen_vals);
        int getNumNearestNeighborOutliers(PointT pt, PointCloud::Ptr cloud_nn, Eigen::Matrix3f eigen_vecs);
};

#endif
