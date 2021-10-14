#include <cmath>
#include <Eigen/Eigenvalues>
#include <pcl/common/distances.h>
#include <pcl/common/centroid.h>
#include <normal_estimation/normal_estimation.h>

NormalEstimation::NormalEstimation(int nn_height, int min_nn_width, int max_nn_width, float max_row_nn_dist,
                                   float min_nn_major_axes_var, float max_nn_minor_axis_var, float max_pt_to_normal_ang){
    nn_half_height_ = std::floor(nn_height/2);
    min_nn_half_width_ = std::floor((min_nn_width-1)/2);
    max_nn_half_width_ = std::floor((max_nn_width-1)/2);
    max_row_nn_dist_sqrd_ = pow(max_row_nn_dist,2);
    max_nn_minor_axis_var_ = max_nn_minor_axis_var;
    min_nn_major_axes_var_ = min_nn_major_axes_var;
    max_pt_to_normal_ang_ = max_pt_to_normal_ang;
    max_valid_nn_normal_dist_ = 2*sqrt(max_nn_minor_axis_var); 
}

PointNormal NormalEstimation::createNaNXYZNormalPoint (){
    PointNormal pt_normal;
    pt_normal.x = std::numeric_limits<double>::quiet_NaN();
    pt_normal.y = std::numeric_limits<double>::quiet_NaN();
    pt_normal.z = std::numeric_limits<double>::quiet_NaN();
    return pt_normal;
}

bool NormalEstimation::getNearestNeighborCloud(const PointCloud::Ptr cloud_in, int row, int col, PointCloud::Ptr cloud_nn){
    if (row - nn_half_height_ < 0 || row + nn_half_height_ >= cloud_in->height){ return false; }
    if (col - max_nn_half_width_ < 0 || col + max_nn_half_width_ >= cloud_in->width){ return false; }
    int cloud_nn_idx = 0;
    cloud_nn->points.resize((2*nn_half_height_+1)*(2*max_nn_half_width_ + 1));
    for( int row_shift = -nn_half_height_; row_shift <= nn_half_height_; row_shift++){
        PointT pt_row_mid = cloud_in->at(col, row+row_shift);
        if (!std::isfinite(pt_row_mid.x)){ return false; }
        int num_pts_row = 0;
        for( int col_shift = -max_nn_half_width_; col_shift <= max_nn_half_width_; col_shift++){
            PointT pt_nn = cloud_in->at(col+col_shift, row+row_shift);
            
            if (std::abs(col_shift) <= min_nn_half_width_ || pcl::squaredEuclideanDistance(pt_row_mid, pt_nn) < max_row_nn_dist_sqrd_){
                cloud_nn->points[cloud_nn_idx] = pt_nn;
                cloud_nn_idx++;
                num_pts_row++;
            }
        }
        if (num_pts_row < min_nn_half_width_){ return false; }
    }
    cloud_nn->points.resize(cloud_nn_idx);
    return true;   
}

void NormalEstimation::performPCA (PointCloud::Ptr cloud_in, Eigen::Matrix3f& eigen_vecs, Eigen::Vector3f& eigen_vals){
   Eigen::Vector4f cloud_centroid;
   pcl::compute3DCentroid(*cloud_in, cloud_centroid);
   Eigen::Matrix3f cloud_cov;
   computeCovarianceMatrixNormalized(*cloud_in, cloud_centroid, cloud_cov);
   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver;
   eigen_solver.computeDirect(cloud_cov.cast<double>(), Eigen::ComputeEigenvectors);
   eigen_vals = eigen_solver.eigenvalues().cast<float>();
   eigen_vecs = eigen_solver.eigenvectors().cast<float>();
}

int NormalEstimation::getNumNearestNeighborOutliers(PointT pt, PointCloud::Ptr cloud_nn, Eigen::Matrix3f eigen_vecs){
    int num_outliers = 0;
    for (size_t i = 0; i < cloud_nn->points.size(); i++){
        PointT pt_nn = cloud_nn->points[i];
        float normal_dist = fabs((pt.x-pt_nn.x)*eigen_vecs.col(0)[0] + (pt.y-pt_nn.y)*eigen_vecs.col(0)[1] + (pt.z-pt_nn.z)*eigen_vecs.col(0)[2]);
        if (normal_dist > max_valid_nn_normal_dist_){ num_outliers++; } 
    }
    return num_outliers;
}

void NormalEstimation::computeSurfaceNormals (const PointCloud::Ptr cloud_in, PointCloudNormal::Ptr cloud_normals){
    float row_downsample_factor = 2;
    float col_downsample_factor = 50; // 3; //
    cloud_normals->is_dense = false;
    cloud_normals->width = cloud_in->width / col_downsample_factor;
    cloud_normals->height = cloud_in->height / row_downsample_factor;
    cloud_normals->points.resize(cloud_normals->width * cloud_normals->height);
    Eigen::Matrix3f eigen_vecs;
    Eigen::Vector3f eigen_vals;    
    PointCloud::Ptr cloud_nn (new PointCloud ());
    for( size_t row_new = 0, row = 0; row_new < cloud_normals->height; row_new++, row += row_downsample_factor){
        for( size_t col_new = 0, col = 0; col_new < cloud_normals->width; col_new++, col += col_downsample_factor){
            PointT pt = cloud_in->at(col,row);
            PointNormal pt_normal = createNaNXYZNormalPoint();
            if (std::isfinite(pt.x)) {
                if (getNearestNeighborCloud(cloud_in, row, col, cloud_nn)){
                    performPCA(cloud_nn, eigen_vecs, eigen_vals);
                    //int num_nn_outliers = getNumNearestNeighborOutliers(pt, cloud_nn, eigen_vecs); 
                    //if (eigen_vals[0] < max_nn_minor_axis_var_ && eigen_vals[1] > min_nn_major_axes_var_ && num_nn_outliers == 0){
                    if (eigen_vals[0] < max_nn_minor_axis_var_ && eigen_vals[1] > min_nn_major_axes_var_){
                        float pt_dot_normal = pt.x*eigen_vecs.col(0)[0] + pt.y*eigen_vecs.col(0)[1] + pt.z*eigen_vecs.col(0)[2];
                        float pt_to_normal_ang = acos(fabs(pt_dot_normal)/sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z));
                        if (pt_to_normal_ang < max_pt_to_normal_ang_){
                            if (pt_dot_normal > 0){
                                eigen_vecs.col(0) *= -1;
                            }
                            pt_normal.x = pt.x;
                            pt_normal.y = pt.y;
                            pt_normal.z = pt.z;
                            pt_normal.normal_x = eigen_vecs.col(0)[0];
                            pt_normal.normal_y = eigen_vecs.col(0)[1];
                            pt_normal.normal_z = eigen_vecs.col(0)[2];
                            pt_normal.curvature = eigen_vals[0];
                        }
                    } 
                }
            }
            cloud_normals->at(col_new,row_new) = pt_normal;
        }
    }
}

