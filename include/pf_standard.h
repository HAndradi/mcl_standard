#ifndef ParticleFilter_H
#define ParticleFilter_H

#include "particle.h"
#include "motion_model.h"
#include "observation_model.h"

struct PFParams{
    int num_particles;
    float init_pose_x, init_pose_y, init_pose_yaw, init_rot_var, init_trans_var_x, init_trans_var_y;
    float odom_trans_var_per_m, odom_trans_var_per_rad, odom_rot_var_per_rad, odom_rot_var_per_m;
    int num_observations;
    float obs_std_dev, max_obs_nn_dist, max_obs_nn_ang_diff, outlier_weight;
};

class ParticleFilter {
    public:
        ParticleFilter(PointCloudNormal::Ptr map_cloud, PFParams params);
        void initializeParticles(Eigen::Vector3f init_pos, Eigen::Quaternionf init_quat);
        void setNewOdom(uint64_t new_odom_timestamp, Eigen::Vector3f new_odom_pos, Eigen::Quaternionf new_odom_quat);
        int filter(PointCloudNormal::Ptr cloud);
        std::vector<Particle> getParticleSet();
        Eigen::Vector3f getAveragePos();
        Eigen::Quaternionf getAverageQuat();
    private:
        MotionModel *motion_model_;
        ObservationModel *observation_model_;
        int num_particles_;
        float init_rot_var_, init_trans_var_x_, init_trans_var_y_;
        std::vector<Particle> particles_;
        std::default_random_engine generator_;
};

#endif
