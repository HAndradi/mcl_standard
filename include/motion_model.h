#ifndef MotionModel_H
#define MotionModel_H

#include <Eigen/Core>
#include <Eigen/Geometry>

class MotionModel {
    public:
        MotionModel(float trans_var_per_m, float trans_var_per_rad, float rot_var_per_rad, float rot_var_per_m);
        void setNewOdom(uint64_t new_odom_timestamp, Eigen::Vector3f new_odom_pos, Eigen::Quaternionf new_odom_quat);
        void resetAccumulatedMotion();
        Eigen::Vector3f getTranslation();
        Eigen::Quaternionf getRotation();
        Eigen::Matrix3f getAccumulatedMotionNoiseCovariance();
    private:
        float trans_var_per_m_, trans_var_per_rad_, rot_var_per_rad_, rot_var_per_m_;
        Eigen::Vector3f motion_trans_;
        Eigen::Quaternionf motion_rot_;
        Eigen::Matrix3f robot_frame_motion_noise_cov_;
        uint64_t last_odom_timestamp_;
        Eigen::Vector3f last_odom_pos_;
        Eigen::Quaternionf last_odom_quat_;
};

#endif
