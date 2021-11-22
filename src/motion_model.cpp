#include "motion_model.h"

MotionModel::MotionModel(float trans_var_per_m, float trans_var_per_rad, float rot_var_per_rad, float rot_var_per_m) {
    last_odom_timestamp_ = 0; 
    motion_trans_ = Eigen::Vector3f(0,0,0);
    motion_rot_ = Eigen::Quaternionf(1,0,0,0);
    robot_frame_motion_noise_cov_ = Eigen::Vector3f(1e-6, 1e-6, 1e-6).asDiagonal();
    trans_var_per_m_ = trans_var_per_m; 
    trans_var_per_rad_ = trans_var_per_rad;
    rot_var_per_rad_ = rot_var_per_rad;
    rot_var_per_m_ = rot_var_per_m;
}

void MotionModel::setNewOdom(uint64_t new_odom_timestamp, Eigen::Vector3f new_odom_pos, Eigen::Quaternionf new_odom_quat) {
    if (last_odom_timestamp_ > 0) {
        Eigen::Quaternionf last_odom_quat_inv = last_odom_quat_.inverse();
        Eigen::Quaternionf last_odom_relative_rot = last_odom_quat_inv * new_odom_quat.normalized();
        Eigen::Vector3f last_odom_relative_trans = last_odom_quat_inv * (new_odom_pos - last_odom_pos_);

        Eigen::Matrix3f state_transition_mat = Eigen::Matrix3f::Identity();
        state_transition_mat(0,2) = -last_odom_relative_trans(1);
        state_transition_mat(1,2) = last_odom_relative_trans(0);
        
        float last_odom_relative_yaw = 2*atan(last_odom_relative_rot.z()/last_odom_relative_rot.w());
        float last_odom_relative_x_var = trans_var_per_m_ * (fabs(last_odom_relative_trans(0)) + 0.1 * fabs(last_odom_relative_trans(1)))
                                         + trans_var_per_rad_ * fabs(last_odom_relative_yaw);
        float last_odom_relative_y_var = trans_var_per_m_ * (fabs(last_odom_relative_trans(1)) + 0.1 * fabs(last_odom_relative_trans(0)))
                                         + trans_var_per_rad_ * fabs(last_odom_relative_yaw);
        float last_odom_relative_yaw_var = rot_var_per_rad_ * fabs(last_odom_relative_yaw) + rot_var_per_m_ * last_odom_relative_trans.norm(); 
        Eigen::Vector3f last_odom_relative_motion_var(last_odom_relative_x_var, last_odom_relative_y_var, last_odom_relative_yaw_var);
        Eigen::Matrix3f last_odom_relative_motion_cov = last_odom_relative_motion_var.asDiagonal();
        Eigen::Matrix3f last_odom_frame_motion_noise_cov = state_transition_mat * robot_frame_motion_noise_cov_ * state_transition_mat.transpose()
                                                           + last_odom_relative_motion_cov;

        motion_trans_ += motion_rot_ * last_odom_relative_trans;
        motion_rot_ *= last_odom_relative_rot; 
        motion_rot_.normalize(); 
        robot_frame_motion_noise_cov_ = last_odom_relative_rot.inverse() * last_odom_frame_motion_noise_cov * last_odom_relative_rot; 
    }
    last_odom_timestamp_ = new_odom_timestamp;
    last_odom_pos_ = new_odom_pos;
    last_odom_quat_ = new_odom_quat.normalized();
}

void MotionModel::resetAccumulatedMotion(){
    motion_trans_ = Eigen::Vector3f(0, 0, 0);
    motion_rot_ = Eigen::Quaternionf(1, 0, 0, 0);
    robot_frame_motion_noise_cov_ = Eigen::Vector3f(1e-6, 1e-6, 1e-6).asDiagonal();
}

Eigen::Vector3f MotionModel::getTranslation() {
    return motion_trans_;
}

Eigen::Quaternionf MotionModel::getRotation() {
    return motion_rot_.normalized();
}

Eigen::Matrix3f MotionModel::getAccumulatedMotionNoiseCovariance() {
    return robot_frame_motion_noise_cov_;
}
