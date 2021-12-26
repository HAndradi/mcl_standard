#include "pf_standard.h"
#include <random>
#include <Eigen/Eigenvalues>
#include <limits>
#include <algorithm>
#include <stdlib.h>
#include <chrono>

using Gaussian = std::normal_distribution<float>;

Eigen::Vector3f sampleZeroMean3DGaussian(Eigen::Matrix3f cov, std::default_random_engine &generator) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver;
    eigen_solver.computeDirect(cov, Eigen::ComputeEigenvectors);
    Eigen::Vector3f eigen_vals = eigen_solver.eigenvalues();
    Eigen::Matrix3f eigen_vecs = eigen_solver.eigenvectors();

    Gaussian distribution_pc1(0, sqrt(std::max(0.0f, eigen_vals(0))));
    Gaussian distribution_pc2(0, sqrt(std::max(0.0f, eigen_vals(1))));
    Gaussian distribution_pc3(0, sqrt(std::max(0.0f, eigen_vals(2))));
    
    Eigen::Vector3f principle_components_sample(distribution_pc1(generator), distribution_pc2(generator), distribution_pc3(generator));
    Eigen::Vector3f sample = eigen_vecs * principle_components_sample;
    return sample; 
}

ParticleFilter::ParticleFilter(PointCloudNormal::Ptr map_cloud, PFParams params) {
    num_particles_ = params.num_particles;
    init_trans_var_x_ = params.init_trans_var_x;
    init_trans_var_y_ = params.init_trans_var_y;
    init_rot_var_ = params.init_rot_var;
    motion_model_ = new MotionModel(params.odom_trans_var_per_m, params.odom_trans_var_per_rad
                                    , params.odom_rot_var_per_rad, params.odom_rot_var_per_m); 
    observation_model_ = new ObservationModel(map_cloud, params.num_observations, params.obs_std_dev, params.max_obs_nn_dist
                                              , params.max_obs_nn_ang_diff, params.outlier_weight);
    initializeParticles(Eigen::Vector3f(params.init_pose_x, params.init_pose_y, 0)
                        , Eigen::Quaternionf(Eigen::AngleAxisf(params.init_pose_yaw, Eigen::Vector3f::UnitZ())));
    generator_.seed(std::time(0));
}

void ParticleFilter::initializeParticles(Eigen::Vector3f init_pos, Eigen::Quaternionf init_quat) {
    Eigen::Matrix3f init_pose_cov = Eigen::Vector3f(init_trans_var_x_, init_trans_var_y_, init_rot_var_).asDiagonal();
    particles_.clear();
    for (size_t i = 0; i < num_particles_; i++) {
        Eigen::Vector3f init_pose_noise = sampleZeroMean3DGaussian(init_pose_cov, generator_);
        Eigen::Vector3f init_pos_noise(init_pose_noise(0), init_pose_noise(1), 0);
        Eigen::Quaternionf init_quat_noise(Eigen::AngleAxisf(init_pose_noise(2), Eigen::Vector3f::UnitZ()));
         
        Eigen::Vector3f particle_pos = init_pos + init_quat * init_pos_noise;
        Eigen::Quaternionf particle_quat = init_quat * init_quat_noise;
        particles_.push_back(Particle(particle_pos, particle_quat, 1.0/num_particles_)); 
    }
}

//void ParticleFilter::initializeParticles(Eigen::Vector3f init_pos, Eigen::Quaternionf init_quat) {
//    Eigen::Matrix3f init_pose_cov = Eigen::Vector3f(init_trans_var_x_, init_trans_var_y_, init_rot_var_).asDiagonal();
//    particles_.clear();
//    for (size_t i = 0; i < num_particles_/3; i++) {
//        Eigen::Vector3f init_pose_noise = sampleZeroMean3DGaussian(init_pose_cov, generator_);
//        Eigen::Vector3f init_pos_noise(init_pose_noise(0), init_pose_noise(1), 0);
//        Eigen::Quaternionf init_quat_noise(Eigen::AngleAxisf(init_pose_noise(2), Eigen::Vector3f::UnitZ()));
//         
//        Eigen::Vector3f particle_pos = init_pos + Eigen::Vector3f(0,0,0) + init_quat * init_pos_noise;
//        Eigen::Quaternionf particle_quat = init_quat * init_quat_noise;
//        particles_.push_back(Particle(particle_pos, particle_quat, 1.0/num_particles_)); 
//    }
//    for (size_t i = 0; i < num_particles_/3; i++) {
//        Eigen::Vector3f init_pose_noise = sampleZeroMean3DGaussian(init_pose_cov, generator_);
//        Eigen::Vector3f init_pos_noise(init_pose_noise(0), init_pose_noise(1), 0);
//        Eigen::Quaternionf init_quat_noise(Eigen::AngleAxisf(init_pose_noise(2), Eigen::Vector3f::UnitZ()));
//         
//        Eigen::Vector3f particle_pos = init_pos + Eigen::Vector3f(-11,0,0) + init_quat * init_pos_noise;
//        Eigen::Quaternionf particle_quat = init_quat * init_quat_noise;
//        particles_.push_back(Particle(particle_pos, particle_quat, 1.0/num_particles_)); 
//    }
//    int num_particles_last_room = num_particles_ - particles_.size();
//    for (size_t i = 0; i < num_particles_last_room; i++) {
//        Eigen::Vector3f init_pose_noise = sampleZeroMean3DGaussian(init_pose_cov, generator_);
//        Eigen::Vector3f init_pos_noise(init_pose_noise(0), init_pose_noise(1), 0);
//        Eigen::Quaternionf init_quat_noise(Eigen::AngleAxisf(init_pose_noise(2), Eigen::Vector3f::UnitZ()));
//         
//        Eigen::Vector3f particle_pos = init_pos + Eigen::Vector3f(11,0,0) + init_quat * init_pos_noise;
//        Eigen::Quaternionf particle_quat = init_quat * init_quat_noise;
//        particles_.push_back(Particle(particle_pos, particle_quat, 1.0/num_particles_)); 
//    }
//}

void ParticleFilter::setNewOdom(uint64_t new_odom_timestamp, Eigen::Vector3f new_odom_pos, Eigen::Quaternionf new_odom_quat) {
    motion_model_->setNewOdom(new_odom_timestamp, new_odom_pos, new_odom_quat);
}

void ParticleFilter::filter(PointCloudNormal::Ptr cloud) {
    Eigen::Vector3f motion_trans = motion_model_->getTranslation();
    Eigen::Quaternionf motion_rot = motion_model_->getRotation();
    Eigen::Matrix3f motion_noise_cov = motion_model_->getAccumulatedMotionNoiseCovariance();
    motion_model_->resetAccumulatedMotion();

    observation_model_->setInputCloud(cloud);
    std::vector<float> particle_log_likelihoods(num_particles_);
    for (size_t i=0; i < num_particles_; i++){
        particles_[i].updateState(motion_trans, motion_rot);
        Eigen::Vector3f motion_noise_sample = sampleZeroMean3DGaussian(motion_noise_cov, generator_);
        Eigen::Vector3f motion_noise_sample_trans = Eigen::Vector3f(motion_noise_sample(0), motion_noise_sample(1), 0);
        Eigen::Quaternionf motion_noise_sample_rot = Eigen::Quaternionf(1, 0, 0, 0) * Eigen::AngleAxisf(motion_noise_sample(2), Eigen::Vector3f::UnitZ());
        particles_[i].updateState(motion_noise_sample_trans, motion_noise_sample_rot);

        particle_log_likelihoods[i] = observation_model_->getLogLikelihood(particles_[i]);
    }

    //////////////////////////// SET WEIGHTS ////////////////////////////////////////

    float max_particle_log_likelihood = *std::max_element(particle_log_likelihoods.begin(), particle_log_likelihoods.end());
    float weight_sum = 0.0;
    for (size_t i=0; i < particle_log_likelihoods.size(); i++) {
        //particles_[i].setWeight(exp((particle_log_likelihoods[i] - max_particle_log_likelihood)/10000));
        particles_[i].setWeight(exp(particle_log_likelihoods[i] - max_particle_log_likelihood));
        weight_sum += particles_[i].getWeight();
    }
    std::cout << "weight SUM : " << weight_sum << std::endl;

    for (size_t i = 0; i < particles_.size(); i++) {
        particles_[i].setWeight(particles_[i].getWeight()/weight_sum); 
    }

    //////////////////////////// RESAMPLE ////////////////////////////////////////
    std::vector<float> cummulative_weights{particles_[0].getWeight()};
    for (size_t i=1; i < particles_.size()-1; i++) {
        cummulative_weights.push_back(cummulative_weights.back() + particles_[i].getWeight());
        //std::cout << cummulative_weights[i] << " , ";
    }
    //std::cout << 1.0 << std::endl;
    cummulative_weights.push_back(1.0);

    std::vector<Particle> new_particles;
    float random_prob = (float)rand()/((float)RAND_MAX+1)/num_particles_;
    size_t last_j = 0;
    for (size_t i = 0; i < num_particles_; i++) {
        for(size_t j = last_j; j < cummulative_weights.size(); j++) {
            if (cummulative_weights[j]>random_prob) {
                new_particles.push_back(particles_[j]);
                last_j = j;
                break;
            }
        }
        random_prob += 1.0/num_particles_;
    }  
    particles_ = new_particles; 
}

std::vector<Particle> ParticleFilter::getParticleSet() {
    return particles_;
}

Eigen::Vector3f ParticleFilter::getAveragePos() {
    Eigen::Vector3f pos_sum(0, 0, 0);
    for (size_t i = 0; i < particles_.size(); i++) {
        pos_sum += particles_[i].getPos();
    }
    return pos_sum / particles_.size();
}

Eigen::Quaternionf ParticleFilter::getAverageQuat() {
    Eigen::Quaternionf quat_sum(0, 0, 0, 0);
    for (size_t i = 0; i < particles_.size(); i++) {
        Eigen::Quaternionf particle_quat = particles_[i].getQuat();
        float sign = (quat_sum.norm() == 0 || quat_sum.dot(particle_quat) > 0) ? 1 : -1;
        quat_sum.x() += sign * particle_quat.x();
        quat_sum.y() += sign * particle_quat.y();
        quat_sum.z() += sign * particle_quat.z();
        quat_sum.w() += sign * particle_quat.w();
    }
    return quat_sum.normalized();
}
