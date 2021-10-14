#ifndef OptimalParticleFilter_H
#define OptimalParticleFilter_H

#include "particle.h"
#include "motion_model.h"
#include "observation_model.h"

class OptimalParticleFilter {
    public:
        OptimalParticleFilter(int num_particles, PointCloudNormal::Ptr map_cloud);
        void initializeParticles(Eigen::Vector3f init_pos, Eigen::Quaternionf init_quat);
        void setNewOdom(uint64_t new_odom_timestamp, Eigen::Vector3f new_odom_pos, Eigen::Quaternionf new_odom_quat);
        void filter(PointCloudNormal::Ptr cloud);
        std::vector<Particle> getParticleSet();
        Eigen::Vector3f getAveragePos();
        Eigen::Quaternionf getAverageQuat();
    private:
        MotionModel *motion_model_;
        ObservationModel *observation_model_;
        int num_particles_;
        std::vector<Particle> particles_;
        std::default_random_engine generator_;
};

#endif
