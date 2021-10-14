#ifndef Particle_H
#define Particle_H

#include <Eigen/Core>
#include <Eigen/Geometry>

class Particle {
    public:
        Particle(Eigen::Vector3f pos, Eigen::Quaternionf quat, float weight);
        void setWeight(float weight);
        void setPose(Eigen::Vector3f pos, Eigen::Quaternionf quat); 
        void updateState(Eigen::Vector3f translation, Eigen::Quaternionf rotation);
        float getWeight();
        Eigen::Vector3f getPos();
        Eigen::Quaternionf getQuat();
    private:
        Eigen::Vector3f pos_;
        Eigen::Quaternionf quat_;
        float weight_;
};

#endif
