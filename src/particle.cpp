#include <particle.h>

Particle::Particle(Eigen::Vector3f pos, Eigen::Quaternionf quat, float weight)
                   : pos_(pos), quat_(quat.normalized()), weight_(weight) {
}

void Particle::setWeight(float weight) {
    weight_ = weight;
}

void Particle::setPose(Eigen::Vector3f pos, Eigen::Quaternionf quat) {
    pos_ = pos; 
    quat_ = quat.normalized(); 
}

void Particle::updateState(Eigen::Vector3f translation, Eigen::Quaternionf rotation) {
    pos_ += quat_ * translation;
    quat_ *= rotation.normalized();
    quat_.normalize();
}

float Particle::getWeight() {
    return weight_;
}

Eigen::Vector3f Particle::getPos() {
    return pos_;
}

Eigen::Quaternionf Particle::getQuat() {
    return quat_;
}
