#include "pf_standard.h"
#include "ros/ros.h"
#include "pcl_ros/point_cloud.h"
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "boost/bind.hpp"
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>

tf::TransformListener *tf_listener_ptr;
tf::TransformBroadcaster* tf_broadcaster_ptr;
ros::Publisher pose_pub, particles_pub;

geometry_msgs::Pose getGeometryMsgPose(Eigen::Vector3f pos, Eigen::Quaternionf quat) {
    geometry_msgs::Pose pose;
    pose.position.x = pos(0);
    pose.position.y = pos(1);
    pose.position.z = pos(2);
    pose.orientation.x = quat.x();
    pose.orientation.y = quat.y();
    pose.orientation.z = quat.z();
    pose.orientation.w = quat.w();
    return pose;
}

void odomCb(const nav_msgs::Odometry::ConstPtr& odom_msg, ParticleFilter &pf_standard) {
    geometry_msgs::Pose pose = odom_msg->pose.pose;
    Eigen::Vector3f new_odom_pos(pose.position.x, pose.position.y, pose.position.z);    
    Eigen::Quaternionf new_odom_quat(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);   
    pf_standard.setNewOdom(odom_msg->header.stamp.toNSec(), new_odom_pos, new_odom_quat);
}

void initPoseCb(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose_stamped_msg, ParticleFilter &pf_standard) {
    geometry_msgs::Pose pose = pose_stamped_msg->pose.pose;
    Eigen::Vector3f init_pos = Eigen::Vector3f(pose.position.x, pose.position.y, pose.position.z);    
    Eigen::Quaternionf init_quat = Eigen::Quaternionf(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);   
    pf_standard.initializeParticles(init_pos, init_quat); 
}

void cloudCb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg, ParticleFilter &pf_standard){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
    PointCloudNormal::Ptr cloud (new PointCloudNormal ());
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);

    try{
        tf::StampedTransform odom_basefootprint_transform, basefootprint_lidar_transform;
        tf_listener_ptr->lookupTransform("/odom", "/base_footprint", ros::Time(0), odom_basefootprint_transform);
        tf_listener_ptr->lookupTransform("/base_footprint", cloud_msg->header.frame_id, ros::Time(0), basefootprint_lidar_transform);
        PointCloudNormal::Ptr transformed_cloud (new PointCloudNormal ());
        pcl_ros::transformPointCloud (*cloud, *transformed_cloud, basefootprint_lidar_transform);

        pf_standard.filter(transformed_cloud);
        std::vector<Particle> particles = pf_standard.getParticleSet();
   
        geometry_msgs::PoseArray particle_array;
        particle_array.header.frame_id = "map";
        particle_array.header.stamp = ros::Time::now();
        for(size_t i = 0; i < particles.size(); i++)
        {
            Eigen::Vector3f particle_pos = particles[i].getPos();
            Eigen::Quaternionf particle_quat = particles[i].getQuat();
            particle_array.poses.push_back(getGeometryMsgPose(particle_pos, particle_quat));
        }
        particles_pub.publish(particle_array);

        Eigen::Vector3f avg_particle_pos = pf_standard.getAveragePos();
        Eigen::Quaternionf avg_particle_quat = pf_standard.getAverageQuat();
 
        geometry_msgs::PoseWithCovarianceStamped pose_msg;
        pose_msg.header.stamp = ros::Time::now();
        pose_msg.header.frame_id = "map";
        pose_msg.pose.pose = getGeometryMsgPose(avg_particle_pos, avg_particle_quat);
        pose_pub.publish(pose_msg);

        tf::Transform map_basefootprint_transform;
        map_basefootprint_transform.setOrigin(tf::Vector3(avg_particle_pos(0), avg_particle_pos(1), 0.0));
        tf::Quaternion map_basefootprint_quat;
        quaternionMsgToTF(pose_msg.pose.pose.orientation, map_basefootprint_quat);
        map_basefootprint_transform.setRotation(map_basefootprint_quat);
        tf::Transform map_odom_transform = map_basefootprint_transform * odom_basefootprint_transform.inverse();
        tf_broadcaster_ptr->sendTransform(tf::StampedTransform(map_odom_transform, cloud_msg->header.stamp, "/map", "/odom"));
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s",ex.what());
    }
}

int main (int argc, char** argv) {
    ros::init (argc, argv, "mcl_standard_node");
    ros::NodeHandle nh("~");

    tf_listener_ptr = new tf::TransformListener();
    tf_broadcaster_ptr = new tf::TransformBroadcaster();

    std::string map_file; 
    nh.param("map_file", map_file, std::string("/home/hk/Files/HBRS/Thesis/catkin_ws_thesis/src/mcl_standard/maps/lego_loam_normals_map.pcd"));
    PointCloudNormal::Ptr map_cloud (new PointCloudNormal ());
    if (pcl::io::loadPCDFile<PointNormal> (map_file, *map_cloud) == -1) {
        PCL_ERROR ("Couldn't read map file: \n");
        return -1;
    }
    
    int num_particles; 
    nh.param("num_particles", num_particles, 3);
    ParticleFilter pf_standard(num_particles, map_cloud);
 
    ros::Subscriber init_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, boost::bind(initPoseCb, _1, boost::ref(pf_standard)));
    ros::Subscriber odom_sub = nh.subscribe<nav_msgs::Odometry>("/odom", 10, boost::bind(odomCb, _1, boost::ref(pf_standard)));    
    ros::Subscriber cloud_sub = nh.subscribe<sensor_msgs::PointCloud2> ("/lidar_cloud_normals", 1, boost::bind(cloudCb, _1, boost::ref(pf_standard)));
    pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped> ("/robot_pose", 1, true); 
    particles_pub = nh.advertise<geometry_msgs::PoseArray> ("particles", 1, true);
   
    ros::spin ();
}

