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
ros::Publisher robot_pose_pub, particles_pub, map_frame_downsampled_cloud_pub;

std::string map_file, map_frame, odom_frame, robot_frame;
PFParams pf_params;

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
        tf::StampedTransform odom_basefootprint_tf, basefootprint_lidar_tf;
        tf_listener_ptr->lookupTransform(odom_frame, robot_frame, ros::Time(0), odom_basefootprint_tf);
        tf_listener_ptr->lookupTransform(robot_frame, cloud_msg->header.frame_id, ros::Time(0), basefootprint_lidar_tf);
        PointCloudNormal::Ptr basefootprint_frame_cloud (new PointCloudNormal ());
        pcl_ros::transformPointCloudWithNormals (*cloud, *basefootprint_frame_cloud, basefootprint_lidar_tf);

        pf_standard.filter(basefootprint_frame_cloud);
        std::vector<Particle> particles = pf_standard.getParticleSet();
   
        geometry_msgs::PoseArray particle_array;
        particle_array.header.frame_id = map_frame;
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
        pose_msg.header.frame_id = map_frame;
        pose_msg.pose.pose = getGeometryMsgPose(avg_particle_pos, avg_particle_quat);
        robot_pose_pub.publish(pose_msg);

        tf::Transform map_basefootprint_tf;
        map_basefootprint_tf.setOrigin(tf::Vector3(avg_particle_pos(0), avg_particle_pos(1), 0.0));
        tf::Quaternion map_basefootprint_quat;
        quaternionMsgToTF(pose_msg.pose.pose.orientation, map_basefootprint_quat);
        map_basefootprint_tf.setRotation(map_basefootprint_quat);
        tf::Transform map_odom_transform = map_basefootprint_tf * odom_basefootprint_tf.inverse();
        tf_broadcaster_ptr->sendTransform(tf::StampedTransform(map_odom_transform, cloud_msg->header.stamp, map_frame, odom_frame));

        Eigen::Affine3f map_basefootprint_transform = Eigen::Affine3f::Identity() * avg_particle_quat;
        map_basefootprint_transform.translation() =  avg_particle_pos;
        PointCloudNormal::Ptr map_frame_downsampled_cloud (new PointCloudNormal ());
        pcl::transformPointCloudWithNormals (*basefootprint_frame_cloud, *map_frame_downsampled_cloud, map_basefootprint_transform);
        sensor_msgs::PointCloud2 output_cloud;
        pcl::toROSMsg(*map_frame_downsampled_cloud, output_cloud);
        output_cloud.header.frame_id = map_frame;
        output_cloud.header.stamp = ros::Time::now();
        map_frame_downsampled_cloud_pub.publish(output_cloud);
    } catch (tf::TransformException ex) {
        ROS_ERROR("%s",ex.what());
    }
}

void getRosParams(ros::NodeHandle &nh) {
    nh.param("map_file", map_file, std::string("/home/hk/Files/HBRS/Thesis/catkin_ws_thesis/src/mcl_standard/maps/lego_loam_normals_map.pcd"));
    nh.param("map_frame", map_frame, std::string("map"));
    nh.param("odom_frame", odom_frame, std::string("odom"));
    nh.param("robot_frame", robot_frame, std::string("base_footprint"));

    nh.param("num_particles", pf_params.num_particles, 3);
    nh.param("init_pose_x", pf_params.init_pose_x, 0.0f);
    nh.param("init_pose_y", pf_params.init_pose_y, 0.0f);
    nh.param("init_pose_yaw", pf_params.init_pose_yaw, 0.0f);
    nh.param("init_trans_var", pf_params.init_trans_var, 0.1f);
    nh.param("init_rot_var", pf_params.init_rot_var, 0.1f);
   
    nh.param("odom_trans_var_per_m", pf_params.odom_trans_var_per_m, 0.1f);         // 10m -> std.dev.=1m
    nh.param("odom_trans_var_per_rad", pf_params.odom_trans_var_per_rad, 0.002f);   // 360deg -> std.dev.=0.1m
    nh.param("odom_rot_var_per_rad", pf_params.odom_rot_var_per_rad, 0.1f);         // 360deg -> std.dev.=45deg
    nh.param("odom_rot_var_per_m", pf_params.odom_rot_var_per_m, 0.06f);            // 10m -> std.dev.=45deg
 
    nh.param("num_observations", pf_params.num_observations, 300);
    nh.param("obs_std_dev", pf_params.obs_std_dev, 0.02f);
    nh.param("max_obs_nn_dist", pf_params.max_obs_nn_dist, 0.3f);
    nh.param("max_obs_nn_ang_diff", pf_params.max_obs_nn_ang_diff, 0.2f);
    nh.param("outlier_weight", pf_params.outlier_weight, 0.0f);
}

int main (int argc, char** argv) {
    ros::init (argc, argv, "mcl_standard_node");
    ros::NodeHandle nh("~");

    tf_listener_ptr = new tf::TransformListener();
    tf_broadcaster_ptr = new tf::TransformBroadcaster();

    getRosParams(nh);
    PointCloudNormal::Ptr map_cloud (new PointCloudNormal ());
    if (pcl::io::loadPCDFile<PointNormal> (map_file, *map_cloud) == -1) {
        PCL_ERROR ("Couldn't read map file: \n");
        return -1;
    }
    
    ParticleFilter pf_standard(map_cloud, pf_params);
 
    ros::Subscriber init_pose_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, boost::bind(initPoseCb, _1, boost::ref(pf_standard)));
    ros::Subscriber odom_sub = nh.subscribe<nav_msgs::Odometry>("/odom", 10, boost::bind(odomCb, _1, boost::ref(pf_standard)));    
    ros::Subscriber cloud_sub = nh.subscribe<sensor_msgs::PointCloud2> ("/lidar_point_normals", 1, boost::bind(cloudCb, _1, boost::ref(pf_standard)));
    robot_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped> ("/robot_pose", 1, true); 
    particles_pub = nh.advertise<geometry_msgs::PoseArray> ("particles", 1, true);
    map_frame_downsampled_cloud_pub = nh.advertise<sensor_msgs::PointCloud2> ("/map_frame_downsampled_cloud", 1);  
 
    ros::spin ();
}

