#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

tf::TransformBroadcaster* brPtr;

void 
odom_cb (nav_msgs::Odometry msg)
{
  tf::Transform transform;
  transform.setOrigin( tf::Vector3(msg.pose.pose.position.x, msg.pose.pose.position.y, 0.0) );
  tf::Quaternion quat;
  quaternionMsgToTF(msg.pose.pose.orientation, quat);
  transform.setRotation(quat);
  brPtr->sendTransform(tf::StampedTransform(transform, msg.header.stamp, "/odom", "/base_footprint"));
}

int
main (int argc, char** argv)
{
  ros::init (argc, argv, "odom_broadcaster_node");
  ros::NodeHandle nh;
   
  tf::TransformBroadcaster br;
  brPtr = &br;
 
  ros::Subscriber sub = nh.subscribe<nav_msgs::Odometry> ("/odom", 10, odom_cb);

  ros::spin ();
}
