#include <mujoco_ros2/mujoco_ros.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    if (argc < 2)
    {
        throw std::invalid_argument("[ERROR] Invalid number of arguments. Usage: low_cost_robot_arm_node path/to/scene.xml");
    }

    try
    {
        auto node = std::make_shared<MuJoCoROS>(argv[1], "low_cost_robot_arm_node");
        rclcpp::spin(node);
        rclcpp::shutdown();
        return 0;
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(rclcpp::get_logger("low_cost_robot_arm_node"), "%s", exception.what());
        rclcpp::shutdown();
        return 1;
    }
}
