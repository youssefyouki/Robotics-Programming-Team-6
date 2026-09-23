#include <mujoco_ros2/mujoco_ros.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    if (argc < 2)
    {
        throw std::invalid_argument("[ERROR] Invalid number of arguments. Usage: ur10e_node path/to/scene.xml");
    }

    try
    {
        auto ur10eNode = std::make_shared<MuJoCoROS>(argv[1], "ur10e_node");
        rclcpp::spin(ur10eNode);
        rclcpp::shutdown();
        return 0;
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(rclcpp::get_logger("ur10e_node"), "%s", exception.what());
        rclcpp::shutdown();
        return 1;
    }
}
