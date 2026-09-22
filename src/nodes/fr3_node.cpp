#include <mujoco_ros2/mujoco_ros.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    if (argc < 2)
    {
        throw std::invalid_argument("[ERROR] Invalid number of arguments. Usage: fr3_node path/to/scene.xml");
    }

    try
    {
        auto fr3Node = std::make_shared<MuJoCoROS>(argv[1], "fr3_node");
        rclcpp::spin(fr3Node);
        rclcpp::shutdown();
        return 0;
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(rclcpp::get_logger("fr3_node"), "%s", exception.what());
        rclcpp::shutdown();
        return 1;
    }
}
