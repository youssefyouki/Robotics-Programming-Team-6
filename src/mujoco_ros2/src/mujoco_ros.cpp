/**
 * @file   mujoco_ros.cpp
 * @author Jon Woolfrey
 * @email  jonathan.woolfrey@gmail.com
 * @date   April 2025
 * @version 1.1
 * @brief  A class for connecting a MuJoCo simulation with ROS2 communication.
 * 
 * @details This class launches a MuJoCo simulation and provides communication channels in ROS2 for controlling it.
 * 
 * @copyright Copyright (c) 2025 Jon Woolfrey
 * 
 * @license GNU General Public License V3
 * 
 * @see https://mujoco.org/ for more information about MuJoCo
 * @see https://docs.ros.org/en/humble/index.html for ROS 2 documentation
 */
 
#include <mujoco_ros2/mujoco_ros.hpp>
#include <algorithm>

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                         Constructor                                            //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::MuJoCoROS(const std::string &xmlLocation, const std::string &nodeName) : Node(nodeName)
{
    // Declare & get parameters for this node
    std::string jointStateTopicName   = this->declare_parameter<std::string>("joint_state_topic_name", "joint_state");
    std::string jointCommandTopicName = this->declare_parameter<std::string>("joint_command_topic_name", "joint_commands");
    std::string controlMode           = this->declare_parameter<std::string>("control_mode", "TORQUE");
    _simFrequency                     = this->declare_parameter<int>("simulation_frequency", 1000);
    int visualisationFrequency        = this->declare_parameter<int>("visualisation_frequency", 20);
    _torquePassthrough                = this->declare_parameter<bool>("torque_passthrough", false);   // true: raw torque commands, held until the next message

    // Load the robot model     
    char errorMessage[1000] = "Could not load model.";                                              // We need this as an input argument.   
    _model = mj_loadXML(xmlLocation.c_str(), nullptr, errorMessage, 1000);                          // Try to load the model  
    if (not _model) throw std::runtime_error("[ERROR] [MuJoCo NODE] Problem loading model: " + std::string(errorMessage)); 
    _model->opt.timestep = 1/((double)_simFrequency);                                               // Match MuJoCo to node frequency
    if (_torquePassthrough) _model->opt.integrator = mjINT_EULER;                                   // Implicit integrators would treat the servo's velocity gain as extra inertia (+kv*dt); plain Euler does not
    
     // Resize arrays based on the number of joints in the model
    _jointState = mj_makeData(_model);                                                              // Initialize joint state
    _holdTarget.assign(_jointState->qpos, _jointState->qpos + _model->nq);                          // Remember the initial pose (used by the torque passthrough before the first command)
    _jointStateMessage.name.resize(_model->nq);
    _jointStateMessage.position.resize(_model->nq);
    _jointStateMessage.velocity.resize(_model->nq);
    _jointStateMessage.effort.resize(_model->nq);    
    _torqueInput.resize(_model->nq, 0.0);

    // Record joint names
    for (int i = 0; i < _model->nq; ++i) _jointStateMessage.name[i] = mj_id2name(_model, mjOBJ_JOINT, i);
       
    // Set the control mode  
         if (controlMode == "POSITION") _controlMode = POSITION;
    else if (controlMode == "VELOCITY") _controlMode = VELOCITY;
    else if (controlMode == "TORQUE"  ) _controlMode = TORQUE;
    else
    {   
        throw std::invalid_argument("[ERROR] [MuJoCo NODE] Unknown control mode. "
                                    "Argument was '" + controlMode + "', but expected 'POSITION', 'VELOCITY', or 'TORQUE'.");
    }

    // Create timers
    _simTimer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000/_simFrequency)),
                                        std::bind(&MuJoCoROS::update_simulation, this));
                                        
    _visTimer = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000/visualisationFrequency)),
                                        std::bind(&MuJoCoROS::update_visualization, this));

   // Create joint state publisher and joint command subscriber
    _jointCommandSubscriber = this->create_subscription<std_msgs::msg::Float64MultiArray>(jointCommandTopicName, 1,  std::bind(&MuJoCoROS::joint_command_callback, this, std::placeholders::_1));
    
    _jointStatePublisher = this->create_publisher<sensor_msgs::msg::JointState>(jointStateTopicName, 1);
                      
             
    // Initialize Graphics Library FrameWork (GLFW)
    if (not glfwInit()) throw std::runtime_error("Failed to initialise Graphics Library Framework (GLFW).");

    _window = glfwCreateWindow(1600, 1000, "MuJoCo Visualization", nullptr, nullptr);
    
    if (not _window) throw std::runtime_error("Failed to create Graphics Library Framework (GLFW) window.");
    
    // Make the OpenGL context current
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);                    // NOTE TO SELF: CHECK THIS ARGUMENT

    // Initialize MuJoCo rendering context
    mjv_defaultCamera(&_camera);
    mjv_defaultOption(&_renderingOptions);
    mjv_defaultPerturb(&_perturbation);
    mjr_defaultContext(&_context);
    mjv_makeScene(_model, &_scene, 1000);
    
    // Declare & get parameters for camera, visualisation
    _camera.azimuth      = this->declare_parameter<double>("camera_azimuth", 135);    
    _camera.distance     = this->declare_parameter<double>("camera_distance", 2.5);
    _camera.elevation    = this->declare_parameter<double>("camera_elevation", -35);
    _camera.orthographic = this->declare_parameter<bool>("camera_orthographic", true);
    
    auto focalPoint = this->declare_parameter<std::vector<double>>("camera_focal_point", {0.0, 0.0, 0.5});
    
    for(int i = 0; i < 3; ++i) _camera.lookat[i] = focalPoint[i]; 

    // Register mouse callbacks so the user can rotate/pan/zoom with the mouse
    glfwSetWindowUserPointer(_window, this);
    glfwSetMouseButtonCallback(_window, mouse_button_callback);
    glfwSetCursorPosCallback(_window,   cursor_pos_callback);
    glfwSetScrollCallback(_window,      scroll_callback);

    // Create MuJoCo rendering context
    glfwMakeContextCurrent(_window);
    mjr_makeContext(_model, &_context, mjFONTSCALE_100);
    
    RCLCPP_INFO(this->get_logger(), "MuJoCo simulation initiated. "
                                    "Publishing the joint state to '%s' topic. "
                                    "Subscribing to joint %s commands via '%s' topic.",
                                    jointStateTopicName.c_str(), controlMode.c_str(), jointCommandTopicName.c_str());

}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Update the simulation                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::update_simulation()
{
    if (not _model and not _jointState)
    {
        RCLCPP_ERROR(this->get_logger(), "MuJoCo model or data is not initialized. How did that happen?");
        return;
    }

    if (_controlMode == TORQUE)
    {
        const int n = std::min(_model->nq, _model->nu);                                             // Never write past the end of ctrl (the Panda has 9 joints but 8 actuators)
        for (int i = 0; i < n; ++i)
        {
            if (_torquePassthrough)
            {
                // The Panda's actuators are position servos:  force = gain*ctrl + bias0 + bias1*q + bias2*qdot.
                // To make the actuator produce exactly the commanded torque we invert that law with the model's
                // own gain / bias numbers:  ctrl = (torque - bias) / gain.  (The model itself is not modified.)
                if (not _torqueReceived and _model->actuator_trntype[i] == mjTRN_JOINT)
                {
                    // No torque command yet: let the actuator's own servo law hold the initial pose, so the arm does not collapse before the controller starts
                    _jointState->ctrl[i] = _holdTarget[_model->jnt_qposadr[_model->actuator_trnid[2*i]]];
                    continue;
                }
                const double tau = _torqueInput[i];                                                 // Commanded joint torque, held until the next command arrives
                if (_model->actuator_trntype[i] == mjTRN_JOINT
                 && _model->actuator_gaintype[i] == mjGAIN_FIXED
                 && _model->actuator_biastype[i] == mjBIAS_AFFINE
                 && _model->actuator_gainprm[mjNGAIN*i] != 0.0)
                {
                    const int joint = _model->actuator_trnid[2*i];
                    const double q  = _jointState->qpos[_model->jnt_qposadr[joint]];
                    const double dq = _jointState->qvel[_model->jnt_dofadr[joint]];
                    const double *bias = &_model->actuator_biasprm[mjNBIAS*i];
                    _jointState->ctrl[i] = (tau - (bias[0] + bias[1]*q + bias[2]*dq)) / _model->actuator_gainprm[mjNGAIN*i];
                }
                else _jointState->ctrl[i] = tau;                                                    // e.g. the tendon-driven gripper: pass the value straight through
            }
            else
            {
                _jointState->ctrl[i] = _torqueInput[i]                                              // Transfer torque input
                                     + _jointState->qfrc_bias[i]                                    // Compensate for gravity
                                     - 0.01*_jointState->qvel[i];                                   // Add some damping

                _torqueInput[i] = 0.0;                                                              // Clear value
            }
        }
    }

    mj_step(_model, _jointState);                                                                   // Take a step in the simulation

    // Add joint state data to ROS2 message
    for (int i = 0; i < _model->nq; ++i)
    {
        _jointStateMessage.position[i] = _jointState->qpos[i];
        _jointStateMessage.velocity[i] = _jointState->qvel[i];
        _jointStateMessage.effort[i]   = _jointState->actuator_force[i];
    }

    _jointStateMessage.header.stamp = this->get_clock()->now();                                     // Add current time stamp (for rqt)
    
    _jointStatePublisher->publish(_jointStateMessage);                                              // Publish the joint state message
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Handle joint commands                                       //
////////////////////////////////////////////////////////////////////////////////////////////////////
void
MuJoCoROS::joint_command_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() != _model->nq)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Received joint command with incorrect size.");
        return;
    }
    else
    {
        switch(_controlMode)
        {
            case POSITION:
            {
                for(int i = 0; i < std::min(_model->nq, _model->nu); ++i)
                {
                    _jointState->ctrl[i] = msg->data[i];                                            // Assign control input directly
                }
                break;
            }
            case VELOCITY:
            {
                for(int i = 0; i < std::min(_model->nq, _model->nu); ++i)
                {
                    _jointState->ctrl[i] += msg->data[i] / (double)_simFrequency;                   // Integrate velocity to position level
                }
                break;
            }
            case TORQUE:
            {
                for(int i = 0; i < _model->nq; ++i)
                {
                    _torqueInput[i] = msg->data[i];
                }
                _torqueReceived = true;
                break;
            }
            default:
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                     "Unknown control mode. Unable to set joint commands.");
                break;
            }
        }
    }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                    Update the 3D simulation                                    //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::update_visualization()
{
    glfwMakeContextCurrent(_window);                                                                // Ensure OpenGL context is current
    
    mjv_updateScene(_model, _jointState, &_renderingOptions, NULL, &_camera, mjCAT_ALL, &_scene);   // Update 3D rendering

    // Get framebuffer size
    int width, height;
    glfwGetFramebufferSize(_window, &width, &height);
    mjrRect viewport = {0, 0, width, height};

    mjr_render(viewport, &_scene, &_context); // Render scene
    
    // Swap buffers and process events
    glfwSwapBuffers(_window);
    glfwPollEvents();
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                        Mouse callbacks                                         //
////////////////////////////////////////////////////////////////////////////////////////////////////
void MuJoCoROS::mouse_button_callback(GLFWwindow* window, int button, int action, int /*mods*/)
{
    auto* self = static_cast<MuJoCoROS*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT)  self->_mouseButtonLeft  = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_RIGHT) self->_mouseButtonRight = (action == GLFW_PRESS);
    glfwGetCursorPos(window, &self->_lastMouseX, &self->_lastMouseY);
}

void MuJoCoROS::cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    auto* self = static_cast<MuJoCoROS*>(glfwGetWindowUserPointer(window));

    double dx = xpos - self->_lastMouseX;
    double dy = ypos - self->_lastMouseY;
    self->_lastMouseX = xpos;
    self->_lastMouseY = ypos;

    if (!self->_mouseButtonLeft && !self->_mouseButtonRight) return;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    mjtMouse action = self->_mouseButtonLeft ? mjMOUSE_ROTATE_V : mjMOUSE_MOVE_H;
    mjv_moveCamera(self->_model, action, dx/height, dy/height, &self->_camera);
}

void MuJoCoROS::scroll_callback(GLFWwindow* window, double /*xoffset*/, double yoffset)
{
    auto* self = static_cast<MuJoCoROS*>(glfwGetWindowUserPointer(window));
    mjv_moveCamera(self->_model, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &self->_camera);
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                           Destructor                                           //
////////////////////////////////////////////////////////////////////////////////////////////////////
MuJoCoROS::~MuJoCoROS()
{
    mj_deleteData(_jointState);
    mj_deleteModel(_model);
    mjv_freeScene(&_scene);
    mjr_freeContext(&_context);
    glfwDestroyWindow(_window);
    glfwTerminate();
}     
