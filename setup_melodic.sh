ROS_WS=asi_ws

cd $HOME/$ROS_WS/src
if [ ! -d kobuki_msgs ]
then
    git clone https://github.com/yujinrobot/kobuki_msgs
fi
if [ ! -d turtlebot_simulator ]
then
    git clone https://github.com/turtlebot/turtlebot_simulator
fi
if [ ! -d kobuki_msgs ]
then
    git clone https://github.com/turtlebot/turtlebot
fi
if [ ! -d kobuki_bumper2pc ] || [ ! -d kobuki_description ]
then
    git clone https://github.com/yujinrobot/kobuki
    mv kobuki/kobuki_bumper2pc .
    mv kobuki/kobuki_description .
    rm -rf kobuki
fi
if [ ! -d kobuki_gazebo ] || [ ! -d kobuki_gazebo_plugins ]
then
    git clone https://github.com/yujinrobot/kobuki_desktop
    mv kobuki_desktop/kobuki_gazebo .
    mv kobuki_desktop/kobuki_gazebo_plugins .
    rm -rf kobuki_desktop
fi

sudo apt install ros-melodic-yocs-cmd-vel-mux \
    ros-melodic-joy \
    ros-melodic-ros-numpy

