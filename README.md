# followbot
A gazebo simulation platform where the automated guided vehicle (AGV) follows the green tape strip laid on the floor. 
## warehouse: 
1. fred robot:

![avg_warehouse](https://github.com/TempleRAIL/followbot/blob/main/images/avg_warehouse_fred_robot.png)

2. turtlebot:

![avg_warehouse](https://github.com/TempleRAIL/followbot/blob/main/images/avg_warehouse.png)

## tape detection:
![tape_detection](https://github.com/TempleRAIL/followbot/blob/main/images/tape_detection.png)

# Usage
## install: 
In order to install necessary requirements, run:
1. Ros Kinetic:
```Bash
sudo apt-get install ros-kinetic-turtlebot
sudo apt-get install ros-kinetic-turtlebot-simulator
sudo apt-get install ros-kinetic-kobuki*
cd ~/catkin_ws/src
git clone https://github.com/TempleRAIL/followbot.git
cd ..
catkin_make
``` 
2. Ros Melodic:
```Bash
cd ~/catkin_ws/src
git clone https://github.com/TempleRAIL/followbot.git
cd followbot
sh setup_melodic.sh
``` 

## run: 
In order to start the simulation, run:
1. fred robot:
```Bash
roslaunch followbot followbot_fred_robot.launch
```  
2. turtlebot 2 robot:
```Bash
roslaunch followbot followbot_turtlebot.launch
```  

## change track: 
There are two different tape tracks: agv test track: and agv test broken track. In order to change the tape track, please modify the "./world/warehouse_20m.material" file:    
a) agv test track:
```
      texture_unit
      {
        texture agv_test_track_v1.png
      }
``` 
b) agv test broken track:
```
      texture_unit
      {
        texture agv_test_broken_track_v1.png
      }
```
