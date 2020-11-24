# followbot
A gazebo simulation platform where the automated guided vehicle (AGV) follows the green tape strip laid on the floor. 
### warehouse: 
![avg_warehouse](https://github.com/TempleRAIL/followbot/blob/master/images/avg_warehouse.png)
### tape detection:
![tape_detection](https://github.com/TempleRAIL/followbot/blob/master/images/tape_detection.png)
# Usage
In order to start the simulation, run:
```Bash
roslaunch followbot followbot.launch
```  
There are two different tape tracks: agv test track: and agv test broken track. In order to change the tape track, please modify the "./world/warehouse.material" file:    
a) agv test track:
```
      texture_unit
      {
        texture agv_test_track.png
      }
``` 
b) agv test broken track:
```
      texture_unit
      {
        texture agv_test_broken_track.png
      }
```
