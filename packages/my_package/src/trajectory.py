#!/usr/bin/env python3

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import math

def load_and_plot_trajectory():
    times = []
    x_positions = []
    y_positions = []
    theta = 0.0
    x = 0.0    
    y = 0.0   
    
    wheel_distance = 0.1 
    wheel_radius = 0.033  
    dt = 0.7           

    try:

        bag = rosbag.Bag('move-old.bag') 
      
        for topic, msg, t in bag.read_messages(topics=['/csc22905/wheels_driver_node/wheels_cmd']):
          
            v_left = msg.vel_left
            v_right = msg.vel_right
            
            
            v = wheel_radius * (v_right + v_left) / 2.0   
            omega = wheel_radius * (v_right - v_left) / wheel_distance 
            
        
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += omega * dt
      
            theta = math.atan2(math.sin(theta), math.cos(theta))
            
     
            times.append(t.to_sec())
            x_positions.append(x)
            y_positions.append(y)
        
        bag.close()
        
        plt.figure(figsize=(10, 10))
        plt.plot(x_positions, y_positions, 'b-', label='Robot Trajectory')
        plt.plot(x_positions[0], y_positions[0], 'go', label='Start')
        plt.plot(x_positions[-1], y_positions[-1], 'ro', label='End')
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Duckiebot D-Pattern Trajectory')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        plt.savefig('d_pattern_trajectory.png')
        plt.show()
        
        print("Trajectory plotting completed!")
        
    except Exception as e:
        print(f"Error processing bag file: {e}")

if __name__ == '__main__':
    load_and_plot_trajectory()