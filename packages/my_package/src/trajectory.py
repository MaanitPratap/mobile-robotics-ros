#!/usr/bin/env python3

import rosbag
import numpy as np
import matplotlib.pyplot as plt
# from duckietown_msgs.msg import WheelsCmdStamped
import math

def load_and_plot_trajectory():
    # Initialize arrays to store trajectory data
    times = []
    x_positions = []
    y_positions = []
    theta = 0.0  # Robot orientation
    x = 0.0      # Initial x position
    y = 0.0      # Initial y position
    
    # Duckiebot parameters
    wheel_distance = 0.1  # Distance between wheels in meters
    wheel_radius = 0.033  # Wheel radius in meters
    dt = 0.7           # Time step for integration (assuming 10Hz messages)

    try:
        # Open the rosbag file
        bag = rosbag.Bag('move-old.bag')  # Replace with your bag filename
        
        # Extract wheel velocities and calculate trajectory
        for topic, msg, t in bag.read_messages(topics=['/csc22905/wheels_driver_node/wheels_cmd']):
            # Get wheel velocities
            v_left = msg.vel_left
            v_right = msg.vel_right
            
            # Calculate robot linear and angular velocities
            v = wheel_radius * (v_right + v_left) / 2.0        # Linear velocity
            omega = wheel_radius * (v_right - v_left) / wheel_distance  # Angular velocity
            
            # Update robot pose using basic kinematics
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += omega * dt
            
            # Normalize theta to [-pi, pi]
            theta = math.atan2(math.sin(theta), math.cos(theta))
            
            # Store positions
            times.append(t.to_sec())
            x_positions.append(x)
            y_positions.append(y)
        
        bag.close()
        
        # Plot the trajectory
        plt.figure(figsize=(10, 10))
        plt.plot(x_positions, y_positions, 'b-', label='Robot Trajectory')
        plt.plot(x_positions[0], y_positions[0], 'go', label='Start')
        plt.plot(x_positions[-1], y_positions[-1], 'ro', label='End')
        
        # Add labels and title
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Duckiebot D-Pattern Trajectory')
        plt.grid(True)
        plt.axis('equal')  # Equal aspect ratio
        plt.legend()
        
        # Save and show the plot
        plt.savefig('d_pattern_trajectory.png')
        plt.show()
        
        print("Trajectory plotting completed!")
        
    except Exception as e:
        print(f"Error processing bag file: {e}")

if __name__ == '__main__':
    load_and_plot_trajectory()