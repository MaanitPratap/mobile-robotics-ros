#!/usr/bin/env python3

import rospy
import os
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import Range
from dt_apriltags import Detector
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class DirectParkingNode(DTROS):
    def __init__(self, node_name):
        super(DirectParkingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh = os.environ['VEHICLE_NAME']

        # Get the stall parameter (1-4)
        self.stall = 3
        rospy.loginfo(f"Direct parking test initialized for stall {self.stall}")

        # State machine variables
        self.state = 0
        self.state_start_time = rospy.get_time()
        
        # TOF sensor reading
        self.tof_distance = 1.0
        self.last_valid_tof = 1.0  # Store last valid TOF reading
        self.tof_last_update = rospy.get_time()
        
        # AprilTag detection variables
        self.tag_detector = Detector(families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        self.tag_detected = False
        self.tag_x_center = 0
        self.tag_id = -1  # Store tag ID for debugging
        self.target_tag_id = -1  # Store the ID of the first detected tag to focus on
        self.has_target_tag = False  # Flag to indicate if we have a target tag
        self.image_width = 0
        self.crop_width = 320  # Reduced width of cropped region in pixels (was 320)
        self.tag_detection_count = 0  # Count consecutive detections
        self.tag_lost_count = 0       # Count consecutive frames without detection
        self.last_valid_tag_center = 0  # Last known position for smoothing
        self.tag_alignment_count = 0  # Count consecutive aligned frames
        
        # Publishers & Subscribers
        self.tof_sub = rospy.Subscriber(f"/{self.veh}/front_center_tof_driver_node/range",
                                        Range,
                                        self.cb_tof,
                                        queue_size=1)
                                        
        # Add camera subscriber for AprilTag detection
        self.camera_sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                         CompressedImage,
                                         self.cb_camera,
                                         queue_size=1)
                                       
        self.vel_pub = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        # Add debug image publisher
        self.debug_pub = rospy.Publisher(f"/{self.veh}/direct_parking_node/debug/image/compressed",
                                      CompressedImage,
                                      queue_size=1)
                                       
        # Constants for parking maneuver
        self.short_drive_time = 2.5  # Time to drive ~7 inches at slow speed
        self.long_drive_time = 4     # Time to drive ~17 inches at slow speed
        self.turn_time = 1.5         # Time for 90-degree turn
        self.drive_speed = 0.3      # Forward speed
        self.turn_speed = 3          # Angular velocity for turns
        self.align_speed = 3.0       # Speed for alignment correction
        self.alignment_threshold = 20 # Pixels from center
        self.stop_distance = 0.25    # Distance to stop in meters (25cm)
        self.emergency_stop_distance = 0.15  # Emergency stop distance (15cm)
        self.tag_confidence_threshold = 3  # Number of consecutive detections needed
        self.tof_stop_enabled = True  # Flag to enable TOF-based stopping

    def cb_tof(self, msg):
        """Process Time-of-Flight sensor data"""
        # Update timestamp for last valid reading
        self.tof_last_update = rospy.get_time()
        
        # Check if reading is valid (not inf or nan)
        if msg.range > 0.0 and msg.range < 10.0 and not np.isnan(msg.range) and not np.isinf(msg.range):
            self.tof_distance = msg.range
            self.last_valid_tof = msg.range
            
            # Emergency stop if we're too close to an obstacle (regardless of state)
            if self.tof_stop_enabled and self.tof_distance < self.emergency_stop_distance and self.state != 3:
                self.state = 3  # Immediately go to stop state
                self.state_start_time = rospy.get_time()
                rospy.logwarn(f"EMERGENCY STOP! TOF distance: {self.tof_distance:.2f}m")
        else:
            # Use last valid reading if current one is invalid
            self.tof_distance = self.last_valid_tof
            rospy.logwarn_throttle(1.0, "Invalid TOF reading, using last valid value")
    
    def cb_camera(self, msg):
        """Process camera images for AprilTag detection"""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Store image width for alignment calculations
            self.image_width = cv_image.shape[1]
            image_height = cv_image.shape[0]
            
            # Calculate crop region based on stall position
            # For stalls 1 and 2 (right side), crop left side of image 
            # For stalls 3 and 4 (left side), crop right side of image
            image_center_x = self.image_width // 2
            
            if self.stall == 1 or self.stall == 3:
                # For right stalls, take a narrow region on the right side
                crop_start = 0
                crop_end = min(self.crop_width, self.image_width)
            else:
                # For left stalls, take a narrow region on the left side
                crop_start = max(0, self.image_width - self.crop_width)
                crop_end = self.image_width
                
            # Crop image to only see the relevant part
            cropped_image = cv_image[:, crop_start:crop_end]
            
            # Convert to grayscale for AprilTag detection
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Create a debug image (copy of original)
            debug_image = cv_image.copy()
            
            # Draw crop region on debug image
            cv2.rectangle(debug_image, 
                          (crop_start, 0), 
                          (crop_end, debug_image.shape[0]), 
                          (0, 255, 0), 2)
            
            # Detect AprilTags
            tags = self.tag_detector.detect(gray)
            
            # Filter for our target tag if we already have one
            target_tag = None
            if tags:
                if self.has_target_tag:
                    # Look for our target tag ID in the detected tags
                    for tag in tags:
                        if tag.tag_id == self.target_tag_id:
                            target_tag = tag
                            break
                else:
                    # First detection - set the first detected tag as our target
                    target_tag = tags[0]
                    self.target_tag_id = target_tag.tag_id
                    self.has_target_tag = True
                    rospy.loginfo(f"*** LOCKING ONTO TARGET TAG ID: {self.target_tag_id} ***")
            
            if target_tag is not None:
                # We found our target tag
                tag = target_tag
                self.tag_id = tag.tag_id
                
                # Calculate tag's center position in the cropped image
                center_x = int(sum(tag.corners[:,0]) / 4)
                # Convert to position in original image
                full_frame_center_x = center_x + crop_start
                
                # Apply smoothing with previous position for stability
                if self.tag_detected:
                    # If already tracking, apply smoothing
                    alpha = 0.7  # Weight for new reading (0.7 new, 0.3 old)
                    smoothed_center = alpha * full_frame_center_x + (1-alpha) * self.last_valid_tag_center
                    self.tag_x_center = int(smoothed_center)
                else:
                    # Initial detection, no smoothing
                    self.tag_x_center = full_frame_center_x
                
                # Store as last valid center
                self.last_valid_tag_center = self.tag_x_center
                
                # Increment detection counter
                self.tag_detection_count += 1
                self.tag_lost_count = 0
                
                # Only consider tag detected if we've seen it for a few frames
                if self.tag_detection_count >= self.tag_confidence_threshold:
                    self.tag_detected = True
                
                # Draw tag detection on debug image
                # Convert corners back to original image coordinates
                corners_original = tag.corners.copy()
                corners_original[:,0] += crop_start
                
                # Draw the tag outline
                for i in range(4):
                    pt1 = (int(corners_original[i][0]), int(corners_original[i][1]))
                    pt2 = (int(corners_original[(i+1)%4][0]), int(corners_original[(i+1)%4][1]))
                    cv2.line(debug_image, pt1, pt2, (0, 255, 0), 2)
                
                # Draw the tag center
                cv2.circle(debug_image, (self.tag_x_center, int(sum(tag.corners[:,1])/4)), 
                           5, (0, 0, 255), -1)
                
                # Add text with tag ID and position
                cv2.putText(debug_image, f"TARGET Tag ID: {tag.tag_id}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(debug_image, f"X: {self.tag_x_center}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Log detection
                rospy.loginfo_throttle(1.0, f"Target AprilTag {tag.tag_id} detected at x={self.tag_x_center} (count={self.tag_detection_count})")
            else:
                # Our target tag not found
                # Increment lost counter and reset detection counter
                self.tag_lost_count += 1
                
                # Only consider tag lost if we haven't seen it for a few frames
                if self.tag_lost_count > 5:
                    self.tag_detection_count = 0
                    self.tag_detected = False
                
                # Add text showing no target tag detected
                if self.has_target_tag:
                    cv2.putText(debug_image, f"Target Tag {self.target_tag_id} not detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(debug_image, "No AprilTag detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add image center line
            image_center = self.image_width // 2
            cv2.line(debug_image, (image_center, 0), (image_center, debug_image.shape[0]), 
                    (255, 0, 0), 1)
            
            # Add current state and ToF distance
            cv2.putText(debug_image, f"State: {self.state}", 
                       (10, debug_image.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(debug_image, f"ToF: {self.tof_distance:.2f}m", 
                       (10, debug_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add emergency stop threshold line
            emergency_text = f"Stop at: {self.stop_distance:.2f}m"
            cv2.putText(debug_image, emergency_text, 
                       (10, debug_image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Draw alignment target zone
            if self.tag_detected:
                left_limit = image_center - self.alignment_threshold
                right_limit = image_center + self.alignment_threshold
                cv2.line(debug_image, (left_limit, 0), (left_limit, debug_image.shape[0]), (0, 255, 255), 1)
                cv2.line(debug_image, (right_limit, 0), (right_limit, debug_image.shape[0]), (0, 255, 255), 1)
            
            # Publish the debug image
            debug_msg = CompressedImage()
            debug_msg.header.stamp = rospy.Time.now()
            debug_msg.format = "jpeg"
            debug_msg.data = np.array(cv2.imencode('.jpg', debug_image)[1]).tostring()
            self.debug_pub.publish(debug_msg)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def park(self):
        """Execute the direct parking state machine"""
        current_time = rospy.get_time()
        elapsed = current_time - self.state_start_time
        
        # Create command message
        cmd = Twist2DStamped()
        
        # Emergency TOF check for all states except state 3 (already stopped)
        if self.state != 3 and self.tof_distance < self.emergency_stop_distance:
            self.state = 3
            self.state_start_time = current_time
            rospy.logwarn(f"EMERGENCY STOP! TOF distance: {self.tof_distance:.2f}m")
            # Send stop command immediately
            cmd.v = 0
            cmd.omega = 0
            self.vel_pub.publish(cmd)
            return
        
        # State machine for direct parking
        if self.state == 0:
            # Initial state - drive forward a specified distance based on stall number
            if self.stall == 1 or self.stall == 3:
                drive_time = self.short_drive_time
                rospy.loginfo_throttle(1.0, f"State 0: Moving forward short distance for stall {self.stall}")
            else:
                drive_time = self.long_drive_time
                rospy.loginfo_throttle(1.0, f"State 0: Moving forward longer distance for stall {self.stall}")
                
            cmd.v = self.drive_speed
            cmd.omega = 0
            
            if elapsed > drive_time:
                self.state = 1
                self.state_start_time = current_time
                rospy.loginfo("Moving to state 1: Turn toward parking stall")
                
        elif self.state == 1:
            # Turn 90 degrees in the appropriate direction
            if self.stall == 1 or self.stall == 2:
                # Right turn for stalls 1 and 2
                cmd.v = 0.1
                cmd.omega = -self.turn_speed
                rospy.loginfo_throttle(1.0, "Turning right 90 degrees")
            else:
                # Left turn for stalls 3 and 4
                cmd.v = 0.1
                cmd.omega = self.turn_speed
                rospy.loginfo_throttle(1.0, "Turning left 90 degrees")
            
            if elapsed > self.turn_time:
                self.state = 1.5  # New state for alignment
                self.state_start_time = current_time
                rospy.loginfo("Moving to state 1.5: Searching for AprilTag")
                # Reset tag tracking when starting search
                if not self.has_target_tag:
                    self.tag_detected = False
                    self.tag_detection_count = 0
                    self.tag_lost_count = 0
                
        elif self.state == 1.5:
            # New state: Move forward until AprilTag is detected
            if not self.tag_detected:
                # Move forward slowly until tag is detected
                cmd.v = 0.2  # Slow forward movement
                cmd.omega = 0  # Keep straight
                
                # Log search info
                rospy.loginfo_throttle(1.0, f"Moving forward to find target AprilTag...")
                
                # If we've been searching too long, move to next state anyway
                if elapsed > 7.0:
                    if self.has_target_tag:
                        rospy.logwarn(f"Could not find target tag {self.target_tag_id}. Moving to state 2 anyway.")
                    else:
                        rospy.logwarn("Could not find any AprilTag. Moving to state 2 anyway.")
                    self.state = 2
                    self.state_start_time = current_time
            else:
                # Once tag is detected, move to alignment state
                self.state = 1.6
                self.state_start_time = current_time
                self.tag_alignment_count = 0  # Reset alignment counter
                rospy.loginfo(f"Target AprilTag {self.target_tag_id} detected. Moving to state 1.6: Aligning with tag.")
                
        elif self.state == 1.6:
            # Alignment state: Center on the AprilTag
            if self.tag_detected:
                # Calculate center of the image
                image_center = self.image_width / 2
                # Calculate alignment error
                error = self.tag_x_center - image_center
                
                # Apply proportional control to align
                cmd.v = 0.15  # Slower forward movement during alignment for better precision
                
                # Adjust error directionality based on stall
                if self.stall == 1 or self.stall == 2:
                    # For right-side stalls, keep original error direction
                    alignment_error = error
                else:
                    # For left-side stalls, invert error direction
                    alignment_error = -error
                
                # Apply proportional control with scaling factor
                # Add deadband to prevent oscillation
                if abs(error) > 5:  # Small deadband to prevent oscillation
                    cmd.omega = -alignment_error * 0.015  # Increased P gain for faster alignment
                else:
                    cmd.omega = 0  # Zero correction when very close to center
                
                # Log alignment info
                rospy.loginfo_throttle(0.5, f"Aligning: Tag at {self.tag_x_center}, Error: {error}")
                
                # Check if TOF is getting close during alignment
                if self.tof_distance < self.stop_distance:
                    self.state = 3  # Go straight to stopping if we're close enough
                    self.state_start_time = current_time
                    rospy.loginfo(f"TOF detected close obstacle during alignment: {self.tof_distance:.2f}m. Stopping.")
                
                # If aligned within threshold, move to next state
                elif abs(error) < self.alignment_threshold:
                    self.tag_alignment_count += 1
                    
                    # Require multiple consecutive aligned frames
                    if self.tag_alignment_count > 5:
                        self.state = 2
                        self.state_start_time = current_time
                        rospy.loginfo(f"Aligned with target tag {self.target_tag_id}. Moving to state 2: Drive straight into stall")
                else:
                    # Reset alignment counter if not aligned
                    self.tag_alignment_count = 0
            else:
                # If tag lost during alignment, continue with last known position for a bit
                if self.tag_lost_count < 10:
                    # Use last known position for a brief period
                    rospy.logwarn_throttle(1.0, f"Target tag {self.target_tag_id} temporarily lost during alignment, using last position")
                    cmd.v = 0.1  # Slower movement when tag is lost
                else:
                    # If tag lost for too long, go back to search state
                    self.state = 1.5
                    self.state_start_time = current_time
                    rospy.logwarn(f"Lost target tag {self.target_tag_id} during alignment. Returning to search.")
                
        elif self.state == 2:
            # Drive straight until TOF sensor detects obstacle
            cmd.v = 0.15  # Slower approach speed for more controlled stopping
            cmd.omega = 0
            
            # Check for TOF sensor health
            tof_age = current_time - self.tof_last_update
            if tof_age > 1.0:  # No TOF updates in more than 1 second
                rospy.logwarn_throttle(1.0, f"TOF sensor not updating ({tof_age:.1f}s old)")
            
            # Log the TOF distance
            rospy.loginfo_throttle(0.5, f"Approaching stall, TOF distance: {self.tof_distance:.2f}m (stop at {self.stop_distance:.2f}m)")
            
            if self.tof_distance < self.stop_distance:  # Obstacle detected within stop_distance
                self.state = 3
                self.state_start_time = current_time
                rospy.loginfo(f"Obstacle detected at {self.tof_distance:.2f}m, stopping")
            elif elapsed > 8.0:  # Safety timeout after 8 seconds
                self.state = 3
                self.state_start_time = current_time
                rospy.loginfo("Safety timeout reached, stopping")
                
        elif self.state == 3:
            # Parking complete - stop the robot
            cmd.v = 0
            cmd.omega = 0
            
            if elapsed < 1.0:  # Just log once
                if self.has_target_tag:
                    rospy.loginfo(f"Parking complete in stall {self.stall} (targeting tag {self.target_tag_id})!")
                else:
                    rospy.loginfo(f"Parking complete in stall {self.stall}!")
        
        # Publish the command
        self.vel_pub.publish(cmd)

    def run(self):
        """Main loop for the node"""
        rate = rospy.Rate(10)  # 10Hz
        
        # Wait a bit before starting
        rospy.sleep(1.0)
        rospy.loginfo("Starting direct parking test")
        
        while not rospy.is_shutdown():
            # Run the parking state machine
            self.park()
            rate.sleep()

    def on_shutdown(self):
        """Called on node shutdown"""
        # Stop the robot
        cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        self.vel_pub.publish(cmd)
        rospy.loginfo("Direct parking test node shutting down")


if __name__ == "__main__":
    # Initialize the node
    node = DirectParkingNode("direct_parking_node")
    
    # Run the node
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Safety in case shutdown hook doesn't run
        node.on_shutdown()