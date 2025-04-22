#!/usr/bin/env python3

import rospy
import os
import numpy as np
import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from turbojpeg import TurboJPEG

class BlueCrosswalkTestNode(DTROS):
    def __init__(self, node_name):
        super(BlueCrosswalkTestNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name
        self.veh = os.environ['VEHICLE_NAME']
        
        # Define color masks
        self.BLUE_MASK = [(90, 100, 50), (130, 255, 200)]  # Blue HSV range
        self.DUCK_MASK = [(22, 120, 120), (30, 255, 255)]  # Yellow HSV range for duck detection
        
        # Initialize detection parameters
        self.blue_crosswalk_detected = True
        self.duck_detected = False
        self.at_crosswalk = False
        self.blue_line_area_threshold = 400  # Minimum area to consider a valid blue line
        self.duck_area_threshold = 500  # Minimum area to consider a valid duck
        self.crosswalk_wait_time = 0
        self.crosswalk_max_wait = 80  # About 10 seconds at 8Hz
        
        # Publishers and subscribers
        self.jpeg = TurboJPEG()
        
        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=1,
            buff_size="20MB"
        )
        
        # Publisher for velocity commands
        self.vel_pub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        
        # Publishers for debug images
        self.blue_line_debug_pub = rospy.Publisher(
            f"/{self.veh}/blue_crosswalk_test/debug/compressed",
            CompressedImage,
            queue_size=1
        )
        
        self.duck_debug_pub = rospy.Publisher(
            f"/{self.veh}/duck_detection_test/debug/compressed",
            CompressedImage,
            queue_size=1
        )
        
        # Initialize velocity commands
        self.twist = Twist2DStamped(v=0, omega=0)
        
        rospy.loginfo(f"[{self.node_name}] Initialized")
    
    def detect_blue_crosswalk(self, img):
        """Process the image to detect blue crosswalk lines on the path"""
        # Use a lower portion of the image to detect blue lines closer to the robot
        crop_img = img[350:-1, :, :]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, self.BLUE_MASK[0], self.BLUE_MASK[1])
        
        # Apply morphological operations to improve detection
        kernel = np.ones((3, 3), np.uint8)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
        
        # Find contours in the blue mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a debug image
        debug_img = crop_img.copy()
        
        # Check for blue line
        # self.blue_crosswalk_detected = False
        max_blue_area = 0
        max_blue_idx = -1
        
        # Get the height of the cropped image to check position
        height = crop_img.shape[0]
        closest_y = height  # Initialize to the bottom of the image
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_blue_area:
                max_blue_area = area
                max_blue_idx = i
            
            # Find the lowest point (closest to robot) in the contour
            if area > self.blue_line_area_threshold / 2:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cy = int(M["m01"] / M["m00"])
                    if cy < closest_y:
                        closest_y = cy
        
        # If we found a significant blue contour, consider it a crosswalk line
        if max_blue_area > self.blue_line_area_threshold:
            # Check if the blue line is in the lower portion of the frame (closer to the robot)
            position_threshold = height / 1.5
            
            if closest_y > position_threshold:
                self.blue_crosswalk_detected = True
                cv2.putText(debug_img, "BLUE CROSSWALK DETECTED!", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(debug_img, "BLUE LINE TOO FAR", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
            # Draw the largest blue contour for debugging
            cv2.drawContours(debug_img, contours, max_blue_idx, (255, 0, 0), 3)
            
            # Add text showing detection
            cv2.putText(debug_img, f"BLUE CROSSWALK DETECTED!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show position information
            cv2.putText(debug_img, f"Closest point: {closest_y}/{height}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Add text showing no detection
            cv2.putText(debug_img, "No blue crosswalk detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add text showing max area
        cv2.putText(debug_img, f"Max blue area: {max_blue_area}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Publish debug image for blue crosswalk detection
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        self.blue_line_debug_pub.publish(debug_msg)
        
        return

    def detect_duck(self, img):
        """Detect if there is a duck (yellow object) in the road, with filtering to avoid detecting lane markings"""
        # Focus on the center of the road rather than the edges where lane markings appear
        height, width = img.shape[:2]
        # Create a region of interest that excludes the sides of the image where lane markings usually appear
        roi_x_start = int(width * 0.45)  # Left 45% excluded
        roi_x_end = int(width * 0.65)    # Right 35% excluded
        roi_y_start = 150                # Top of region
        roi_y_end = 350                  # Bottom of region
        
        # Extract the region of interest
        crop_img = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # Create mask for yellow color (duck)
        duck_mask = cv2.inRange(hsv, self.DUCK_MASK[0], self.DUCK_MASK[1])
        
        # Apply morphological operations to improve detection
        kernel = np.ones((5, 5), np.uint8)
        # Opening operation to remove small noise
        duck_mask = cv2.morphologyEx(duck_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Dilation to connect nearby parts
        duck_mask = cv2.dilate(duck_mask, kernel, iterations=2)
        
        # Find contours in the duck mask
        contours, _ = cv2.findContours(duck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a debug image
        debug_img = crop_img.copy()
        
        # Check for duck with improved filtering
        self.duck_detected = False
        valid_duck_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.duck_area_threshold:
                continue
                
            # Filter by shape - ducks are more compact than line markings
            # Calculate aspect ratio of the contour bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Lane markings have high aspect ratio (wide and short)
            # Ducks should have aspect ratio closer to 1 (more square-like)
            if aspect_ratio > 3.0:  # Skip elongated objects (likely lane markings)
                continue
                
            # Check circularity/roundness
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Duck should be more circular than a line
            if circularity < 0.3:  # Adjust this threshold as needed
                continue
                
            # If it passed all filters, add to valid contours
            valid_duck_contours.append(contour)
            
        # Check if we have valid duck contours
        if valid_duck_contours:
            # Find the largest valid contour
            largest_contour = max(valid_duck_contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            self.duck_detected = True
            
            # Draw the contour on debug image
            cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 255), 3)
            
            # Add text showing detection
            cv2.putText(debug_img, f"DUCK DETECTED! Area: {largest_area}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Add text showing no detection
            cv2.putText(debug_img, "No duck detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw ROI boundaries on the original image for debugging
        full_debug = img.copy()
        cv2.rectangle(full_debug, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        
        if self.duck_detected:
            cv2.putText(full_debug, "DUCK DETECTED", (width//2 - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Publish debug image for duck detection
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(full_debug))
        self.duck_debug_pub.publish(debug_msg)
        
        return self.duck_detected
    
    def image_callback(self, msg):
        """Process incoming image messages"""
        try:
            # Decode image
            img = self.jpeg.decode(msg.data)
            
            # Check if we're already at a crosswalk
            if self.at_crosswalk:
                # Detect ducks
                self.detect_duck(img)
                
                # Handle crosswalk behavior
                self.handle_crosswalk()
            else:
                # Detect blue crosswalk
                if self.blue_crosswalk_detected:
                    self.detect_blue_crosswalk(img)
                    rospy.loginfo("Blue crosswalk detected!")
                    self.at_crosswalk = True
                    self.crosswalk_wait_time = 0
                    
                # Normal driving behavior when not at crosswalk
                self.twist.v = 0.3  # Forward velocity
                self.twist.omega = 0  # No turning
                self.vel_pub.publish(self.twist)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
    
    def handle_crosswalk(self):
        """Handle behavior at crosswalk"""
        # Stop the robot
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        
        # Check for duck
        if self.duck_detected:
            rospy.loginfo("Duck detected at crosswalk, waiting for it to cross...")
            self.crosswalk_wait_time = 0  # Reset wait time
        else:
            # Increment wait counter
            self.crosswalk_wait_time += 1
            rospy.loginfo(f"Waiting at crosswalk, no duck detected. Wait time: {self.crosswalk_wait_time}/{self.crosswalk_max_wait}")
            
            # If we've waited long enough without seeing a duck or the duck has passed
            if self.crosswalk_wait_time >= self.crosswalk_max_wait:
                rospy.loginfo("Resuming after crosswalk")
                self.at_crosswalk = False
                self.crosswalk_wait_time = 0
                self.duck_detected = False
                self.blue_crosswalk_detected = False
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(8)  # 8Hz
        
        while not rospy.is_shutdown():
            # Update status messages based on current detection status
            if self.at_crosswalk:
                if self.duck_detected:
                    rospy.loginfo_throttle(2, "At crosswalk: Duck detected, waiting...")
                else:
                    rospy.loginfo_throttle(2, f"At crosswalk: No duck, waiting... ({self.crosswalk_wait_time}/{self.crosswalk_max_wait})")
            else:
                rospy.loginfo_throttle(2, "Looking for blue crosswalk...")
            
            rate.sleep()

if __name__ == "__main__":
    # Initialize the node
    node = BlueCrosswalkTestNode(node_name="blue_crosswalk_test_node")
    # Run the node
    node.run()