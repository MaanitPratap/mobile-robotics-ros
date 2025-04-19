#!/usr/bin/env python3

import rospy
import os

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped

from led_service.srv import SetLEDColor, SetLEDColorRequest

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
RED_MASK = [(0, 140, 100), (10, 255, 255)]  # Lower red HSV range
RED_MASK2 = [(170, 140, 100), (180, 255, 255)]  # Upper red HSV range (wrapped around)
BLUE_MASK = [(90, 100, 50), (130, 255, 200)]  # Adjusted for duckiebot blue
DEBUG = True
ENGLISH = True
SAFETY = True
AUSSIE = False


class LaneFollowWithDetectionNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowWithDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']

        # Initialize LED service connection
        rospy.loginfo("Initializing LED service connection...")
        self.led_service = None
        try:
            led_service_name = "set_led_color"
            rospy.wait_for_service(led_service_name, timeout=2.0)
            self.led_service = rospy.ServiceProxy(led_service_name, SetLEDColor)
            rospy.loginfo("Connected to LED service")
        except rospy.ROSException:
            rospy.logwarn("LED service not available, will continue without LED indicators")

        # Publishers & Subscribers
        self.tof_sub = rospy.Subscriber("/" + self.veh + "/front_center_tof_driver_node/range",
                                        Range,
                                        self.cb_tof,
                                        queue_size=1)
                                        
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
                                   
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
                                    
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        # Debug publisher for duckiebot detection and red line detection
        self.pub_debug_img = rospy.Publisher("/" + self.veh + "/detection_node/debug/compressed",
                                            CompressedImage,
                                            queue_size=1)
        
        self.pub_red_line_img = rospy.Publisher("/" + self.veh + "/red_line_node/debug/compressed",
                                               CompressedImage,
                                               queue_size=1)
        

        # Debug publisher for blue detection
        self.pub_blue_detect_img = rospy.Publisher("/" + self.veh + "/blue_detection_node/debug/compressed",
                                                CompressedImage,
                                                queue_size=1)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized Combined Lane Following and Detection Node")

        # PID Variables for lane following
        self.proportional = None
        if ENGLISH:
            self.offset = 220
        else:
            self.offset = 220
        if AUSSIE:
            self.offset = 0
            
        # Base velocity and control parameters
        self.base_velocity = 0.3
        self.current_velocity = self.base_velocity
        self.twist = Twist2DStamped(v=self.current_velocity, omega=0)

        # PID parameters
        self.P = 0.025
        self.D = -0.0025
        self.I = 0
        if AUSSIE:
            self.P = 0.025
            self.D = -0.0025
            self.I = 0

        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()
        
        # Safety parameters
        self.tof_distance = 1.0
        self.obj_stop = False
        
        # Duckiebot detection parameters
        self.duckiebot_detected = False
        self.distance_to_duckiebot = float('inf')
        self.safe_distance = 0.4  # Target following distance in meters
        self.min_safe_distance = 0.25  # Absolute minimum distance
        self.following_velocity = 0.25  # Normal following velocity
        self.slowing_velocity = 0.15  # Velocity when getting close

        # Red line detection parameters
        self.red_line_detected = False
        self.red_line_count = 0  # Counter for red lines
        self.red_line_stop_time = 2  # Time to stop at red line (seconds)
        self.red_line_cooldown = 0  # Cooldown to avoid duplicate detections
        self.red_line_cooldown_time = 10  # Cooldown period after detecting a red line
        self.at_red_line = False  # Flag to indicate we're currently stopped at a red line
        self.red_line_area_threshold = 400  # Minimum area of red contour to be considered a line
        self.detection_distance_factor = 1.5  # Lower value means detection closer to the line

        self.current_led_state = "off"  # Initial LED state
        self.was_following = False  # Track if we were following a lane


        # Blue detection parameters
        self.blue_detected = False
        self.blue_area_threshold = 200  # Minimum area for blue detection
        self.blue_detection_active = False  # Flag to indicate when to look for blue
        self.blue_detection_completed = False  # Flag to indicate if detection is complete
        self.last_image = None  # Store the last image for blue detection

        # Turn parameters
        self.turn_duration = 1.5  # How long to execute the turn (seconds)
        self.left_turn_omega = 3  # Angular velocity for left turn
        self.right_turn_omega = -2.5 # Angular velocity for right turn
        
        # Dot pattern detection parameters
        self.min_dot_area = 20  # Minimum area of a dot to be detected
        self.min_pattern_dots = 5  # Minimum number of dots to consider a valid pattern
        
        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        self.set_led_color("green")  # Set initial LED color

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def set_led_color(self, color):
        """Use the LED service to set a color."""
        if not self.led_service:
            return False
            
        if self.current_led_state == color:
            return True  # Already set to this color
            
        try:
            req = SetLEDColorRequest()
            req.color = color
            result = self.led_service(req)
            if result.success:
                self.current_led_state = color
                return True
            return False
        except rospy.ServiceException as e:
            rospy.logwarn(f"LED service call failed: {e}")
            return False
        
    def cb_tof(self, msg):
        """Process Time-of-Flight sensor data"""
        self.tof_distance = msg.range
        if 0.05 < self.tof_distance <= 0.3:
            self.obj_stop = True

    def callback(self, msg):
        """Main image processing callback - handles lane following, duckiebot detection, and red line detection"""
        img = self.jpeg.decode(msg.data)
        self.last_image = img.copy()
        # Process lane following
        self.process_lane_following(img)
        
        # Process duckiebot detection in the upper part of the image
        self.process_duckiebot_detection(img)
        
        # Process red line detection in the lower part of the image
        if not self.at_red_line and self.red_line_cooldown <= 0:  # Only check if not already at a red line
            self.process_red_line_detection(img)
        
        self.detect_blue(img)

    def detect_blue(self, img):
        """Detect if blue is on left or right side of frame"""
        # Use the entire image for blue detection
        crop_img = img.copy()
        height, width = crop_img.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, BLUE_MASK[0], BLUE_MASK[1])
        
        # Split frame into left and right halves
        left_mask = blue_mask[:, :width//2]
        right_mask = blue_mask[:, width//2:]
        
        # Count blue pixels in each half
        left_blue_pixels = cv2.countNonZero(left_mask)
        right_blue_pixels = cv2.countNonZero(right_mask)
        
        # Create debug image
        debug_img = crop_img.copy()
        
        # Draw a vertical line dividing the frame
        cv2.line(debug_img, (width//2, 0), (width//2, height), (0, 255, 0), 2)
        
        # Add pixel counts
        cv2.putText(debug_img, f"Left: {left_blue_pixels}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(debug_img, f"Right: {right_blue_pixels}", (width//2 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Threshold for blue detection
        threshold = 100  # Minimum number of blue pixels to consider detection
        
        # Determine which side has more blue (if above threshold)
        blue_on_left = left_blue_pixels > threshold and left_blue_pixels > right_blue_pixels
        blue_on_right = right_blue_pixels > threshold and right_blue_pixels > left_blue_pixels
        
        # Visualize the results
        if blue_on_left:
            cv2.putText(debug_img, "BLUE ON LEFT", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif blue_on_right:
            cv2.putText(debug_img, "BLUE ON RIGHT", (width//2 + 10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(debug_img, "NO BLUE DETECTED", (width//4, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Publish debug image
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        self.pub_blue_detect_img.publish(debug_msg)
        
        return blue_on_left, blue_on_right
    
    def turn_left(self):
        """Execute a left turn with a larger, smoother curve like in the map"""
        rospy.loginfo("Executing left curve turn")
        # self.set_led_color("blue")
        
        # Temporarily disable lane following by setting proportional to None
        saved_proportional = self.proportional
        self.proportional = None
        
        # Set up turn command for a larger curve
        turn_cmd = Twist2DStamped()
        
        # Use higher forward velocity and moderate turning to create a larger curve
        turn_cmd.v = 0.35  # Higher forward velocity for a wider curve
        turn_cmd.omega = 2.0  # Moderate but consistent turning rate
        
        # Execute the turn for longer duration to complete the curve
        turn_duration = 2.5  # Longer duration for a bigger curve
        
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < turn_duration:
            self.vel_pub.publish(turn_cmd)
            rospy.sleep(0.05)
        
        # Optional: Straighten out slightly at the end
        turn_cmd.v = 0.3
        turn_cmd.omega = 0.5
        for i in range(3):
            self.vel_pub.publish(turn_cmd)
            rospy.sleep(0.1)
        
        # Restore lane following
        self.proportional = saved_proportional
        
        rospy.loginfo("Left curve turn completed")

    def turn_right(self):
        """Execute a sharper right turn with a tighter curve"""
        rospy.loginfo("Executing sharp right curve turn")
        # self.set_led_color("purple")
        
        # Temporarily disable lane following
        saved_proportional = self.proportional
        self.proportional = None
        
        # Set up turn command for a sharper curve
        turn_cmd = Twist2DStamped()
        
        # Less forward velocity and stronger turning for a tighter curve
        turn_cmd.v = 0.25  # Lower forward velocity for a tighter turn
        turn_cmd.omega = -3.0  # Stronger turning rate for a sharper curve
        
        # Execute the turn for a shorter duration
        turn_duration = 1.8  # Shorter duration for a sharper curve
        
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < turn_duration:
            self.vel_pub.publish(turn_cmd)
            rospy.sleep(0.05)
        
        # Restore lane following
        self.proportional = saved_proportional
        
        rospy.loginfo("Right curve turn completed")

    def process_lane_following(self, img):
        """Process the image for lane following"""
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            # Define white color range
            WHITE_MASK = [(0, 0, 180), (180, 30, 255)]
            white_mask = cv2.inRange(hsv, WHITE_MASK[0], WHITE_MASK[1])
            crop = cv2.bitwise_and(crop, crop, mask=white_mask)
            white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Search for white line
            max_white_area = 20
            max_white_idx = -1
            for i in range(len(white_contours)):
                area = cv2.contourArea(white_contours[i])
                if area > max_white_area:
                    max_white_idx = i
                    max_white_area = area
            
            if max_white_idx != -1:
                M = cv2.moments(white_contours[max_white_idx])
                try:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    # For white line, adjust the offset differently
                    self.proportional = cx - int(crop_width / 2) - 300
                    if DEBUG:
                        cv2.drawContours(crop, white_contours, max_white_idx, (255, 0, 0), 3)
                        cv2.circle(crop, (cx, cy), 7, (255, 0, 0), -1)
                except:
                    pass
        
        # Publish debug image if needed
        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def process_duckiebot_detection(self, img):
        """Process the image to detect the Duckiebot dot pattern"""
        # Process the top portion of the image where the dot pattern would be
        crop_img = img[50:250, :, :]
        
        # Convert to grayscale for better dot detection
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to isolate black dots
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of the dots
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a debug image
        debug_img = crop_img.copy()
        
        # Filter contours by size and shape to find dots
        dot_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_dot_area and area < 500:  # Filter out very small or large contours
                # Check if the contour is approximately circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.6:  # Threshold for circularity (1.0 is perfect circle)
                        dot_contours.append(contour)
                        # Draw the detected dot on debug image
                        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
        
        # Check if we have enough dots to consider it the pattern
        if len(dot_contours) >= self.min_pattern_dots:
            self.duckiebot_detected = True
            
            # Calculate bounding box of the pattern
            all_dots = np.vstack([dot_contours[i] for i in range(len(dot_contours))])
            x, y, w, h = cv2.boundingRect(all_dots)
            
            # Draw the bounding box
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Estimate distance based on the size of the pattern
            # The actual formula would depend on camera calibration
            pattern_width_in_meters = 0.1  # Approximate width in meters
            focal_length = 350  # This is approximate and would need calibration
            
            # Distance = (object_width_in_meters * focal_length) / width_in_pixels
            self.distance_to_duckiebot = (pattern_width_in_meters * focal_length) / w
            
            # Display the estimated distance
            distance_text = f"Dist: {self.distance_to_duckiebot:.2f}m"
            cv2.putText(debug_img, distance_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            self.duckiebot_detected = False
            self.distance_to_duckiebot = float('inf')
            cv2.putText(debug_img, "No Duckiebot detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Publish debug image
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        self.pub_debug_img.publish(debug_msg)

    def process_red_line_detection(self, img):
        """Process the image to detect red lines on the path"""
        # Use a lower portion of the image to detect red lines closer to the robot
        # Adjust this value to look for red lines closer to the robot - higher value = closer
        crop_img = img[350:-1, :, :]  # Increased from 300 to 350 to look lower in the frame
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color (two ranges needed for red in HSV)
        mask1 = cv2.inRange(hsv, RED_MASK[0], RED_MASK[1])   # Lower red range
        mask2 = cv2.inRange(hsv, RED_MASK2[0], RED_MASK2[1]) # Upper red range
        red_mask = cv2.bitwise_or(mask1, mask2)              # Combine masks
        
        # Apply morphological operations to improve detection
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        
        # Find contours in the red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a debug image
        debug_img = crop_img.copy()
        
        # Check for red line
        self.red_line_detected = False
        max_red_area = 0
        max_red_idx = -1
        
        # Get the height of the cropped image to check position
        height = crop_img.shape[0]
        closest_y = height  # Initialize to the bottom of the image
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_red_area:
                max_red_area = area
                max_red_idx = i
            
            # Find the lowest point (closest to robot) in the contour
            if area > self.red_line_area_threshold / 2:  # Use lower threshold for position check
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cy = int(M["m01"] / M["m00"])
                    if cy < closest_y:
                        closest_y = cy
        
        # If we found a significant red contour, consider it a line
        if max_red_area > self.red_line_area_threshold:
            # Check if the red line is in the lower portion of the frame (closer to the robot)
            # The smaller this divisor, the closer the robot will be to the line when detecting
            position_threshold = height / 1.5  # Adjust this value to control detection distance
            
            if closest_y > position_threshold:
                self.red_line_detected = True
                cv2.putText(debug_img, "CLOSE ENOUGH TO STOP", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(debug_img, "RED LINE TOO FAR", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
            # Draw the largest red contour for debugging
            cv2.drawContours(debug_img, contours, max_red_idx, (0, 255, 255), 3)
            
            # Add text showing detection
            cv2.putText(debug_img, f"RED LINE DETECTED! Count: {self.red_line_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show position information
            cv2.putText(debug_img, f"Closest point: {closest_y}/{height}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Add text showing no detection
            cv2.putText(debug_img, f"No red line. Count: {self.red_line_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add text showing max area
        cv2.putText(debug_img, f"Max red area: {max_red_area}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                   
        # Add text showing cooldown
        if self.red_line_cooldown > 0:
            cv2.putText(debug_img, f"Cooldown: {self.red_line_cooldown}s", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Publish debug image for red line detection
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        self.pub_red_line_img.publish(debug_msg)

    def adjust_velocity_based_on_detection(self):
        """Adjust velocity based on duckiebot detection"""
        # Default to base velocity
        target_velocity = self.base_velocity
        
        # If we detect the Duckiebot pattern, adjust velocity
        if self.duckiebot_detected:
            if self.distance_to_duckiebot > self.safe_distance:
                # Far enough to maintain normal following speed
                target_velocity = self.following_velocity
            elif self.distance_to_duckiebot > self.min_safe_distance:
                # Getting close, slow down
                target_velocity = self.slowing_velocity
            else:
                # Too close, stop
                target_velocity = 0.0
                
        # If TOF sensor detects something very close, override with stop
        if self.tof_distance is not None and self.tof_distance < self.min_safe_distance:
            target_velocity = 0.0
            
        # Update current velocity (can add smoothing here if needed)
        self.current_velocity = target_velocity
        
        return self.current_velocity

    def update_leds(self):
        """Update LED colors based on current state"""
        # Check if following state changed
        currently_following = self.duckiebot_detected and self.distance_to_duckiebot < 1.0
        
        # Check for red line stop state
        if self.at_red_line:
            # At red line - use red LED
            if self.current_led_state != "red":
                self.set_led_color("red")
                rospy.loginfo("Stopped at red line - LEDs set to red")
            return
            
        # Normal operation LEDs
        if currently_following != self.was_following:
            # State changed, update LEDs
            if currently_following:
                # In following mode - use the special 'following' pattern
                self.set_led_color("following")
                rospy.loginfo("Following mode activated - LEDs set to following pattern")
            else:
                # Not following - use searching pattern
                self.set_led_color("searching")
                rospy.loginfo("Following mode deactivated - LEDs set to searching pattern")
            
            # Update state tracking
            self.was_following = currently_following
            
        # Special case - if we're following and too close (unsafe distance)
        elif currently_following and self.distance_to_duckiebot < self.min_safe_distance:
            # Too close! Warning pattern
            self.set_led_color("warning")
            rospy.logwarn("Too close to Duckiebot - LEDs set to warning pattern")

    def drive(self):
        """Main control function that integrates lane following with duckiebot following"""
        # Decrement red line cooldown if active
        if self.red_line_cooldown > 0:
            self.red_line_cooldown -= 1/8.0  # Assuming 8Hz loop rate
        
        # Check for red line detection and handle it
        if self.red_line_detected and not self.at_red_line and self.red_line_cooldown <= 0:
            self.at_red_line = True
            self.red_line_count += 1
            rospy.loginfo(f"RED LINE DETECTED! Stopping for {self.red_line_stop_time} seconds. Count: {self.red_line_count}")
            
            # Stop the robot
            self.stop_at_red_line()
            
            # Set cooldown to avoid detecting the same line multiple times
            self.red_line_cooldown = self.red_line_cooldown_time
            
            # Reset flag after handling
            self.red_line_detected = False
            self.at_red_line = False
            return
            
        # First handle any safety stops
        if self.obj_stop:
            self.stop(8)
            self.obj_stop = False
            self.logwarn("About to stop")
            self.set_led_color("yellow")
            rospy.sleep(1.0)
            self.logwarn("Stopped for one second")
            return

        self.set_led_color("green")  # Reset LED color to green
        # Adjust velocity based on duckiebot detection
        self.current_velocity = self.adjust_velocity_based_on_detection()
        
        # Update LED colors based on state
        self.set_led_color("green")  # Reset LED color to green
        # self.update_leds()
        
        # Handle lane following
        if self.proportional is None:
            self.twist.omega = 0
            self.last_error = 0
        else:
            # P Term
            P = -self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            # I term
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            if dt <= 0:
                I = 0
            else:
                self.integral += self.proportional * dt
                I = self.I * self.integral
            
            # Set velocity based on detection and steering based on lane following
            self.twist.v = self.current_velocity
            self.twist.omega = P + D + I
            
            # if DEBUG:
            #     self.loginfo(f"P: {P}, D: {D}, I: {I}, omega: {self.twist.omega}, v: {self.twist.v}")
            #     if self.duckiebot_detected:
            #         self.loginfo(f"Duckiebot detected at distance: {self.distance_to_duckiebot:.2f}m")

        # Publish the command
        self.vel_pub.publish(self.twist)
        
    def stop_at_red_line(self):
        """Stop at red line and check for blue on left or right side"""
        # Set LED to red
        self.set_led_color("red")
        
        # Stop the robot
        self.twist.v = 0
        self.twist.omega = 0
        
        rospy.loginfo(f"Stopped at red line. Count: {self.red_line_count}")
        
        # Publish the stop command initially
        self.vel_pub.publish(self.twist)
        
        # Wait a bit for the robot to completely stop
        rospy.sleep(0.5)
        
        # If this is the first time seeing a red line (count = 1), check for blue
        if self.red_line_count == 1:
            rospy.loginfo("First red line detected. Looking for blue on left or right...")
            
            # Check multiple frames for robustness
            left_votes = 0
            right_votes = 0
            frames_to_check = 5
            
            for i in range(frames_to_check):
                if self.last_image is not None:
                    blue_on_left, blue_on_right = self.detect_blue(self.last_image)
                    if blue_on_left:
                        left_votes += 1
                    if blue_on_right:
                        right_votes += 1
                # Short sleep between checks
                rospy.sleep(0.2)
            
            # Make decision based on majority
            if left_votes > right_votes:
                rospy.loginfo("Blue detected on LEFT side! Turning LEFT.")
                self.turn_left()
            else:
                rospy.loginfo("Blue detected on RIGHT side or no blue detected! Turning RIGHT.")
                self.turn_right()
        else:
            # For other red lines, just wait
            rospy.loginfo(f"Red line {self.red_line_count} detected. Waiting {self.red_line_stop_time} seconds.")
            # Keep publishing stop command for the duration
            start_time = rospy.get_time()
            while rospy.get_time() - start_time < self.red_line_stop_time:
                self.vel_pub.publish(self.twist)
                rospy.sleep(0.1)
        
        rospy.loginfo("Resuming after red line stop")
        
        # Return to previous LED state
        if self.was_following:
            self.set_led_color("following")
        else:
            self.set_led_color("searching")

    def stop(self, duration):
        """Stop the robot for a given duration"""
        self.twist.v = 0
        self.twist.omega = 0
        for i in range(duration):
            self.vel_pub.publish(self.twist)

    def hook(self):
        """Shutdown hook"""
        print("SHUTTING DOWN")
        # Display final stats
        rospy.loginfo(f"Final red line count: {self.red_line_count}")
        
        # Turn off LEDs
        self.set_led_color("off")  
        
        # Stop motors
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowWithDetectionNode("lane_follow_with_detection_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()