#!/usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32, String, Int32, Bool
from turbojpeg import TurboJPEG
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from dt_apriltags import Detector

from led_service.srv import SetLEDColor, SetLEDColorRequest

# Color masks for detection
ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
RED_MASK = [(0, 140, 100), (10, 255, 255)]  # Lower red HSV range
RED_MASK2 = [(170, 140, 100), (180, 255, 255)]  # Upper red HSV range (wrapped around)
BLUE_MASK = [(90, 100, 50), (130, 255, 200)]  # Adjusted for duckiebot blue
DEBUG = True
ENGLISH = True
SAFETY = True

class NavigationNode(DTROS):

    def __init__(self, node_name):
        super(NavigationNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']

        # ----------- LED SERVICE -----------
        rospy.loginfo("Initializing LED service connection...")
        self.led_service = None
        try:
            led_service_name = "set_led_color"
            rospy.wait_for_service(led_service_name, timeout=2.0)
            self.led_service = rospy.ServiceProxy(led_service_name, SetLEDColor)
            rospy.loginfo("Connected to LED service")
        except rospy.ROSException:
            rospy.logwarn("LED service not available, will continue without LED indicators")

        # ----------- SUBSCRIBERS/PUBLISHERS -----------
        # Subscribe to camera info and TOF sensor
        self.tof_sub = rospy.Subscriber("/" + self.veh + "/front_center_tof_driver_node/range",
                                        Range,
                                        self.cb_tof,
                                        queue_size=1)

        # Publish mask image
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
                                   
        # Subscribe to camera
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
                                    
        # Publish velocity commands
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)
        
        # # Debug image publishers
        # self.pub_debug_img = rospy.Publisher("/" + self.veh + "/detection_node/debug/compressed",
        #                                     CompressedImage,
        #                                     queue_size=1)
        
        # self.pub_red_line_img = rospy.Publisher("/" + self.veh + "/red_line_node/debug/compressed",
        #                                        CompressedImage,
        #                                        queue_size=1)
        
        # self.pub_blue_detect_img = rospy.Publisher("/" + self.veh + "/blue_detection_node/debug/compressed",
        #                                         CompressedImage,
        #                                         queue_size=1)

        # Communication with next node
        # self.task_complete_pub = rospy.Publisher("/" + self.veh + "/navigation_complete", Bool, queue_size=1)

        # self.jpeg = TurboJPEG()

        # PID Variables for lane following
        self.proportional = None
        if ENGLISH:
            self.offset = 190
        else:
            self.offset = 220
            
        # Base velocity and control parameters
        self.base_velocity = 0.3
        self.current_velocity = self.base_velocity
        self.twist = Twist2DStamped(v=self.current_velocity, omega=0)

        # PID parameters
        self.P = 0.025
        self.D = -0.0025
        self.I = 0

        self.last_error = 0
        self.integral = 0
        self.last_time = rospy.get_time()
        
        # Safety parameters
        self.tof_distance = 1.0
        self.obj_stop = False
        
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
        
        # Turn parameters
        self.turn_duration = 1.5  # How long to execute the turn (seconds)
        self.left_turn_omega = 3  # Angular velocity for left turn
        self.right_turn_omega = -2.5  # Angular velocity for right turn
        self.first_turn_direction = None
        
        # April Tag detection parameters
        self.tag_detector = Detector(families='tag36h11',
                                nthreads=1,
                                quad_decimate=1.0,
                                quad_sigma=0.0,
                                refine_edges=1,
                                decode_sharpening=0.25,
                                debug=0)
        
        self.detect_april_tags = False  # Flag to enable/disable April Tag detection
        self.camera_params = [305.57, 308.83, 303.02, 231.14]  # fx, fy, cx, cy - calibrate these for your camera
        self.tag_size = 0.065  # Tag size in meters
        self.detected_tag = 0  # Store the detected April Tag ID
        
        # Task completion flag
        self.is_completed = False

        # Start with green LED
        self.set_led_color("green")

        # Shutdown hook
        rospy.on_shutdown(self.hook)
        
        rospy.loginfo("Navigation Node initialized. Ready to start.")

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
            # Normal behavior - stop for obstacles
            self.obj_stop = True

    def callback(self, msg):
        """Main image processing callback"""
        if self.is_completed:
            return

        img = self.jpeg.decode(msg.data)
        
        # Process lane following
        self.process_lane_following(img)
        
        # Process red line detection
        if not self.at_red_line and self.red_line_cooldown <= 0:  
            self.process_red_line_detection(img)
        
        # Process April Tag detection if enabled
        if self.detect_april_tags:
            tag_id, _ = self.detect_april_tag(img)
            if tag_id is not None:
                rospy.loginfo(f"April Tag detected while driving: {tag_id}")
                self.detected_tag = tag_id
                self.detect_april_tags = False

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
        
        # # Publish debug image
        # debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        # self.pub_blue_detect_img.publish(debug_msg)
        
        return blue_on_left, blue_on_right

    def turn_left(self):
        """Execute a left turn with a larger, smoother curve like in the map"""
        rospy.loginfo("Executing left curve turn")
        
        # Temporarily disable lane following by setting proportional to None
        saved_proportional = self.proportional
        self.proportional = None
        
        # Set up turn command for a larger curve
        turn_cmd = Twist2DStamped()
        
        # Use higher forward velocity and moderate turning to create a larger curve
        turn_cmd.v = 0.35  # Higher forward velocity for a wider curve
        turn_cmd.omega = 1.5  # Moderate but consistent turning rate
        
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
        
        # Temporarily disable lane following
        saved_proportional = self.proportional
        self.proportional = None
        
        # Set up turn command for a sharper curve
        turn_cmd = Twist2DStamped()
        
        # Less forward velocity and stronger turning for a tighter curve
        turn_cmd.v = 0.3  # Lower forward velocity for a tighter turn
        turn_cmd.omega = -5.5  # Stronger turning rate for a sharper curve
        
        # Execute the turn for a shorter duration
        turn_duration = 1.5  # Shorter duration for a sharper curve
        
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < turn_duration:
            self.vel_pub.publish(turn_cmd)
            rospy.sleep(0.05)
        
        # Restore lane following
        self.proportional = saved_proportional
        
        rospy.loginfo("Right curve turn completed")

    def detect_april_tag(self, img):
        """Detect April tags in the image"""
        if not self.detect_april_tags:
            return None, None
            
        # Convert to grayscale for April Tag detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect April tags
        tags = self.tag_detector.detect(gray, estimate_tag_pose=False, 
                                    camera_params=self.camera_params, 
                                    tag_size=self.tag_size)
        
        # Create debug image
        debug_img = img.copy()
        
        tag_id = None
        if len(tags) > 0:
            # Get the tag with highest decision margin (confidence)
            best_tag = max(tags, key=lambda x: x.decision_margin)
            tag_id = best_tag.tag_id
            
            # Draw tag outline
            for idx in range(4):
                cv2.line(debug_img, 
                        tuple(best_tag.corners[idx-1, :].astype(int)), 
                        tuple(best_tag.corners[idx, :].astype(int)),
                        (0, 255, 0), 2)
            
            # Put tag ID on image
            center = best_tag.center.astype(int)
            cv2.putText(debug_img, f"Tag ID: {tag_id}", (center[0], center[1]+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            rospy.loginfo(f"Detected April Tag ID: {tag_id}")
        else:
            cv2.putText(debug_img, "No April Tag detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Publish debug image
        # debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        # self.pub_debug_img.publish(debug_msg)
        
        return tag_id, debug_img

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
                    self.proportional = cx - int(crop_width / 2) - 320
                    if DEBUG:
                        cv2.drawContours(crop, white_contours, max_white_idx, (255, 0, 0), 3)
                        cv2.circle(crop, (cx, cy), 7, (255, 0, 0), -1)
                except:
                    pass
            else:
                # No line found
                turn_cmd = Twist2DStamped()
                turn_cmd.v = 0.3
                turn_cmd.omega = 2

        # Publish debug image if needed
        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def process_red_line_detection(self, img):
        """Process the image to detect red lines on the path"""
        # Use a lower portion of the image to detect red lines closer to the robot
        crop_img = img[350:-1, :, :]
        
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
            position_threshold = height / 1.5
            
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
        # debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        # self.pub_red_line_img.publish(debug_msg)

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
                self.first_turn_direction = "left"
                self.turn_left()
            else:
                rospy.loginfo("Blue detected on RIGHT side or no blue detected! Turning RIGHT.")
                self.first_turn_direction = "right"
                self.turn_right()

        elif self.red_line_count == 2:
            rospy.loginfo("Second red line detected. Moving straight for 2 seconds...")
            
            # Temporarily disable lane following
            saved_proportional = self.proportional
            self.proportional = None
            
            # Set command for moving straight
            straight_cmd = Twist2DStamped()
            straight_cmd.v = 0.3  # Forward velocity
            straight_cmd.omega = 0  # No turning
            
            # Move straight for 2 seconds
            straight_duration = 3.5
            start_time = rospy.get_time()
            while rospy.get_time() - start_time < straight_duration:
                self.vel_pub.publish(straight_cmd)
                rospy.sleep(0.05)
            
            # Restore lane following
            self.proportional = saved_proportional
            rospy.loginfo("Straight movement completed, resuming lane following")
            
        elif self.red_line_count == 3:
            rospy.loginfo("Third red line detected. Making opposite turn from first red line...")
            
            # Make the opposite turn from what was done at the first red line
            if self.first_turn_direction == "right":
                rospy.loginfo("First turn was RIGHT, now turning LEFT")
                self.turn_left()
            else:
                rospy.loginfo("First turn was LEFT, now turning RIGHT")
                self.turn_right()
                
            self.red_line_cooldown = self.red_line_cooldown_time / 4
            rospy.loginfo(f"Reduced cooldown time to {self.red_line_cooldown} seconds for faster next line detection")
            
            # Enable April tag detection after the third red line
            self.detect_april_tags = True
            rospy.loginfo("April Tag detection enabled for next red line")
            
            # We've completed navigation node's tasks after the 3rd red line
            # Signal completion to start the next node
            self.set_led_color("blue")
            rospy.loginfo("Navigation node task complete after 3rd red line! Signaling to start crosswalk node.")
            self.is_completed = True
            complete_msg = Bool()
            complete_msg.data = True
            self.task_complete_pub.publish(complete_msg)
            
        else:
            # For other red lines, just wait
            rospy.loginfo(f"Red line {self.red_line_count} detected. Waiting {self.red_line_stop_time} seconds.")
            # Keep publishing stop command for the duration
            start_time = rospy.get_time()
            while rospy.get_time() - start_time < self.red_line_stop_time:
                self.vel_pub.publish(self.twist)
                rospy.sleep(0.1)
        
        rospy.loginfo("Resuming after red line stop")

    def stop(self, duration):
        """Stop the robot for a given duration"""
        self.twist.v = 0
        self.twist.omega = 0
        for i in range(duration):
            self.vel_pub.publish(self.twist)
            rospy.sleep(1.0)

    def drive(self):
        """Main control function for lane following with red line detection"""
        # Skip if navigation is complete
        if self.is_completed:
            return

        # Decrement red line cooldown if active
        if self.red_line_cooldown > 0:
            self.red_line_cooldown -= 1/8.0  # Assuming 8Hz loop rate
        
        # Handle safety stop
        if self.