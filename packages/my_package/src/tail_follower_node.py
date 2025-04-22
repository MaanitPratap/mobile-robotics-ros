#!/usr/bin/env python3

import rospy
import os
import subprocess

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage, Range
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from dt_apriltags import Detector

from led_service.srv import SetLEDColor, SetLEDColorRequest

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
RED_MASK = [(0, 140, 100), (10, 255, 255)]  # Lower red HSV range
RED_MASK2 = [(170, 140, 100), (180, 255, 255)]  # Upper red HSV range (wrapped around)
BLUE_MASK = [(90, 100, 50), (130, 255, 200)]  # Adjusted for duckiebot blue
DUCK_MASK = [(20, 100, 100), (30, 255, 255)]  # Yellow HSV range for duck detection
DEBUG = True
ENGLISH = True
SAFETY = True
AUSSIE = False


import argparse

# Add command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Lane following with parking capabilities')
    parser.add_argument('--stall', type=int, default=3, choices=[1, 2, 3, 4],
                        help='Parking stall number (1-4)')
    args = parser.parse_args()
    return args


class LaneFollowWithDetectionNode(DTROS):

    def __init__(self, node_name, args=None):
        super(LaneFollowWithDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
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

        # Subscribe to camera info
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
        
        # Debug publisher for blue detection
        self.pub_blue_line_detect_img = rospy.Publisher("/" + self.veh + "/blue_line_detection_node/debug/compressed",
                                                CompressedImage,
                                                queue_size=1)
        
        self.pub_duck_detect_img = rospy.Publisher("/" + self.veh + "/duck_detection_node/debug/compressed",
                                         CompressedImage,
                                         queue_size=1)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized Combined Lane Following and Detection Node")

        # PID Variables for lane following
        self.proportional = None
        if ENGLISH:
            self.offset = 210
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
        self.red_line_count = 5  # Counter for red lines
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

        self.first_turn_direction = None
        
        # Dot pattern detection parameters
        self.min_dot_area = 20  # Minimum area of a dot to be detected
        self.min_pattern_dots = 5  # Minimum number of dots to consider a valid pattern

        self.done_man = False  # Flag to indicate if we have completed the maneuver


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

        # self.start_blue_detection = False  # Flag to start blue detection
        # Duck detection parameters
        self.duck_detected = False
        self.duck_detected_false_counter = 0  # Counter for false detections
        self.duck_area_threshold = 300  # Minimum area to consider a valid duck
        self.duck_detection_enabled = False  # Only enable after fifth red line
        self.at_crosswalk = False  # Flag to indicate we're at a crosswalk
        self.crosswalk_april_tags = [48, 50]  # April tags that mark crosswalks
        self.crosswalk_wait_time = 0  # Counter for waiting at crosswalk
        self.crosswalk_max_wait = 80  # Maximum wait time (approximately 10 seconds at 8Hz)



        self.blue_line_detection_enabled = False
        self.blue_crosswalk_detected = False
          # Flag to control when to detect blue lines
        self.blue_crosswalk_count = 0  # Counter for blue crosswalks
        self.blue_line_area_threshold = 350  # Minimum area of blue contour to be considered a line


        self.maneuvering = False
        self.maneuver_state = 0
        self.state_time = 0
        self.broken_bot_detected = False
        self.passed_first_crosswalk = False

        self.crosswalk_timeout = 0
        self.crosswalk_timeout_duration = 40  # 5 seconds at 8Hz


        # Get the stall parameter from command line arguments or use default
        if args and hasattr(args, 'stall'):
            self.stall = args.stall
        else:
            # Get from ROS parameter server or environment variable if available
            self.stall = rospy.get_param('~stall', int(os.environ.get('PARKING_STALL', 3)))
            
        rospy.loginfo(f"Initialized with parking stall {self.stall}")

        # Parking state variables
        self.parking_enabled = False
        self.parking_state = 0
        self.parking_state_start_time = rospy.get_time()
        self.stall = 3  # Default stall number, can be changed
        
        # AprilTag detection for parking (reuse from the direct_parking_node)
        # This is already defined in the current code, but make sure to add parking-specific variables
        self.tag_detected = False
        self.tag_x_center = 0
        self.tag_id = -1
        self.target_tag_id = -1
        self.has_target_tag = False
        self.tag_detection_count = 0
        self.tag_lost_count = 0
        self.last_valid_tag_center = 0
        self.tag_alignment_count = 0
        
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
        self.crop_width = 320  # Reduced width of cropped region in pixels

        
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
            if self.red_line_count >= 5:
                # If we've passed first crosswalk, this might be the broken duckiebot
                if self.passed_first_crosswalk and not self.at_crosswalk:
                    rospy.loginfo(f"Potential broken duckiebot detected at distance: {self.tof_distance}m")
                    self.obj_stop = True

                else:
                    # After 5th line but before/at first crosswalk - ignore normal obstacle stops
                    rospy.loginfo(f"Obstacle detected but ignoring (after 5th line): {self.tof_distance}m")
                    self.obj_stop = False
            else:
                # Normal behavior before 5th line - stop for obstacles
                self.obj_stop = True


    def callback(self, msg):
        """Main image processing callback - handles lane following, duckiebot detection, red line detection, and april tag detection"""
        img = self.jpeg.decode(msg.data)
        self.last_image = img.copy()


        # If in parking mode, process for parking
        if self.parking_enabled:
            self.process_parking_apriltag(img)
            return
        
        # Process lane following
        self.process_lane_following(img)
        
        # Process duckiebot detection in the upper part of the image
        self.process_duckiebot_detection(img)
        
        # Process red line detection in the lower part of the image
        if not self.at_red_line and self.red_line_cooldown <= 0:  # Only check if not already at a red line
            self.process_red_line_detection(img)
        
        # Process blue detection
        # self.detect_blue(img)

        # self.process_blue_crosswalk_detection(img)
        # Process April Tag detection if enabled
        if self.detect_april_tags:
            tag_id, _ = self.detect_april_tag(img)
            if tag_id is not None:
                rospy.loginfo(f"April Tag detected while driving: {tag_id}")
                self.detected_tag = tag_id
                self.detect_april_tags = False


        # rospy.loginfo(f"Blue crosswalk detected before enable: {self.blue_crosswalk_detected}")
        # Process blue crosswalk detection if enabled
        if self.blue_line_detection_enabled and not self.at_crosswalk:
            self.blue_crosswalk_detected = self.process_blue_crosswalk_detection(img)
            rospy.loginfo(f"Blue crosswalk detected: {self.blue_crosswalk_detected}")
            if self.blue_crosswalk_detected:
                rospy.loginfo("Blue crosswalk detected!")
                
                self.crosswalk_wait_time = 0
                self.blue_crosswalk_count += 1

                rospy.loginfo("Enabling duck detection and at Crosswalk")
                self.at_crosswalk = True
                self.duck_detection_enabled = True
                self.blue_line_detection_enabled = False
                rospy.loginfo("Disabling blue line detection")


        # Add this code to the callback method, replacing the equivalent section:

        if self.duck_detection_enabled and self.at_crosswalk:
            # Check for duck detection
            self.duck_detected = self.detect_duck(img)
            if self.duck_detected:
                # Just set LED here but don't call stop() - we'll handle stopping in the drive() method
                self.set_led_color("purple")
                rospy.loginfo(f"Duck detected in callback: {self.duck_detected}, counter: {self.duck_detected_false_counter}")
            else:
                rospy.loginfo(f"No duck detected in callback, counter: {self.duck_detected_false_counter}/80")


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
        turn_cmd.omega = 1.5 # Moderate but consistent turning rate
        
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
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
        self.pub_debug_img.publish(debug_msg)
        
        return tag_id, debug_img
    

    def process_blue_crosswalk_detection(self, img):
        """Process the image to detect blue crosswalk lines on the path"""
        # Use a lower portion of the image to detect blue lines closer to the robot
        crop_img = img[280:-1, :, :]  # Same as red line detection
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, BLUE_MASK[0], BLUE_MASK[1])
        
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
            if area > self.blue_line_area_threshold / 2:  # Use same threshold as red line for position check
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cy = int(M["m01"] / M["m00"])
                    if cy < closest_y:
                        closest_y = cy
        
        # If we found a significant blue contour, consider it a crosswalk line
        if max_blue_area > self.blue_line_area_threshold:  # Use same threshold as red line
            # Check if the blue line is in the lower portion of the frame (closer to the robot)
            position_threshold = height / 2.5  # Same as red line detection
            
            if closest_y > position_threshold:
                self.blue_crosswalk_detected = True
                self.twist.v = 0
                self.twist.omega = 0
                self.vel_pub.publish(self.twist)

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
        self.pub_blue_line_detect_img.publish(debug_msg)
        
        return self.blue_crosswalk_detected
    

    def detect_duck(self, img):
        """Detect if there is a duck (yellow object) in the road, with improvements to avoid detecting lane markings"""
        # 1. Focus on the center of the road rather than the edges where lane markings appear
        height, width = img.shape[:2]
        # Create a region of interest that excludes the sides of the image where lane markings usually appear
        # Focus on the middle 50% of the width and a specific height range
        roi_x_start = int(width * 0.55)  # Left 45% excluded
        roi_x_end = int(width * 0.95)    # Right 25% excluded
        roi_y_start = 150                # Top of region - adjust based on where ducks appear
        roi_y_end = 350                  # Bottom of region
        
        # Extract the region of interest
        crop_img = img[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        
        # 2. Convert to HSV for color detection
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        
        # 3. Refine the yellow color range to better match the duck and exclude lane markings
        # Adjust these values based on testing - this is more saturated yellow for the duck
        duck_lower = (22, 120, 120)  # Higher saturation and value than lane markings
        duck_upper = (30, 255, 255)
        
        # Create mask for yellow color (duck)
        duck_mask = cv2.inRange(hsv, duck_lower, duck_upper)
        
        # 4. Apply morphological operations to improve detection
        kernel = np.ones((5, 5), np.uint8)
        # Opening operation to remove small noise
        duck_mask = cv2.morphologyEx(duck_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Dilation to connect nearby parts
        duck_mask = cv2.dilate(duck_mask, kernel, iterations=2)
        
        # 5. Find contours in the duck mask
        contours, _ = cv2.findContours(duck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a debug image
        debug_img = crop_img.copy()
        
        # 6. Check for duck with improved filtering
        duck_currently_visible = False
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
                
        # 7. Check if we have valid duck contours
        if valid_duck_contours:
            # Find the largest valid contour
            largest_contour = max(valid_duck_contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            duck_currently_visible = True
            self.duck_detected_false_counter = 0  # Reset counter when duck is seen
            
            # Draw the contour on debug image
            cv2.drawContours(debug_img, [largest_contour], -1, (0, 255, 255), 3)
            
            # Add text showing detection
            cv2.putText(debug_img, f"DUCK DETECTED! Area: {largest_area}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Update the duck detection state based on current frame and counter
        if duck_currently_visible:
            self.duck_detected = True
        elif not duck_currently_visible:
            # Only increment counter if duck isn't visible this frame
            self.duck_detected_false_counter += 1
            
            # Only set duck_detected to False after many consecutive misses
            if self.duck_detected_false_counter >= 20:  # Require more consecutive misses (8Hz x 2.5s = 20)
                self.duck_detected = False
                
            # Add text showing no detection
            cv2.putText(debug_img, f"No duck detected (counter: {self.duck_detected_false_counter}/20)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw ROI boundaries on the original image for debugging
        full_debug = img.copy()
        cv2.rectangle(full_debug, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        
        if self.duck_detected:
            cv2.putText(full_debug, "DUCK DETECTED", (width//2 - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Publish debug image for duck detection
        debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(full_debug))
        self.pub_duck_detect_img.publish(debug_msg)
        
        return self.duck_detected
    
    def maneuver_around_bot(self):
        """Execute a maneuver to go around a broken duckiebot"""
        rospy.loginfo(f"MANEUVERING... State: {self.maneuver_state}, Time: {self.state_time}")
        
        # Increment state time counter
        self.state_time += 1
        
        # Define maneuver parameters
        turn_angle = 4.0     # Angular velocity for turns
        turn_time = 10       # Duration of turns (in cycles)
        straight_time = 20   # Duration of straight segments (in cycles)
        wait_time = 5        # Initial wait time
        
        # Create command message
        cmd = Twist2DStamped()
        
        # State machine for the maneuver sequence
        if self.state_time < wait_time:
            # Initial pause
            cmd.v = 0
            cmd.omega = 0
        elif self.maneuver_state == 0:
            # Wait for a moment
            if self.state_time > 15:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Initial waiting completed")
            cmd.v = 0
            cmd.omega = 0
        elif self.maneuver_state == 1:
            # Turn left to face other lane
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Left turn completed")
            cmd.v = 0.25
            cmd.omega = turn_angle
        elif self.maneuver_state == 2:
            # Drive straight into other lane
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Drive to other lane completed")
            cmd.v = 0.2
            cmd.omega = 0
        elif self.maneuver_state == 3:
            # Turn right to drive past the obstacle
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Right turn completed")
            cmd.v = 0.2
            cmd.omega = -turn_angle
        elif self.maneuver_state == 4:
            # Drive straight past the obstacle
            if self.state_time > straight_time * 1.5:  # Drive longer to pass completely
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Drive past obstacle completed")
            cmd.v = 0.2
            cmd.omega = 0
        elif self.maneuver_state == 5:
            # Turn right to face back toward original lane
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Right turn to original lane completed")
            cmd.v = 0.2
            cmd.omega = -turn_angle
        elif self.maneuver_state == 6:
            # Drive straight into original lane
            if self.state_time > straight_time:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Return to original lane completed")
            cmd.v = 0.2
            cmd.omega = 0
        elif self.maneuver_state == 7:
            # Turn left to face forward
            if self.state_time > turn_time:
                self.maneuver_state += 1
                self.state_time = 0
                rospy.loginfo("Maneuver: Final alignment completed")
            cmd.v = 0.2
            cmd.omega = turn_angle
        else:
            # End maneuver
            self.maneuvering = False
            self.done_man = True
            self.broken_bot_detected = False
            self.state_time = 0
            self.maneuver_state = 0
            cmd.v = self.base_velocity
            cmd.omega = 0
            self.blue_line_detection_enabled = True
            rospy.loginfo("Maneuver sequence completed")
        
        # Publish the command
        self.vel_pub.publish(cmd)
        
        # Return true while still maneuvering
        return True
    
    def process_parking_apriltag(self, img):
        """Process camera images for AprilTag detection during parking"""
        if not self.parking_enabled:
            return
            
        try:
            # Calculate crop region based on stall position
            # For stalls 1 and 2 (right side), crop left side of image 
            # For stalls 3 and 4 (left side), crop right side of image
            self.image_width = img.shape[1]
            image_width = img.shape[1]
            image_height = img.shape[0]
            image_center_x = image_width // 2
            
            if self.stall == 1 or self.stall == 3:
                # For right stalls, take a narrow region on the right side
                crop_start = 0
                crop_end = min(self.crop_width, image_width)
            else:
                # For left stalls, take a narrow region on the left side
                crop_start = max(0, image_width - self.crop_width)
                crop_end = image_width
                
            # Crop image to only see the relevant part
            cropped_image = img[:, crop_start:crop_end]
            
            # Convert to grayscale for AprilTag detection
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            
            # Create a debug image (copy of original)
            debug_image = img.copy()
            
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
                    rospy.loginfo(f"*** PARKING: LOCKING ONTO TARGET TAG ID: {self.target_tag_id} ***")
            
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
                rospy.loginfo_throttle(1.0, f"Parking: Target AprilTag {tag.tag_id} detected at x={self.tag_x_center} (count={self.tag_detection_count})")
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
            image_center = image_width // 2
            cv2.line(debug_image, (image_center, 0), (image_center, debug_image.shape[0]), 
                    (255, 0, 0), 1)
            
            # Add current state and ToF distance
            cv2.putText(debug_image, f"Parking State: {self.parking_state}", 
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
            self.pub_debug_img.publish(debug_msg)
                
        except Exception as e:
            rospy.logerr(f"Error processing image for parking: {e}")
    
    # Add the parking method
    def park(self):
        """Execute the direct parking state machine"""
        # Skip if not in parking mode
        if not self.parking_enabled:
            return False
            
        current_time = rospy.get_time()
        elapsed = current_time - self.parking_state_start_time
        
        # Create command message
        cmd = Twist2DStamped()
        
        # Emergency TOF check for all states except state 3 (already stopped)
        if self.parking_state != 3 and self.tof_distance < self.emergency_stop_distance:
            self.parking_state = 3
            self.parking_state_start_time = current_time
            rospy.logwarn(f"PARKING: EMERGENCY STOP! TOF distance: {self.tof_distance:.2f}m")
            # Send stop command immediately
            cmd.v = 0
            cmd.omega = 0
            self.vel_pub.publish(cmd)
            return True
        
        # State machine for direct parking
        if self.parking_state == 0:
            # Initial state - drive forward a specified distance based on stall number
            if self.stall == 1 or self.stall == 3:
                drive_time = self.short_drive_time
                rospy.loginfo_throttle(1.0, f"Parking State 0: Moving forward short distance for stall {self.stall}")
            else:
                drive_time = self.long_drive_time
                rospy.loginfo_throttle(1.0, f"Parking State 0: Moving forward longer distance for stall {self.stall}")
                
            cmd.v = self.drive_speed
            cmd.omega = 0
            
            if elapsed > drive_time:
                self.parking_state = 1
                self.parking_state_start_time = current_time
                rospy.loginfo("Parking: Moving to state 1: Turn toward parking stall")
                
        elif self.parking_state == 1:
            # Turn 90 degrees in the appropriate direction
            if self.stall == 1 or self.stall == 2:
                # Right turn for stalls 1 and 2
                cmd.v = 0.1
                cmd.omega = -self.turn_speed
                rospy.loginfo_throttle(1.0, "Parking: Turning right 90 degrees")
            else:
                # Left turn for stalls 3 and 4
                cmd.v = 0.1
                cmd.omega = self.turn_speed
                rospy.loginfo_throttle(1.0, "Parking: Turning left 90 degrees")
            
            if elapsed > self.turn_time:
                self.parking_state = 1.5  # New state for alignment
                self.parking_state_start_time = current_time
                rospy.loginfo("Parking: Moving to state 1.5: Searching for AprilTag")
                # Reset tag tracking when starting search
                if not self.has_target_tag:
                    self.tag_detected = False
                    self.tag_detection_count = 0
                    self.tag_lost_count = 0
                
        elif self.parking_state == 1.5:
            # New state: Move forward until AprilTag is detected
            if not self.tag_detected:
                # Move forward slowly until tag is detected
                cmd.v = 0.2  # Slow forward movement
                cmd.omega = 0  # Keep straight
                
                # Log search info
                rospy.loginfo_throttle(1.0, f"Parking: Moving forward to find target AprilTag...")
                
                # If we've been searching too long, move to next state anyway
                if elapsed > 7.0:
                    if self.has_target_tag:
                        rospy.logwarn(f"Parking: Could not find target tag {self.target_tag_id}. Moving to state 2 anyway.")
                    else:
                        rospy.logwarn("Parking: Could not find any AprilTag. Moving to state 2 anyway.")
                    self.parking_state = 2
                    self.parking_state_start_time = current_time
            else:
                # Once tag is detected, move to alignment state
                self.parking_state = 1.6
                self.parking_state_start_time = current_time
                self.tag_alignment_count = 0  # Reset alignment counter
                rospy.loginfo(f"Parking: Target AprilTag {self.target_tag_id} detected. Moving to state 1.6: Aligning with tag.")
                
        elif self.parking_state == 1.6:
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
                rospy.loginfo_throttle(0.5, f"Parking: Aligning: Tag at {self.tag_x_center}, Error: {error}")
                
                # Check if TOF is getting close during alignment
                if self.tof_distance < self.stop_distance:
                    self.parking_state = 3  # Go straight to stopping if we're close enough
                    self.parking_state_start_time = current_time
                    rospy.loginfo(f"Parking: TOF detected close obstacle during alignment: {self.tof_distance:.2f}m. Stopping.")
                
                # If aligned within threshold, move to next state
                elif abs(error) < self.alignment_threshold:
                    self.tag_alignment_count += 1
                    
                    # Require multiple consecutive aligned frames
                    if self.tag_alignment_count > 5:
                        self.parking_state = 2
                        self.parking_state_start_time = current_time
                        rospy.loginfo(f"Parking: Aligned with target tag {self.target_tag_id}. Moving to state 2: Drive straight into stall")
                else:
                    # Reset alignment counter if not aligned
                    self.tag_alignment_count = 0
            else:
                # If tag lost during alignment, continue with last known position for a bit
                if self.tag_lost_count < 10:
                    # Use last known position for a brief period
                    rospy.logwarn_throttle(1.0, f"Parking: Target tag {self.target_tag_id} temporarily lost during alignment, using last position")
                    cmd.v = 0.1  # Slower movement when tag is lost
                else:
                    # If tag lost for too long, go back to search state
                    self.parking_state = 1.5
                    self.parking_state_start_time = current_time
                    rospy.logwarn(f"Parking: Lost target tag {self.target_tag_id} during alignment. Returning to search.")
                
        elif self.parking_state == 2:
            # Drive straight until TOF sensor detects obstacle
            cmd.v = 0.15  # Slower approach speed for more controlled stopping
            cmd.omega = 0
            
            # Check for TOF sensor health
            tof_age = current_time - self.tof_last_update
            if tof_age > 1.0:  # No TOF updates in more than 1 second
                rospy.logwarn_throttle(1.0, f"Parking: TOF sensor not updating ({tof_age:.1f}s old)")
            
            # Log the TOF distance
            rospy.loginfo_throttle(0.5, f"Parking: Approaching stall, TOF distance: {self.tof_distance:.2f}m (stop at {self.stop_distance:.2f}m)")
            
            if self.tof_distance < self.stop_distance:  # Obstacle detected within stop_distance
                self.parking_state = 3
                self.parking_state_start_time = current_time
                rospy.loginfo(f"Parking: Obstacle detected at {self.tof_distance:.2f}m, stopping")
            elif elapsed > 8.0:  # Safety timeout after 8 seconds
                self.parking_state = 3
                self.parking_state_start_time = current_time
                rospy.loginfo("Parking: Safety timeout reached, stopping")
                
        elif self.parking_state == 3:
            # Parking complete - stop the robot
            cmd.v = 0
            cmd.omega = 0
            
            if elapsed < 1.0:  # Just log once
                if self.has_target_tag:
                    rospy.loginfo(f"Parking complete in stall {self.stall} (targeting tag {self.target_tag_id})!")
                else:
                    rospy.loginfo(f"Parking complete in stall {self.stall}!")

                self.set_led_color("purple")

                rospy.Timer(rospy.Duration(2.0), self.shutdown_robot, oneshot=True)
        
        # Publish the command
        self.vel_pub.publish(cmd)
        return True


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
            else:
                # Define white color range
                turn_cmd = Twist2DStamped()
            
                # Less forward velocity and stronger turning for a tighter curve
                turn_cmd.v = 0.3  # Lower forward velocity for a tighter turn
                turn_cmd.omega = 2  # Stronger turning rate for a sharper curve

        
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


        if self.parking_enabled:
            if self.park():
                return
        # Decrement red line cooldown if active
        if self.red_line_cooldown > 0:
            self.red_line_cooldown -= 1/8.0  # Assuming 8Hz loop rate

        # Enable blue line detection after the fifth red line
        if self.red_line_count >= 5 and not self.blue_line_detection_enabled and not self.duck_detection_enabled and self.blue_crosswalk_count == 0:
            self.blue_line_detection_enabled = True
            rospy.loginfo("Blue crosswalk detection enabled")
        
        # Handle crosswalk logic
        if self.at_crosswalk:
            rospy.loginfo(f"At crosswalk, waiting for duck (duck_detected: {self.duck_detected}, counter: {self.duck_detected_false_counter}/80)")
            
            # Stop the robot while duck is detected or while counter hasn't reached threshold
            if self.duck_detected:
                rospy.loginfo("Duck detected at crosswalk, stopping")
                self.set_led_color("purple")  # Use purple LED for duck waiting
                self.stop(2)
            
            # Only proceed if we're confident no duck is present
            if not self.duck_detected and self.duck_detected_false_counter >= 80:
                rospy.loginfo(f"No duck detected for {self.duck_detected_false_counter} consecutive frames. Resuming after crosswalk")
                self.at_crosswalk = False
                self.duck_detection_enabled = False
                self.blue_line_detection_enabled = False
                self.blue_crosswalk_detected = False
                self.set_led_color("green")
            
            # Skip the rest of the drive function while at crosswalk
            return

        # Set passed_first_crosswalk flag when robot passes first blue crosswalk
        if self.blue_crosswalk_count >= 1 and not self.passed_first_crosswalk and not self.at_crosswalk:
            self.passed_first_crosswalk = True
            rospy.loginfo("Passed first blue crosswalk. Will monitor for broken duckiebot")
        
        # Check for broken duckiebot after first crosswalk
        if self.passed_first_crosswalk and self.obj_stop and not self.maneuvering and not self.done_man:
            rospy.loginfo("Broken duckiebot detected! Starting maneuvering sequence.")
            self.broken_bot_detected = True
            self.maneuvering = True
            self.maneuver_state = 0
            self.state_time = 0
        
        # Handle maneuvering around broken duckiebot
        if self.maneuvering:
            self.maneuver_around_bot()
            # self.maneuvering = False
            # self.blue_line_detection_enabled = False
            # self.blue_crosswalk_detected = False
            # self.duck_detection_enabled = False
            return
        
        # Check for red line detection and handle it
        if self.red_line_detected and not self.at_red_line and self.red_line_cooldown <= 0:
            self.at_red_line = True
            self.red_line_count += 1
            rospy.loginfo(f"RED LINE DETECTED! Stopping for {self.red_line_stop_time} seconds. Count: {self.red_line_count}")
            
            # Stop the robot
            self.stop_at_red_line()
            
            # Set cooldown to avoid detecting the same line multiple times
            self.red_line_cooldown = self.red_line_cooldown_time

            if self.red_line_count >= 3:
                self.red_line_cooldown = self.red_line_cooldown_time / 3
            
            # Reset flag after handling
            self.red_line_detected = False
            self.at_red_line = False



            if self.red_line_count == 6:
                rospy.loginfo("SIXTH RED LINE DETECTED! Enabling parking mode.")
                self.parking_enabled = True
                self.parking_state = 0
                self.parking_state_start_time = rospy.get_time()
                # Reset tag tracking for parking
                self.tag_detected = False
                self.tag_detection_count = 0
                self.tag_lost_count = 0
                self.has_target_tag = False
                self.target_tag_id = -1
                # Set LED to indicate parking mode
                self.set_led_color("blue")
                return
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
            straight_cmd.omega = -1  # No turning
            
            # Move straight for 2 seconds
            straight_duration = 4.0
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
            self.red_line_cooldown = self.red_line_cooldown_time / 3
            rospy.loginfo(f"Reduced cooldown time to {self.red_line_cooldown} seconds for faster fourth line detection")
        
            # Enable April tag detection after the third red line
            self.detect_april_tags = True
            rospy.loginfo("April Tag detection enabled for next red line")


        elif self.red_line_count == 4:
            rospy.loginfo("Fourth red line detected. Looking for April Tag...")
            rospy.loginfo(type(self.detected_tag))
            # Make the opposite turn from what was done at the first red line
            if self.detected_tag == 48:  # Tag 48
                rospy.loginfo("Tag 48 detected! Turning LEFT.")
                self.turn_left()
                self.detect_april_tags = True
            elif self.detected_tag == 50:
                rospy.loginfo("Tag 50 detected! Turning RIGHT.")
                self.turn_right()
                self.detect_april_tags = True
            else:
                rospy.loginfo(f"No valid tag detected (got {self.detected_tag}). Waiting at red line.")
                # Wait at the red line
                start_time = rospy.get_time()
                while rospy.get_time() - start_time < self.red_line_stop_time:
                    self.vel_pub.publish(self.twist)
                    rospy.sleep(0.1)
        
        elif self.red_line_count == 5:
            rospy.loginfo("Fifth red line detected. Enabling Blue detection at crosswalks.")

            rospy.loginfo(type(self.detected_tag))
            # Make the opposite turn from what was done at the first red line
            if self.detected_tag == 48:  # Tag 48
                rospy.loginfo("Tag 48 detected! Turning LEFT.")
                self.turn_left()
            elif self.detected_tag == 50:
                rospy.loginfo("Tag 50 detected! Turning RIGHT.")
                self.turn_right()
            else:
                rospy.loginfo(f"No valid tag detected (got {self.detected_tag}). Waiting at red line.")
                # Wait at the red line
                start_time = rospy.get_time()
                while rospy.get_time() - start_time < self.red_line_stop_time:
                    self.vel_pub.publish(self.twist)
                    rospy.sleep(0.1)
            self.blue_line_detection_enabled = True
            self.blue_crosswalk_detected = False
            self.duck_detection_enabled = False

        elif self.red_line_count == 6:
            rospy.loginfo("Sixth red line detected. Preparing for parking maneuver...")
            # Just wait at the red line before enabling parking mode
            # The actual parking mode will be enabled in the drive method after returning
            start_time = rospy.get_time()
            while rospy.get_time() - start_time < self.red_line_stop_time:
                self.vel_pub.publish(self.twist)
                rospy.sleep(0.1)
            return

            
        #     # Check multiple frames for April Tags
        #     detected_tag_id = None
        #     frames_to_check = 10
            
        #     for i in range(frames_to_check):
        #         if self.last_image is not None:
        #             tag_id, _ = self.detect_april_tag(self.last_image)
        #             if tag_id is not None:
        #                 detected_tag_id = tag_id
        #                 break
        #         # Short sleep between checks
        #         rospy.sleep(0.2)
            
        #     # Make decision based on tag ID
        #     if detected_tag_id == 59:
        #         rospy.loginfo("Tag 48 detected! Turning LEFT.")
        #         self.turn_left()
        #     elif detected_tag_id == 113:
        #         rospy.loginfo("Tag 50 detected! Turning RIGHT.")
        #         self.turn_right()
        #     else:
        #         rospy.loginfo(f"No valid tag detected (got {detected_tag_id}). Waiting at red line.")
        #         # Wait at the red line
        #         start_time = rospy.get_time()
        #         while rospy.get_time() - start_time < self.red_line_stop_time:
        #             self.vel_pub.publish(self.twist)
        #             rospy.sleep(0.1)
            
        #     # Disable April tag detection after handling
        #     self.detect_april_tags = False
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

    def shutdown_robot(self, event=None):
        """Shutdown the robot using the dts command"""
        rospy.loginfo("PARKING COMPLETE - SHUTTING DOWN ROBOT")
        
        # Final stop command to ensure motors are off
        cmd = Twist2DStamped()
        cmd.v = 0
        cmd.omega = 0
        for i in range(5):  # Send multiple stop commands to ensure it's received
            self.vel_pub.publish(cmd)
            rospy.sleep(0.1)
        
        # Set LEDs to off
        self.set_led_color("off")
        
        # Get the robot hostname
        hostname = os.environ.get('VEHICLE_NAME', 'csc22905')
        
        # Log shutdown message
        rospy.loginfo(f"Executing shutdown command for {hostname}.local")
        
        try:
            # Create the shutdown command
            shutdown_command = f"dts duckiebot shutdown {hostname}.local"
            
            # Execute the shutdown command
            subprocess.Popen(shutdown_command, shell=True)
            
            rospy.loginfo(f"Shutdown command sent: {shutdown_command}")
        except Exception as e:
            rospy.logerr(f"Error executing shutdown command: {e}")
            
        # Also request ROS node shutdown (as a fallback)
        rospy.signal_shutdown("Parking maneuver completed successfully")

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
    args = parse_args()
    node = LaneFollowWithDetectionNode("lane_follow_with_detection_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()