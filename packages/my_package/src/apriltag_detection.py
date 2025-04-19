#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
import re
import math
from dt_apriltags import Detector  # use dt_apriltags library
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown.dtros import DTROS, NodeType
from led_service.srv import SetLEDColor, SetLEDColorRequest
from std_msgs.msg import String  # standard String message
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped

class ApriltagNode(DTROS):
    def __init__(self, node_name):
        super(ApriltagNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        self.vehicle_name = os.environ["VEHICLE_NAME"]
        self.bridge = CvBridge()
        
        # --- Encoder Variables for Distance Measurement ---
        self.initial_ticks_left = None
        self.initial_ticks_right = None
        self.ticks_left = 0
        self.ticks_right = 0
        self.distance_traveled = 0.0  # in meters
        self.WHEEL_RADIUS = 0.0318     # in meters
        self.TICKS_PER_ROTATION = 135
        
        rospy.Subscriber(f"/{self.vehicle_name}/left_wheel_encoder_node/tick", WheelEncoderStamped, self.left_encoder_callback)
        rospy.Subscriber(f"/{self.vehicle_name}/right_wheel_encoder_node/tick", WheelEncoderStamped, self.right_encoder_callback)
        
        # --- Camera Calibration ---
        self.has_camera_info = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_mtx = None
        
        # --- Publishers ---
        self.pub_augmented = rospy.Publisher(
            f"/{self.vehicle_name}/apriltag_detection/augmented_image/compressed",
            CompressedImage,
            queue_size=1
        )
        self.wheel_pub = rospy.Publisher(
            f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1
        )
        
        # --- Subscribers for Camera & Lane Info ---
        rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.camera_info_callback
        )
        rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback
        )
        rospy.Subscriber("lane_detections", String, self.lane_detection_callback)
        self.lane_info = None  # will hold lane detection string
        
        # --- AprilTag Detector ---
        self.detection_divisor = rospy.get_param("~detection_divisor", 1)
        self.frame_count = 0
        quad_decimate = rospy.get_param("~quad_decimate", 1.0)
        quad_sigma = rospy.get_param("~quad_sigma", 0.0)
        refine_edges = rospy.get_param("~refine_edges", 1)
        decode_sharpening = rospy.get_param("~decode_sharpening", 0.25)
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
            debug=0
        )
        
        # --- LED Service ---
        rospy.wait_for_service('set_led_color')
        self.led_service = rospy.ServiceProxy('set_led_color', SetLEDColor)
        
        # --- AprilTag Detection Result ---
        # self.current_tag will be:
        #   "S" if Stop Sign (tag id 1) is detected,
        #   "T" if T-Intersection (tag id 2) is detected,
        #   "U" if UofA tag (tag id 3) is detected,
        #   None if no relevant tag is detected.
        self.current_tag = None
        
        # --- Motion Control State Machine ---
        self.state = "MOVING"   # states: "MOVING" or "RED_STOP"
        self.stop_start_time = None
        self.current_stop_duration = 0.0  # seconds to stop
        self.forward_speed = rospy.get_param("~forward_speed", 0.3)  # m/s
        self.min_distance_before_stop = rospy.get_param("~min_distance_before_stop", 0.30)  # meters
        self.red_lane_threshold = rospy.get_param("~red_lane_threshold", 5.0)  # cm
        
        # Timer for state machine (10 Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.main_loop)
        
        rospy.loginfo("ApriltagNode initialized.")
    
    # --- Encoder Callbacks & Distance Update ---
    def left_encoder_callback(self, msg):
        if self.initial_ticks_left is None:
            self.initial_ticks_left = msg.data
        self.ticks_left = msg.data - self.initial_ticks_left
        self.update_distance()
    
    def right_encoder_callback(self, msg):
        if self.initial_ticks_right is None:
            self.initial_ticks_right = msg.data
        self.ticks_right = msg.data - self.initial_ticks_right
        self.update_distance()
    
    def update_distance(self):
        avg_ticks = (self.ticks_left + self.ticks_right) / 2.0
        circumference = 2 * math.pi * self.WHEEL_RADIUS
        self.distance_traveled = (avg_ticks / self.TICKS_PER_ROTATION) * circumference
    
    def reset_encoders(self):
        self.initial_ticks_left = None
        self.initial_ticks_right = None
        self.ticks_left = 0
        self.ticks_right = 0
        self.distance_traveled = 0.0
        rospy.sleep(0.5)
    
    # --- Camera Calibration Callback ---
    def camera_info_callback(self, msg):
        if not self.has_camera_info:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.has_camera_info = True
            img_width = msg.width
            img_height = msg.height
            self.new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (img_width, img_height), 1, (img_width, img_height)
            )
            rospy.loginfo("Camera calibration received.")
    
    # --- Lane Detection Callback ---
    def lane_detection_callback(self, msg):
        self.lane_info = msg.data
        rospy.loginfo_throttle(5, f"Lane detection info received: {self.lane_info}")
    
    # --- Image Callback & AprilTag Detection ---
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.has_camera_info and self.new_camera_mtx is not None:
                cv_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs, None, self.new_camera_mtx)
            self.frame_count += 1
            if self.frame_count % self.detection_divisor == 0:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                detections = self.detector.detect(gray)
                # Default: no relevant tag detected
                self.current_tag = None
                led_color = "white"
                for detection in detections:
                    # Check for Stop Sign (tag id 1)
                    if detection.tag_id == 162:
                        self.current_tag = "S"
                        led_color = "red"
                        rospy.loginfo("Stop Sign tag detected; LED set to red.")
                        break
                    # Check for T-Intersection (tag id 2)
                    elif detection.tag_id == 133:
                        self.current_tag = "T"
                        led_color = "blue"
                        rospy.loginfo("T-Intersection tag detected; LED set to blue.")
                        break
                    # Check for UofA tag (tag id 3)
                    elif detection.tag_id == 200:
                        self.current_tag = "U"
                        led_color = "green"
                        rospy.loginfo("UofA tag detected; LED set to green.")
                        break
                try:
                    req = SetLEDColorRequest()
                    req.color = led_color
                    self.led_service(req)
                except Exception as e:
                    rospy.logerr(f"LED service call failed: {e}")
                # Draw overlays for visualization
                for detection in detections:
                    corners = detection.corners.astype(int)
                    cv2.polylines(cv_image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
                    center = detection.center.astype(int)
                    cv2.putText(cv_image, f"ID:{detection.tag_id}", tuple(center),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            aug_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image)
            self.pub_augmented.publish(aug_msg)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")
    
    # --- Wheel Command Publisher ---
    def publish_wheel_cmd(self, left_speed, right_speed):
        cmd = WheelsCmdStamped()
        cmd.vel_left = left_speed
        cmd.vel_right = right_speed
        self.wheel_pub.publish(cmd)
    
    # --- Main State Machine Loop ---
    def main_loop(self, event):
        current_time = rospy.Time.now()
        # Parse the red lane distance from lane_info (expected format: "red_lane:7.50,...")
        red_lane_distance = None
        if self.lane_info:
            match = re.search(r'red_lane:(\d+(\.\d+)?)', self.lane_info)
            if match:
                try:
                    red_lane_distance = float(match.group(1))
                except ValueError:
                    red_lane_distance = None
        
        if self.state == "MOVING":
            # Trigger stop only if red lane is detected within threshold and minimum distance traveled is reached.
            if (red_lane_distance is not None and red_lane_distance < self.red_lane_threshold and 
                self.distance_traveled >= self.min_distance_before_stop):
                # Choose stop duration based on the detected tag.
                if self.current_tag == "T":
                    stop_dur = 2.0  # T-Intersection
                elif self.current_tag == "U":
                    stop_dur = 1.0  # UofA tag
                elif self.current_tag == "S":
                    stop_dur = 3.0  # Stop Sign
                else:
                    stop_dur = 0.5  # No tag detected
                self.current_stop_duration = stop_dur
                rospy.loginfo(f"Red lane detected (distance: {red_lane_distance} cm) and distance traveled: {self.distance_traveled:.2f} m; stopping for {stop_dur} seconds before intersection.")
                self.state = "RED_STOP"
                self.stop_start_time = current_time
                self.publish_wheel_cmd(0, 0)
            else:
                self.publish_wheel_cmd(self.forward_speed, self.forward_speed)
        elif self.state == "RED_STOP":
            if (current_time - self.stop_start_time).to_sec() >= self.current_stop_duration:
                rospy.loginfo("Stop duration elapsed; resuming forward motion.")
                self.state = "MOVING"
                self.reset_encoders()  # reset distance measurement after stop
                self.publish_wheel_cmd(self.forward_speed, self.forward_speed)
            else:
                self.publish_wheel_cmd(0, 0)
    
    # --- Shutdown Handler ---
    def on_shutdown(self):
        self.publish_wheel_cmd(0, 0)
        try:
            req = SetLEDColorRequest()
            req.color = "off"
            self.led_service(req)
        except Exception as e:
            rospy.logerr(f"LED service call failed during shutdown: {e}")
        rospy.loginfo("Shutting down. Stopping robot and turning off LED.")
        super().on_shutdown()

if __name__ == '__main__':
    node = ApriltagNode(node_name='apriltag_detector_node')
    rospy.spin()
