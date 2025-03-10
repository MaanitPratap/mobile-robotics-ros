#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge

from duckietown_msgs.msg import WheelEncoderStamped

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)
        
        # Get vehicle name
        self.vehicle_name = os.environ['VEHICLE_NAME']
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Controller type
        self.controller_type = 'PD'
        
        # PID gains (tune these values)
        self.Kp = 0.5  # Proportional gain
        self.Ki = 0.1  # Integral gain
        self.Kd = 0.3  # Derivative gain
        
        # Control variables
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.get_time()
        
        # Movement parameters
        self.base_speed = 0.3
        self.min_speed = 0.1
        self.max_speed = 0.5
        
        # Distance tracking
        self.target_distance = 1.5  # meters
        self.distance_traveled = 0.0
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.has_camera_info = False
        
        
        # Color detection parameters (HSV)
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 220])
        self.upper_white = np.array([255, 35, 255])

        # Initialize publishers/subscribers
        self.wheel_pub = rospy.Publisher(
            f'/{self.vehicle_name}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1
        )
        
        self.image_pub = rospy.Publisher(
            '~lane_detection/compressed',
            CompressedImage,
            queue_size=1
        )
        
        # Subscribe to camera topics
        self.camera_info_sub = rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/camera_info",
            CameraInfo,
            self.camera_info_callback
        )
        
        self.image_sub = rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self.image_callback
        )

    def camera_info_callback(self, msg):
        if not self.has_camera_info:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.has_camera_info = True
            rospy.loginfo("Camera calibration received")

    def undistort_image(self, image):
        if not self.has_camera_info:
            return image
            
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 0.8, (w, h)
        )
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, newcameramtx
        )
        return undistorted

    def preprocess_image(self, image):
        target_width = 320
        target_height = 240
        resized = cv2.resize(image, (target_width, target_height))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)
        white_mask = cv2.inRange(hsv_image, self.lower_white, self.upper_white)
        return {"yellow": yellow_mask, "white": white_mask}

    def detect_lane(self, image, masks):
        colors = {"yellow": (0, 255, 255), "white": (255, 255, 255)}
        detected_white, detected_yellow = False, False
        yellow_max_x = 0
        white_min_x = 1000

        for color_name, mask in masks.items():
            if color_name == "white":
                detected_white = True
            elif color_name == "yellow":
                detected_yellow = True
            else:
                continue

            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    x, y, w, h = cv2.boundingRect(contour)
                    if color_name == "yellow":
                        yellow_max_x = max(yellow_max_x, x + w / 2)
                    elif color_name == "white":
                        white_min_x = min(white_min_x, x + w / 2)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

        final_yellow_x = yellow_max_x if detected_yellow else 0
        final_white_x = white_min_x if detected_white else image.shape[1]
        return image, final_yellow_x, final_white_x

    def calculate_error(self, image):
        undistorted = self.undistort_image(image)
        preprocessed = self.preprocess_image(undistorted)
        masks = self.detect_lane_color(preprocessed)
        lane_detected, yellow_x, white_x = self.detect_lane(preprocessed, masks)

        # Publish processed image
        msg = self.bridge.cv2_to_compressed_imgmsg(lane_detected)
        self.image_pub.publish(msg)

        # Calculate error
        center = image.shape[1] // 2
        yellow_error = center - yellow_x
        white_error = white_x - center
        return (yellow_error - white_error) / center  # Normalized error

    def calculate_p_control(self, error):
        return self.Kp * error

    def calculate_pd_control(self, error):
        dt = rospy.get_time() - self.last_time
        if dt <= 0:
            return 0
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.last_time = rospy.get_time()
        return self.Kp * error + self.Kd * derivative

    def calculate_pid_control(self, error):
        dt = rospy.get_time() - self.last_time
        if dt <= 0:
            return 0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.last_time = rospy.get_time()
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def image_callback(self, msg):
        if not self.has_camera_info:
            return

        # Convert compressed image to CV2
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Calculate error from lane detection
        error = self.calculate_error(cv_image)

        # Calculate control output based on controller type
        if self.controller_type == 'P':
            control = self.calculate_p_control(error)
        elif self.controller_type == 'PD':
            control = self.calculate_pd_control(error)
        else:  # PID
            control = self.calculate_pid_control(error)

        # Calculate wheel speeds
        left_speed = self.base_speed - control
        right_speed = self.base_speed + control

        # Ensure speeds are within limits
        left_speed = max(min(left_speed, self.max_speed), self.min_speed)
        right_speed = max(min(right_speed, self.max_speed), self.min_speed)

        # Publish wheel commands
        msg = WheelsCmdStamped()
        msg.vel_left = left_speed
        msg.vel_right = right_speed
        self.wheel_pub.publish(msg)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()