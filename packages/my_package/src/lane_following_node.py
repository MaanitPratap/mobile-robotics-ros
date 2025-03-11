#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from duckietown_msgs.msg import WheelsCmdStamped
from cv_bridge import CvBridge


class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        self.vehicle_name = os.environ['VEHICLE_NAME']
        self.bridge = CvBridge()

        # PD controller parameters tuned for better cornering
        self.kp = 0.6
        self.kd = 0.35
        self.ki = 0.03

        self.prev_error = 0
        self.integral = 0.0
        self.last_time = rospy.get_time()

        self.base_speed = 0.4
        self.max_speed = 0.7
        self.min_speed = 0.1

        self.camera_matrix = None
        self.dist_coeffs = None

        # HSV thresholds for yellow and white
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 150])
        self.upper_white = np.array([180, 60, 255])

        # Publishers
        self.pub_cmd = rospy.Publisher(
            f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd", WheelsCmdStamped, queue_size=1
        )
        self.image_pub = rospy.Publisher(
            f"/{self.vehicle_name}/processed_image", Image, queue_size=10
        )

        # Subscribers
        rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/camera_info", CameraInfo, self.camera_info_callback
        )
        rospy.Subscriber(
            f"/{self.vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback
        )

    def camera_info_callback(self, msg):
        """Initialize camera parameters once."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.dist_coeffs = np.array(msg.D)

    def undistort_image(self, image):
        """Undistort using known camera parameters (if available)."""
        if self.camera_matrix is None:
            return image
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        # Crop the valid region
        return undistorted[y:y+h, x:x+w]

    def preprocess_image(self, image):
        """Resize and blur the image to reduce noise."""
        resized = cv2.resize(image, (320, 240))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred

    def detect_lane_color(self, image):
        """Convert to HSV and threshold for yellow and white lanes."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_white = cv2.inRange(hsv, self.lower_white, self.upper_white)
        return {"yellow": mask_yellow, "white": mask_white}

    def detect_lane(self, image, masks):
        """
        Find the largest contour for each color (yellow or white),
        draw a bounding box, and compute its center x-position.
        """
        colors = {"yellow": (0, 255, 255), "white": (255, 255, 255)}
        yellow_x, white_x = None, None

        for color_name, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0
            best_contour = None

            # Find the largest contour for the current color
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    best_contour = contour

            # If we found a largest contour, draw it and record its center x
            if best_contour is not None:
                x, y, w, h = cv2.boundingRect(best_contour)
                cx = x + w / 2
                # Assign the center x to the appropriate color
                if color_name == "yellow":
                    yellow_x = cx
                else:
                    white_x = cx
                # Draw bounding box for visualization
                cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)

        return image, yellow_x, white_x

    def calculate_error(self, image):
        """
        1. Preprocess image
        2. Detect yellow and white contours
        3. Compute lane center based on which colors are found
        4. Return normalized cross-track error relative to the image midpoint
        """
        preprocessed_image = self.preprocess_image(image)
        masks = self.detect_lane_color(preprocessed_image)
        lane_detected_image, yellow_x, white_x = self.detect_lane(preprocessed_image, masks)

        # Publish processed image for visualization/debug
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(lane_detected_image, "bgr8"))

        # Midpoint of the processed image (width direction)
        mid_point = preprocessed_image.shape[1] / 2

        # Decide where the lane center is, based on what we detected
        if (yellow_x is not None) and (white_x is not None):
            lane_center = (yellow_x + white_x) / 2
        elif yellow_x is not None:
            # We only see yellow line; assume we should be 80 px to the right
            lane_center = yellow_x + 80
        elif white_x is not None:
            # We only see white line; assume we should be 80 px to the left
            lane_center = white_x - 100
        else:
            # No lines found; fall back to the image center
            lane_center = mid_point

        error = (mid_point - lane_center) / mid_point
        return error

    def pd_control(self, error):
        """
        Simple PD control: control = Kp * error + Kd * d(error)/dt
        """
        current_time = rospy.get_time()
        dt = current_time - self.last_time
        # Avoid division by zero if dt is extremely small
        derivative = (error - self.prev_error) / max(dt, 1e-4)

        control = self.kp * error + self.kd * derivative

        self.prev_error = error
        self.last_time = current_time
        return control
    
    def calculate_pid_control(self, error):
        dt = rospy.get_time() - self.last_time
        if dt <= 0:
            return 0
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.last_time = rospy.get_time()
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def image_callback(self, msg):
        """Main image callback: 1) Undistort, 2) Crop, 3) Compute error, 4) PD control, 5) Publish wheel cmds."""
        # Convert ROS compressed image to OpenCV
        image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        undistorted = self.undistort_image(image)

        # Crop bottom half (where the lane is most visible)
        cropped_image = undistorted[undistorted.shape[0] // 2 :, :]

        # Calculate cross-track error
        error = self.calculate_error(cropped_image)

        # PD control
        control = self.pd_control(error)

        # Adjust speeds based on error magnitude
        speed_factor = max(1 - abs(error), 0.5)
        left_speed = np.clip(self.base_speed * speed_factor - control, self.min_speed, self.max_speed)
        right_speed = np.clip(self.base_speed * speed_factor + control, self.min_speed, self.max_speed)

        # Publish wheel commands
        cmd = WheelsCmdStamped()
        cmd.vel_left = left_speed
        cmd.vel_right = right_speed
        self.pub_cmd.publish(cmd)


if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.spin()
