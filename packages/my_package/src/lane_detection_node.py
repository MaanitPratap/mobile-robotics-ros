#!/usr/bin/env python3
import os
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import String

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # CV bridge and flags
        self.bridge = CvBridge()
        self.has_camera_info = False
        
        # Subscribe to camera topics
        vehicle_name = os.environ['VEHICLE_NAME']
        self.camera_info_sub = rospy.Subscriber(f"/{vehicle_name}/camera_node/camera_info",
                                                CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber(f"/{vehicle_name}/camera_node/image/compressed",
                                          CompressedImage, self.image_callback)
        
        # Publishers: one for the processed image and one for lane detection results
        self.undistorted_pub = rospy.Publisher('~undistorted_image/compressed', CompressedImage, queue_size=1)
        self.lane_pub = rospy.Publisher('lane_detections', String, queue_size=1)
        
      
        self.color_thresholds = {
            "blue":   (np.array([100, 150, 50]), np.array([140, 255, 255])),
            "red":    (np.array([0, 150, 50]),   np.array([10, 255, 255])),
            "green":  (np.array([35, 50, 50]),   np.array([90, 200, 255])),
            "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
            "white":  (np.array([0, 0, 220]),    np.array([255, 35, 255]))
        }
        

        self.homography = np.array([
            -4.3206292146280124e-05,  0.0004805216196272236,  0.2869625589246484,
            -0.00160575582723828,     6.154315694680119e-05,   0.397570514773939,
            -0.0001917830439245288,   0.010136558604291714,   -1.0932556526691932
        ]).reshape(3, 3)
        
        rospy.loginfo("LaneDetectionNode initialized.")
    
    def camera_info_callback(self, msg):
        # Obtain camera calibration parameters for undistortion
        if not self.has_camera_info:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.has_camera_info = True
            rospy.loginfo("Camera calibration parameters received.")
    
    def undistort_image(self, image):
        # Only undistort if calibration parameters are available
        if not self.has_camera_info:
            return image
        h, w = image.shape[:2]
        new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix,
                                                         self.dist_coeffs,
                                                         (w, h),
                                                         0.8,
                                                         (w, h))
        undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_cam_mtx)
        x, y, w, h = roi
        return undistorted[y:y+h, x:x+w] if w > 0 and h > 0 else undistorted

    def preprocess_image(self, image):
        # Resize for faster processing and smooth the image
        resized = cv2.resize(image, (320, 240))
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        return blurred
    
    def detect_lane_color(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {}
        for color, (lower, upper) in self.color_thresholds.items():
            masks[color] = cv2.inRange(hsv, lower, upper)
        return masks
    
    def calculate_lane_dimension(self, u, v):

        pixel_coord = np.array([u, v, 1]).reshape(3, 1)
        world_coord = self.homography.dot(pixel_coord)
        world_coord /= world_coord[2]
        return world_coord[:2].flatten()
    
    def detect_lane(self, image, masks):

        draw_colors = {
            "blue":   (255, 0, 0),
            "red":    (0, 0, 255),
            "green":  (0, 255, 0),
            "yellow": (0, 255, 255),
            "white":  (255, 255, 255)
        }
        detected_colors = []
        lane_distances = {}  # Stores lane "length" in centimeters per color
        

        for color, mask in masks.items():
            masked = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(image, (x, y), (x + w, y + h), draw_colors[color], 2)
                    # Use the top edge of the bounding box as reference
                    lane_start = self.calculate_lane_dimension(x, y)
                    lane_end = self.calculate_lane_dimension(x + w, y)
                    lane_length_cm = abs(lane_end[0] - lane_start[0]) * 100  # convert to centimeters
                    # Keep the smallest measured lane length per color (if multiple)
                    if color not in lane_distances or lane_length_cm < lane_distances[color]:
                        lane_distances[color] = lane_length_cm
                    cv2.putText(image, f"{lane_length_cm:.2f}cm", (x, y + h + 15),
                                cv2.FONT_HERSHEY_PLAIN, 1, draw_colors[color], 1)
                    if color not in detected_colors:
                        detected_colors.append(color)
        return image, detected_colors, lane_distances
    
    def image_callback(self, msg):
        # Convert the incoming ROS CompressedImage to an OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            rospy.logwarn("Received an empty image.")
            return
        

        undistorted = self.undistort_image(cv_image)
        processed = self.preprocess_image(undistorted)
        
        # Detect lane colors
        masks = self.detect_lane_color(processed)
        lane_image, detected_colors, lane_distances = self.detect_lane(processed.copy(), masks)
        

        comp_img_msg = self.bridge.cv2_to_compressed_imgmsg(lane_image)
        self.undistorted_pub.publish(comp_img_msg)
       
        lane_msg = ",".join([f"{color}_lane:{lane_distances[color]:.2f}" for color in lane_distances])
        self.lane_pub.publish(lane_msg)
        
    
        if detected_colors:
            rospy.loginfo_throttle(5, f"Detected lanes: {lane_msg}")
        else:
            rospy.loginfo_throttle(5, "No lanes detected.")

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()
