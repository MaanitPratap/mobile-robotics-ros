#!/usr/bin/env python3

# potentially useful for question - 1.1 - 1.4 and 2.1

# import required libraries
import os
import rospy
import numpy as np


from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image, CameraInfo


import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String

from led_service.srv import LaneDetect

class LaneDetectionNode(DTROS):
    def __init__(self, node_name):
        super(LaneDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        # add your code here

        # Create CV bridge
        self.bridge = CvBridge()
          # Other variables
        self.rate = rospy.Rate(3)

        self.last_color = None



        vehicle_name = os.environ['VEHICLE_NAME']
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.has_camera_info = False
        
        # Subscribe to camera topics
        self.camera_info_sub = rospy.Subscriber(f"/{vehicle_name}/camera_node/camera_info", CameraInfo, self.camera_info_callback)
        
        
        # Publisher for undistorted image
        self.undistorted_pub = rospy.Publisher('~undistorted_image/compressed', CompressedImage, queue_size=1)
        
        # Flag to save calibration images once
        self.saved_calibration_images = False
        
        self.log("Image undistorter initialized")

        # Color detection parameters in HSV format
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([140, 255, 255])
        self.lower_red = np.array([0, 150, 50])
        self.upper_red = np.array([10, 255, 255])
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([90, 200, 255])
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.lower_white = np.array([0, 0, 220])
        self.upper_white = np.array([255, 35, 255])
        


        # lane detection publishers
        self.lane_behavior_service = None

        try:
            rospy.wait_for_service('behavior_service', timeout=1)
            self.lane_behavior_service = rospy.ServiceProxy('behavior_service', LaneDetect)
        except rospy.ROSException:
            self.lane_behavior_service = None
    
        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber(f"/{vehicle_name}/camera_node/image/compressed", CompressedImage, self.callback)
    
        # Other variables

        # LED
        
        # ROI vertices
        
        # define other variables as needed
    
    def camera_info_callback(self, msg):
        """Callback to receive camera intrinsic parameters"""
        if not self.has_camera_info:
            # Extract camera matrix and distortion coefficients
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            # Define ground projection homography matrix
            # self.homography = np.array([
            #         -2.8347669831541546e-06,
            #         0.0005025276979442388,
            #         0.2905589459220745,
            #         -0.0016178269315664051,
            #         2.2655060573732427e-05,
            #         0.5233209044481271,
            #         0.00012540721704065155,
            #         0.011871550622460428,
            #         -1.7115851603291103
            # ]).reshape(3, 3)

            self.homography = np.array([
                -4.3206292146280124e-05,
                0.0004805216196272236,
                0.2869625589246484,
                -0.00160575582723828,
                6.154315694680119e-05,
                0.397570514773939,
                -0.0001917830439245288,
                0.010136558604291714,
                -1.0932556526691932,
            ]).reshape(3, 3)
            
            self.log("Using calibrated homography matrix")
                
            self.has_camera_info = True
            self.log("Camera calibration parameters received")
        

    def undistort_image(self, image):
        # add your code here
        
        if not self.has_camera_info:
            return image
        
        h, w = image.shape[:2]
        
        # Get optimal new camera matrix and undistort the image
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 0.8, (w, h)
        )
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, newcameramtx
        )
        
        # Crop the image if needed
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted

    def preprocess_image(self, image):
        """
        Preprocess the image by resizing and applying smoothing
        
        Args:
            image: Input image (numpy array)
                
        Returns:
            Preprocessed image
        """
        # Resize the image
        target_width = 320
        target_height = 240
        resized = cv2.resize(image, (target_width, target_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        return blurred
    

    def detect_lane_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {
            "blue": cv2.inRange(hsv_image, self.lower_blue, self.upper_blue),
            "red": cv2.inRange(hsv_image, self.lower_red, self.upper_red),
            "green": cv2.inRange(hsv_image, self.lower_green, self.upper_green),
            # "yellow": cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow),
            # "white": cv2.inRange(hsv_image, self.lower_white, self.upper_white)
        }
        return masks
    

    def calculate_lane_dimension(self, u, v):
        pixel_coord = np.array([u, v, 1]).reshape(3, 1)
        world_coord = np.dot(self.homography, pixel_coord)
        world_coord /= world_coord[2]
        return world_coord[:2].flatten()

    def detect_lane(self, image, masks):
        colors = {
            "blue": (255, 0, 0), 
            "red": (0, 0, 255), 
            "green": (0, 255, 0), 
            "yellow": (0, 255, 255), 
            "white": (255, 255, 255)
        }
        detected_colors = []

        for color_name, mask in masks.items():
            masked_color = cv2.bitwise_and(image, image, mask=mask)
            gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 200: 
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), colors[color_name], 2)
                    lane_start = self.calculate_lane_dimension(x, y)
                    lane_end = self.calculate_lane_dimension(x + w, y)
                    lane_length = abs(lane_end[0] - lane_start[0]) * 100
                    cv2.putText(image, f"{lane_length:.2f} cm", (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, colors[color_name])
                    detected_colors.append(color_name)
        return image, detected_colors
    
    def callback(self, msg):
        # add your code here

        # Convert compressed image to CV2
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Skip processing if camera parameters not yet received
        if not self.has_camera_info:
            self.logwarn_throttle(1, "Waiting for camera calibration parameters...")
            return
        
        # Undistort image
        undistorted = self.undistort_image(cv_image)
        
        # 2: Preprocess the undistorted image (resize and blur)
        processed = self.preprocess_image(undistorted)
    
        # # Publish undistorted image
        # undistorted_msg = self.bridge.cv2_to_compressed_imgmsg(processed)
        # self.undistorted_pub.publish(undistorted_msg)
        
        # detect lanes - 2.1 

        # Detect lanes and colors
        masks = self.detect_lane_color(processed)
        lane_detected_image, detected_colors = self.detect_lane(processed.copy(), masks)
    
    
        # Publish processed image
        self.undistorted_pub.publish(self.bridge.cv2_to_compressed_imgmsg(lane_detected_image))
    
    
        # Publish lane detection results (color)
        if detected_colors:
            detected_color = detected_colors[0]  # Publish the first detected color
    
            if detected_color != self.last_color:
                if self.lane_behavior_service is not None:
                    self.lane_behavior_service(detected_color)
                # self.color_pub.publish(detected_color)
                self.last_color = detected_color
                rospy.loginfo(f"Detected lane color: {detected_color}")
                    
                try:
                    if self.lane_behavior_service is not None:
                        self.lane_behavior_service("shutdown")
                except rospy.service.ServiceException:
                    rospy.signal_shutdown("Task completed.")
        else:
            if self.last_color != "None":
                if self.lane_behavior_service is not None:
                    self.lane_behavior_service("None")
                # self.color_pub.publish("None")
                self.last_color = "None"
                rospy.loginfo(f"No color detected")
        
        # control LEDs based on detected colors

        # anything else you want to add here
    

    # add other functions as needed

if __name__ == '__main__':
    node = LaneDetectionNode(node_name='lane_detection_node')
    rospy.spin()