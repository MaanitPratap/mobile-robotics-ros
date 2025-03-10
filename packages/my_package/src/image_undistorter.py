#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType

class ImageUndistorter(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS node
        super(ImageUndistorter, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        vehicle_name = os.environ['VEHICLE_NAME']
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.has_camera_info = False
        
        # Subscribe to camera topics
        self.camera_info_sub = rospy.Subscriber(f"/{vehicle_name}/camera_node/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber(f"/{vehicle_name}/camera_node/image/compressed", CompressedImage, self.image_callback)
        
        # Publisher for undistorted image
        self.undistorted_pub = rospy.Publisher('~undistorted_image/compressed', CompressedImage, queue_size=1)
        
        # Flag to save calibration images once
        self.saved_calibration_images = False
        
        self.log("Image undistorter initialized")
    
    def camera_info_callback(self, msg):
        """Callback to receive camera intrinsic parameters"""
        if not self.has_camera_info:
            # Extract camera matrix and distortion coefficients
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.has_camera_info = True
            self.log("Camera calibration parameters received")
        
        # if self.has_camera_info:
        #     self.log(f"Camera matrix shape: {self.camera_matrix.shape}")
        #     self.log(f"Distortion coeffs: {self.dist_coeffs}")
    
    def undistort_image(self, image):
        """Undistort the input image using camera intrinsic parameters"""
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
    
    def image_callback(self, msg):
        """Callback for processing and undistorting images"""
        try:
            # Convert compressed image to CV2
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Skip processing if camera parameters not yet received
            if not self.has_camera_info:
                self.logwarn_throttle(1, "Waiting for camera calibration parameters...")
                return
            
            # Save calibration board images if not already saved
            if not self.saved_calibration_images:
                # Save original distorted image
                cv2.imwrite('distorted_calibration.jpg', cv_image)
                
                # Get undistorted image
                undistorted = self.undistort_image(cv_image)
                
                # Save undistorted image
                cv2.imwrite('undistorted_calibration.jpg', undistorted)
                self.saved_calibration_images = True
                self.log("Saved calibration images")
            else:
                # Undistort image
                undistorted = self.undistort_image(cv_image)
            

            # Step 2: Preprocess the undistorted image (resize and blur)
            processed = self.preprocess_image(undistorted)
        
            # Save processed image for debugging
            if not hasattr(self, 'saved_processed'):
                cv2.imwrite('processed_image.jpg', processed)
                self.saved_processed = True
            # Publish undistorted image
            undistorted_msg = self.bridge.cv2_to_compressed_imgmsg(processed)
            self.undistorted_pub.publish(undistorted_msg)
            
        except Exception as e:
            self.logerr(f"Error in image callback: {e}")
    
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
        blurred = cv2.GaussianBlur(resized, (1, 1), 0)
        
        return blurred

if __name__ == '__main__':
    try:
        undistorter = ImageUndistorter(node_name='image_undistorter_node')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass