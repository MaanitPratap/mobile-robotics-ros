#!/usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from dt_apriltags import Detector
from sensor_msgs.msg import CompressedImage
from turbojpeg import TurboJPEG
from duckietown.dtros import DTROS, NodeType

class AprilTagDetectionNode(DTROS):
    def __init__(self, node_name):
        super(AprilTagDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.node_name = node_name
        self.veh = os.environ['VEHICLE_NAME']
        
        # Initialize April Tag detector
        self.tag_detector = Detector(families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)
        
        # Camera calibration parameters (fx, fy, cx, cy)
        # These are approximate, replace with your calibrated values
        self.camera_params = [305.57, 308.83, 303.02, 231.14]
        self.tag_size = 0.065  # Tag size in meters
        
        # Initialize JPEG decoder
        self.jpeg = TurboJPEG()
        
        # Subscribe to camera topic
        self.sub = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                   CompressedImage,
                                   self.callback,
                                   queue_size=1,
                                   buff_size="20MB")
        
        # Publisher for debug image with detected tags
        self.pub_debug_img = rospy.Publisher(f"/{self.veh}/apriltag_detection_node/debug/compressed",
                                           CompressedImage,
                                           queue_size=1)
        
        rospy.loginfo(f"[{self.node_name}] Initialized.")
    
    def callback(self, msg):
        """Process camera images and detect April Tags"""
        try:
            # Decode JPEG image
            img = self.jpeg.decode(msg.data)
            
            # Convert to grayscale for April Tag detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect April Tags
            tags = self.tag_detector.detect(gray, 
                                          estimate_tag_pose=False,
                                          camera_params=self.camera_params, 
                                          tag_size=self.tag_size)
            
            # Create debug image
            debug_img = img.copy()
            
            if len(tags) > 0:
                rospy.loginfo(f"[{self.node_name}] Detected {len(tags)} tags")
                
                # Process each detected tag
                for tag in tags:
                    tag_id = tag.tag_id
                    tag_family = tag.tag_family.decode('utf-8')
                    decision_margin = tag.decision_margin
                    
                    # Log detection information
                    rospy.loginfo(f"[{self.node_name}] Detected tag ID: {tag_id}, " +
                                 f"Family: {tag_family}, Confidence: {decision_margin:.2f}")
                    
                    # Draw tag outline
                    for idx in range(4):
                        cv2.line(debug_img, 
                               tuple(tag.corners[idx-1, :].astype(int)), 
                               tuple(tag.corners[idx, :].astype(int)),
                               (0, 255, 0), 2)
                    
                    # Draw tag center and ID
                    center = tag.center.astype(int)
                    cv2.circle(debug_img, tuple(center), 5, (0, 0, 255), -1)
                    cv2.putText(debug_img, f"ID: {tag_id}", (center[0] - 10, center[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Additional info
                    cv2.putText(debug_img, f"Conf: {decision_margin:.2f}", (center[0] - 10, center[1] + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                # No tag detected
                cv2.putText(debug_img, "No April Tag detected", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add some instructions to the debug image
            height = debug_img.shape[0]
            cv2.putText(debug_img, "Looking for tags: 48, 50", (10, height - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Publish debug image
            debug_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(debug_img))
            self.pub_debug_img.publish(debug_msg)
            
        except Exception as e:
            rospy.logerr(f"[{self.node_name}] Error processing image: {str(e)}")

if __name__ == "__main__":
    # Initialize the node
    node = AprilTagDetectionNode("apriltag_detection_node")
    # Keep it spinning
    rospy.spin()