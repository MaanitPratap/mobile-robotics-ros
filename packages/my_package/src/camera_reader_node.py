#!/usr/bin/env python3

import os
import rospy
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        
        # create publisher for processed images
        self._pub = rospy.Publisher(
            f'/{self._vehicle_name}/processed_image/compressed',
            CompressedImage,
            queue_size=1
        )
        
        # subscribe to camera topic
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        # convert compressed image to opencv format
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
        # get image dimensions
        height, width = image.shape[:2]
        rospy.loginfo(f"Image size: {width}x{height}")
        
        # convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # convert back to BGR for adding colored text
        annotated_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # add text to image
        text = f"Duck {self._vehicle_name} says, 'Cheese! Capturing {width}X{height} - quack-tastic!'"
        cv2.putText(
            annotated_image,
            text,
            (10, 30),  # position
            cv2.FONT_HERSHEY_SIMPLEX,  # font
            1,  # font scale
            (0, 255, 0),  # color (green)
            2  # thickness
        )
        
        # display frame
        cv2.imshow(self._window, annotated_image)
        cv2.waitKey(1)
        
        # convert back to ROS compressed image and publish
        processed_msg = self._bridge.cv2_to_compressed_imgmsg(annotated_image)
        self._pub.publish(processed_msg)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()