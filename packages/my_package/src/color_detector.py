#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from std_srvs.srv import Empty, EmptyResponse
import os

class ColorDetector:
    def __init__(self):
        # Initialize the node
        rospy.init_node('color_detector')
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Color detection parameters in HSV format - you'll manually define these
        # Default values provided but you should replace with your own determined values
        self.hsv_ranges = {
            'blue': {'lower': np.array([94, 80, 2], np.uint8), 'upper': np.array([120, 255, 255], np.uint8)},
            'red': {'lower': np.array([136, 87, 111], np.uint8), 'upper': np.array([180, 255, 255], np.uint8)},
            'green': {'lower': np.array([25, 52, 72], np.uint8), 'upper': np.array([102, 255, 255], np.uint8)}
        }
        
        vehicle_name = os.environ['VEHICLE_NAME']
        # Store the latest image
        self.latest_image = None
        
        # Subscribe to preprocessed image
        self.image_sub = rospy.Subscriber(f"/{vehicle_name}/image_undistorter/undistorted_image/compressed", CompressedImage, self.image_callback)
        
        # Publisher for detection image
        self.detection_pub = rospy.Publisher(f"/{vehicle_name}/detection_image/compressed", CompressedImage, queue_size=1)
        
        # Service to update HSV values if needed
        self.update_hsv_service = rospy.Service('~update_hsv_values', Empty, self.update_hsv_values_service)
        
        rospy.loginfo("Color detector initialized")
        rospy.loginfo("Manual HSV value setup - use the update_hsv_values service if needed")
    
    def update_hsv_values_service(self, req):
        """Service to update HSV values for detection - you would implement this to set your manually determined values"""
        # RECOMMENDED LOCATION FOR IMAGES:
        # Place your images in: /home/maanitpratap/mobile-robotics-ros/lane_images/
        # Example file paths:
        #   /home/maanitpratap/mobile-robotics-ros/lane_images/blue_lane.jpg
        #   /home/maanitpratap/mobile-robotics-ros/lane_images/red_lane.jpg 
        #   /home/maanitpratap/mobile-robotics-ros/lane_images/green_lane.jpg
        
        # Example of manually setting values - replace these with your determined values
        self.hsv_ranges['blue']['lower'] = np.array([94, 80, 2], np.uint8)  # Your blue lower value
        self.hsv_ranges['blue']['upper'] = np.array([120, 255, 255], np.uint8)  # Your blue upper value
        
        self.hsv_ranges['red']['lower'] = np.array([136, 87, 111], np.uint8)  # Your red lower value
        self.hsv_ranges['red']['upper'] = np.array([180, 255, 255], np.uint8)  # Your red upper value
        
        self.hsv_ranges['green']['lower'] = np.array([25, 52, 72], np.uint8)  # Your green lower value
        self.hsv_ranges['green']['upper'] = np.array([102, 255, 255], np.uint8)  # Your green upper value
        
        rospy.loginfo("HSV values updated manually")
        return EmptyResponse()
    
    def detect_lane_color(self, image):
        """Detect lane colors (blue, red, green) in the image"""
        # Convert to HSV
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create a kernel for morphological operations
        kernel = np.ones((5, 5), "uint8")
        
        results = {}
        
        # Process each color separately using your manually determined HSV ranges
        color_masks = {}
        
        # Blue color detection
        blue_mask = cv2.inRange(hsv_frame, self.hsv_ranges['blue']['lower'], self.hsv_ranges['blue']['upper'])
        blue_mask = cv2.dilate(blue_mask, kernel)
        color_masks['blue'] = blue_mask
        
        # Red color detection
        red_mask = cv2.inRange(hsv_frame, self.hsv_ranges['red']['lower'], self.hsv_ranges['red']['upper'])
        red_mask = cv2.dilate(red_mask, kernel)
        color_masks['red'] = red_mask
        
        # Green color detection
        green_mask = cv2.inRange(hsv_frame, self.hsv_ranges['green']['lower'], self.hsv_ranges['green']['upper'])
        green_mask = cv2.dilate(green_mask, kernel)
        color_masks['green'] = green_mask
        
        # Find contours for each color and get dimensions
        for color, mask in color_masks.items():
            # Create bitwise_and result for visualization if needed
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]
            
            if valid_contours:
                # Get largest contour
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Store the detection result
                results[color] = {
                    'contour': largest_contour,
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(largest_contour),
                    'mask': mask
                }
        
        return results
    
    def get_lane_dimensions(self, bbox, image_height, image_width):
        """Calculate lane dimensions and estimate distance"""
        x, y, w, h = bbox
        
        # Calculate pixel dimensions
        pixel_dimensions = {'width': w, 'height': h}
        
        # Calculate aspect ratio (width to height)
        aspect_ratio = w / h if h > 0 else 0
        
        # Calculate position in frame (center of bounding box)
        center_x = x + w/2
        center_y = y + h/2
        
        # Normalized coordinates (0-1 range)
        norm_x = center_x / image_width
        norm_y = center_y / image_height
        
        # Distance from center of frame
        center_offset_x = norm_x - 0.5  # Negative = left, Positive = right
        
        # Estimate distance - combination of factors:
        # 1. Y position (lower in frame = closer)
        pos_factor = y / image_height  # 0 = top, 1 = bottom
        
        # 2. Size factor (larger = closer)
        size_factor = (w * h) / (image_width * image_height)
        normalized_size = min(1.0, size_factor * 100)  # Scale appropriately
        
        # Combined distance estimate (lower = further, higher = closer)
        # Higher weight on position as it's more reliable
        estimated_distance = 0.7 * pos_factor + 0.3 * normalized_size
        
        return {
            'pixel': pixel_dimensions,
            'aspect_ratio': aspect_ratio,
            'position': {'x': norm_x, 'y': norm_y},
            'center_offset': center_offset_x,
            'estimated_distance': estimated_distance
        }
    
    def image_callback(self, msg):
        """Callback for processing images"""
        try:
            # Store the latest image
            self.latest_image = msg
            
            # Convert compressed image to CV2
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Get image dimensions
            height, width = cv_image.shape[:2]
            
            # Detect lanes and colors
            detection_results = self.detect_lane_color(cv_image)
            
            # Create detection visualization
            detection_image = cv_image.copy()
            
            # Define color tuples for visualization
            color_bgr = {
                'blue': (255, 0, 0),    # Blue in BGR
                'green': (0, 255, 0),   # Green in BGR
                'red': (0, 0, 255)      # Red in BGR
            }
            
            # Process detection results for each color
            for color, data in detection_results.items():
                # Draw bounding box
                x, y, w, h = data['bbox']
                
                # Rectangle around detected lane
                cv2.rectangle(detection_image, (x, y), (x + w, y + h), color_bgr[color], 2)
                
                # Color label
                cv2.putText(detection_image, f"{color.capitalize()}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr[color], 2)
                
                # Get lane dimensions
                dimensions = self.get_lane_dimensions(data['bbox'], height, width)
                
                # Add dimension info below rectangle
                info_text = f"W:{w}px H:{h}px"
                cv2.putText(detection_image, info_text, (x, y + h + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color], 2)
                
                dist_text = f"Dist:{dimensions['estimated_distance']:.2f} Offset:{dimensions['center_offset']:.2f}"
                cv2.putText(detection_image, dist_text, (x, y + h + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color], 2)
                
                # Draw contour
                cv2.drawContours(detection_image, [data['contour']], -1, color_bgr[color], 2)
            
            # Add title and info to the frame
            cv2.putText(detection_image, "Lane Color Detection", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Publish detection image
            detection_msg = self.bridge.cv2_to_compressed_imgmsg(detection_image)
            self.detection_pub.publish(detection_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

if __name__ == '__main__':
    try:
        detector = ColorDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass