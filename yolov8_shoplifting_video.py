import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO("yolov8x.pt")

# Define class labels and confidence threshold
class_labels = model.names  # Get class labels from the model
confidence_threshold = 0.5  # Adjust based on detection accuracy needs

# Input and output video paths
video_path = r"C:\\Users\\leand\\OneDrive\\Desktop\\shoplift\\IMG_2404.MOV"
output_dir = r"C:\\Users\\leand\\OneDrive\\Desktop\\shoplift\\result"
output_path = os.path.join(output_dir, "output_video.mp4")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0.0 or fps is None:
    fps = 30.0  # Set a default FPS value

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Try 'mp4v', 'XVID', 'MJPG' if needed
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Check if VideoWriter opened successfully
if not out.isOpened():
    print("Error: Could not open VideoWriter.")
    cap.release()
    exit()

# Font and color settings
font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 1.0
font_thickness = 2
box_color = (0, 255, 0)      # Green for bounding boxes
text_color = (255, 255, 255) # White for labels
warning_color = (0, 0, 255)  # Red for warnings
outline_color = (0, 255, 255) # Yellow for outline (BGR format)

# Function to check if bounding boxes overlap
def boxes_overlap(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    return x_min1 < x_max2 and x_max1 > x_min2 and y_min1 < y_max2 and y_max1 > y_min2

# Function to check if bottle is in contact with a bag and display a warning
def check_bottle_in_bag(frame, result):
    boxes = result.boxes  # boxes object

    if boxes is not None:
        # Get the coordinates, class ids, and confidence scores
        xyxy = boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes
        confs = boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = boxes.cls.cpu().numpy().astype(int)  # Class IDs

        num_detections = len(boxes)

        # List of items that could hide a stolen object
        bag_classes = ["backpack", "handbag", "suitcase", "umbrella", "briefcase"]

        # Draw detections only for "bottle" and bag_classes
        for i in range(num_detections):
            x_min, y_min, x_max, y_max = xyxy[i]
            class_id = class_ids[i]
            conf = confs[i]
            class_name = class_labels[class_id]

            if class_name == "bottle" or class_name in bag_classes:
                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} {conf:.2f}",
                    (x_min, y_min - 10),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

        # Check if bottle is in contact with a bag
        for i in range(num_detections):
            if class_labels[class_ids[i]] == "bottle":
                x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle = xyxy[i]
                conf_bottle = confs[i]

                for j in range(num_detections):
                    if i == j:
                        continue
                    if class_labels[class_ids[j]] in bag_classes:
                        x_min_bag, y_min_bag, x_max_bag, y_max_bag = xyxy[j]

                        # Check if the bottle bounding box overlaps with the bag bounding box
                        if boxes_overlap(
                            (x_min_bottle, y_min_bottle, x_max_bottle, y_max_bottle),
                            (x_min_bag, y_min_bag, x_max_bag, y_max_bag)
                        ) and conf_bottle > confidence_threshold:
                            # Display warning message with outline
                            text = "WARNING - TORCEDOR DO CORINTHIANS EM AÇÃO"
                            # Draw outline
                            cv2.putText(
                                frame,
                                text,
                                (x_min_bottle, y_min_bottle - 25),
                                font,
                                font_scale,
                                outline_color,
                                font_thickness + 1,
                                lineType=cv2.LINE_AA,
                            )
                            # Draw text
                            cv2.putText(
                                frame,
                                text,
                                (x_min_bottle, y_min_bottle - 25),
                                font,
                                font_scale,
                                warning_color,
                                font_thickness,
                                lineType=cv2.LINE_AA,
                            )
                            # Highlight the bottle bounding box
                            cv2.rectangle(
                                frame,
                                (x_min_bottle, y_min_bottle),
                                (x_max_bottle, y_max_bottle),
                                warning_color,
                                2,
                            )
                            break  # Stop checking other bags for this bottle

# Process the video frames
frame_count = 0
try:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_count += 1
            print(f"Processing frame {frame_count}")

            results = model(frame)

            # Check for bottle in contact with a bag
            check_bottle_in_bag(frame, results[0])

            # Write the frame to the output video
            out.write(frame)
        else:
            break
except Exception as e:
    print(f"An error occurred: {e}")

# Release resources
cap.release()
out.release()
print("Processing completed.")
