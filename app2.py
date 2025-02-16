import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np

# Load the pre-trained Faster R-CNN model with a MobileNetV3 backbone
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# COCO dataset labels for object detection (add person, car, bike, truck, van)
COCO_LABELS = {
    1: 'person',    # Person
    3: 'car',       # Car
    44: 'bike',     # Bike
    8: 'truck',     # Truck
    38: 'van'       # Van
}

# Camera intrinsic parameters (focal length in pixels, assumed values)
focal_length_pixels = 1000  # Approximate value (need camera-specific data for accurate results)

# Real-world width of the objects (in meters)
REAL_WORLD_WIDTH = {
    'person': 0.5,   # Approximate width of a person in meters
    'car': 2.0,      # Approximate width of a car in meters
    'bike': 0.5,     # Approximate width of a bike in meters
    'truck': 2.5,    # Approximate width of a truck in meters
    'van': 2.0       # Approximate width of a van in meters
}

# Initialize video capture (replace with your video file or camera input)
cap = cv2.VideoCapture('IMG_1464.mov')  # Change to your video path

# Set up video writer to save output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (can also use 'MP4V' for .mp4)
output_video = cv2.VideoWriter('detection_output.avi', fourcc, 20.0, (frame_width, frame_height))  # Adjust FPS as needed

frame_count = 0
detection_interval = 5  # Process every 5th frame

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_count += 1
    if frame_count % detection_interval != 0:
        continue  # Skip frames to speed up processing

    # Resize the frame to reduce the input resolution (you can tweak the resolution)
    frame_resized = cv2.resize(frame, (640, 480))  # Resize to a smaller resolution
    frame_resized=frame
    # Convert the frame to a PIL image for processing by the model
    image_tensor = F.to_tensor(frame_resized).unsqueeze(0)  # Convert to tensor and add batch dimension

    # Perform object detection
    with torch.no_grad():
        prediction = model(image_tensor)

    # Get bounding boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # Filter for person, car, bike, truck, and van
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        label = labels[i].cpu().numpy()
        score = scores[i].cpu().numpy()

        if score > 0.7 and int(label) in COCO_LABELS:
            # Draw bounding box around the detected object
            x1, y1, x2, y2 = map(int, box)
            label_name = COCO_LABELS[int(label)]

            # Calculate the width of the bounding box (in pixels)
            object_width_pixels = x2 - x1

            # Estimate the distance using the formula
            if label_name in REAL_WORLD_WIDTH:
                real_world_width = REAL_WORLD_WIDTH[label_name]

                # Calculate the distance (Z) to the object
                distance = (focal_length_pixels * real_world_width) / object_width_pixels

                # Display the distance on the object
                distance_text = f"Dist_{label_name}: {distance:.2f} m"
                cv2.putText(frame_resized, distance_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw rectangle around the object
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label to the bounding box
            #cv2.putText(frame_resized, f"{label_name}: {score:.2f}", (x1, y1 - 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    output_video.write(frame_resized)

    # Print how many frames have been processed
    print(f"Processed {frame_count} frames.")

# Release the resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

print("Detection video saved as 'detection_output.avi'.")
