import os
import cv2 as cv
import numpy as np

# The source path for the training images, from which negative examples will be extracted
input_path = "./antrenare/"
# Path where the negative examples will be stored (image dimension: 64x64)
output_path = "./antrenare/exempleNegative64"

# Check if the directory for negative examples exists; if not, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

characters = ["barney", "betty", "fred", "wilma"]

# Target size for the negative examples (64x64 pixels)
target_size = (64, 64)
# Number of negative examples to generate per rescaled image
num_samples_per_image = 40
# Different scaling factors to resize the image 
scales = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

cnt = 0  # Counter to track the number of negative examples generated

# Function that extracts the coordinates of all bounding boxes from an annotation file
def read_bounding_boxes_from_file(file_path):
    # Dictionary to store bounding boxes for each image
    image_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            image_file, x_min, y_min, x_max, y_max, who = line.strip().split(" ")

            # Store all bounding boxes for each image 
            if image_file in image_dict:
                image_dict[image_file].append((int(x_min), int(y_min), int(x_max), int(y_max)))
            else:
                image_dict[image_file] = [(int(x_min), int(y_min), int(x_max), int(y_max))]

    return image_dict

# Function that computes the Intersection over Union (IoU) for two bounding boxes
# Function adapted from the lab to ensure negative examples don’t overlap with character faces
def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    # Calculate the area of overlap (intersection area)
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Calculate the area of both bounding boxes
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    # Calculate the IoU - divide the intersection area by the union of both boxes
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

# Function that generates random negative examples from the images
def generate_random_neg_examples(input_path, output_path, bounding_boxes, cnt, scales,
                                 target_size, num_samples_per_image_scale):
    # Loop over all images in the input directory
    for filename in os.listdir(input_path):
        image_path = os.path.join(input_path, filename)
        img = cv.imread(image_path)

        # Retrieve all bounding boxes for the current image
        all_bboxes_file = bounding_boxes.get(filename)

        # Generate negative samples for each rescaled image
        for scale in scales:
            # List to store sampled bounding boxes to avoid duplicates
            boxes_for_samples = []
            img_scaled = cv.resize(img, (0, 0), fx=scale, fy=scale)
            num_rows = img_scaled.shape[0]
            num_cols = img_scaled.shape[1]

            # Generate the required number of negative examples for each rescaled image 
            for i in range(num_samples_per_image_scale):
                # Randomly select the top-left corner of the bounding box
                x = np.random.randint(low=0, high=num_cols - target_size[0])
                y = np.random.randint(low=0, high=num_rows - target_size[1])

                current_bbox = [x, y, x + target_size[0], y + target_size[1]]
                negative_example = img_scaled[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]]

                # Apply a color filter to detect regions that are likely to contain skin tones 
                skin_patch_hsv = cv.cvtColor(negative_example.copy(), cv.COLOR_BGR2HSV)
                skin_patch = cv.inRange(skin_patch_hsv, (0, 20, 70), (20, 255, 255))

                # Check if the skin area is significant enough 
                if np.mean(skin_patch) >= 100:
                    # Check if the current bounding box overlaps with detected faces
                    iou_with_faces = sum(intersection_over_union(current_bbox, np.array(bbox) * scale) for bbox in all_bboxes_file)

                    # Check for overlap with previously extracted negative samples
                    if len(boxes_for_samples) > 0:
                        iou_negative_samples = sum(intersection_over_union(current_bbox, np.array(bbox) * scale) for bbox in boxes_for_samples)
                    else:
                        iou_negative_samples = 0

                    # Save the negative example if it doesn't overlap with detected faces and has minimal overlap with other negative samples
                    if iou_with_faces == 0 and iou_negative_samples < 0.339:
                        cnt += 1  
                        print(cnt)

                        # Scale down the bounding box and store it to avoid duplicates
                        x_min = int(x / scale)
                        y_min = int(y / scale)
                        x_max = int((x + target_size[0]) / scale)
                        y_max = int((y + target_size[1]) / scale)
                        boxes_for_samples.append([x_min, y_min, x_max, y_max])
                       
                        # Save the negative example as an image file
                        filename = os.path.join(output_path, f"{cnt:07d}.jpg")
                        cv.imwrite(filename, negative_example)

    return cnt


# For each character, read its annotation file, take all bounding boxes and generate negative examples
for character in characters:
    # Get all bounding boxes from a character's annotation file
    bounding_boxes = read_bounding_boxes_from_file(f"{input_path}{character}_annotations.txt")
    images_folder_character = f"{input_path}{character}"

    # Generate negative examples that do not overlap with the bounding boxes that contain faces
    cnt = generate_random_neg_examples(images_folder_character, output_path, bounding_boxes, cnt, scales,
                                       target_size, num_samples_per_image)