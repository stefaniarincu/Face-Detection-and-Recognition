import cv2 as cv
import os
import numpy as np

# The source path for the training images, from which positive examples will be extracted
input_path = "./data/antrenare/"
# Path where the positive examples will be stored (image dimension: 64x64)
path_positives = "./data/antrenare/positiveExamples64/"

# Check if the directory for positive examples exists; if not, create it
if not os.path.exists(path_positives):
    os.makedirs(path_positives)

characters = ["barney", "betty", "fred", "wilma"]
# Mapping each character name to a numeric label in order to store the positive examples with the appropriate label
dict_characters = {"barney": 0, "betty": 1, "fred": 2, "wilma": 3, "unknown": 4}

cnt = 0  # Initialize a counter to track the number of positive examples processed

# Iterate through each character in the list
for character in characters:
    # Open the corresponding annotation file for the character
    with open(f"{input_path}{character}_annotations.txt", 'r') as file:
        for line in file:
            # Extract the file name, bounding box coordinates and character name 
            file_name, x_min, y_min, x_max, y_max, who = line.strip().split(" ")
            image_path = os.path.join(f"{input_path}{character}", file_name)  

            image = cv.imread(image_path)  

            if image is not None:
                # Extract the face based on the bounding box and resize it to 64x64 pixels
                extracted_rectangle = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                resized_face = cv.resize(extracted_rectangle, (64, 64))

                # Split the image into its blue, green and red color channels in order to apply a filter for skin detection
                blue_frame, green_frame, red_frame = cv.split(resized_face.copy())

                # Find the maximum and minimum intensity values among the three channels
                bgr_max = np.maximum.reduce([blue_frame, green_frame, red_frame])
                bgr_min = np.minimum.reduce([blue_frame, green_frame, red_frame])

                # Define conditions for detecting skin tones under normal lighting conditions
                case_normal_light = np.logical_and.reduce([red_frame > 75, green_frame > 40, blue_frame > 20,
                                                           bgr_max - bgr_min > 5, abs(red_frame - green_frame) > 5,
                                                           red_frame > green_frame, red_frame > blue_frame])
                # Define conditions for detecting skin tones under artificial lighting conditions
                case_artificial_light = np.logical_and.reduce([red_frame > 220, green_frame > 210, blue_frame > 170,
                                                               abs(red_frame - green_frame) <= 15,
                                                               red_frame > blue_frame,
                                                               green_frame > blue_frame])

                # Combine the results of normal and artificial light detection to create an RGB skin mask
                rgb_mask = np.logical_or(case_normal_light, case_artificial_light)

                # Create an HSV mask for detecting skin tones based on the hue values
                hsv_frame = cv.cvtColor(resized_face.copy(), cv.COLOR_BGR2HSV)
                hsv_mask = np.logical_or(hsv_frame[..., 0] < 50, hsv_frame[..., 0] > 150)

                # Combine the RGB and HSV masks to form a final mask for skin detection
                final_skin_mask = np.logical_and.reduce([rgb_mask, hsv_mask]).astype(np.uint8) * 255

                # If the skin mask contains a sufficient amount of skin pixels, save the image
                if np.mean(final_skin_mask) > 50:
                    cnt += 1 # Increment the counter for positive examples
                    print(cnt)

                    # Generate a unique file name for the positive example using the character's label
                    filename = f"{dict_characters.get(who)}_{cnt:04d}.jpg"

                    # Save the resized face to the specified directory
                    output_path = os.path.join(f"{path_positives}", f"{filename}")
                    cv.imwrite(output_path, resized_face)