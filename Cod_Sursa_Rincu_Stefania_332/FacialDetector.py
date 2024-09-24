import random
from Parameters import *
import numpy as np
import glob
import cv2 as cv
import pickle
import ntpath
import timeit
from skimage.feature import hog
from sklearn.neural_network import MLPClassifier

class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        # Inialize placeholder for the best model for the detection task
        self.best_model_detection = None

    # Function that extracts the feature vectors for positive examples
    def get_positive_descriptors(self):
        positive_descriptors, labels = [], []
        cnt = 0

        # Load positive example images
        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        random.shuffle(files)
        num_images = len(files)

        for i in range(num_images):
            print('Detection task -- Processing positive example number %d...' % cnt)
            cnt += 1

            print(files[i])
            # Read image in grayscale
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            # Get the image dimensions and calculate the center for rotation
            width, height = np.shape(img)
            center = (width // 2, height // 2)

            # Apply transformations (rotation, translation) to the face image to augment the dataset of positive examples
            for angle in self.params.angles:
                # Rotate the image by a specified angle
                rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
                rotated_image = cv.warpAffine(img.copy(), rotation_matrix, (width, height), borderMode=cv.BORDER_REPLICATE)

                # Translate the image based on the predefined coordinates
                for c in self.params.coord_detection:
                    translation_matrix = np.float32([[1, 0, c[0]], [0, 1, c[1]]])
                    translated_image = cv.warpAffine(rotated_image, translation_matrix, (width, height),
                                                     borderMode=cv.BORDER_REPLICATE)

                    # Extract HOG features from the translated image
                    features = hog(translated_image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                   cells_per_block=(2, 2), feature_vector=True)
                    # Append the features and corresponding label
                    positive_descriptors.append(features)
                    labels.append(int(files[i][-10]))

            # If flipping images is enabled, process the flipped version as well
            if self.params.use_flip_images:
                img_flipped = np.fliplr(img).copy()

                for angle in self.params.angles:
                    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
                    rotated_image = cv.warpAffine(img_flipped.copy(), rotation_matrix, (width, height),
                                                  borderMode=cv.BORDER_REPLICATE)

                    for c in self.params.coord_detection:
                        translation_matrix = np.float32([[1, 0, c[0]], [0, 1, c[1]]])
                        translated_image = cv.warpAffine(rotated_image, translation_matrix, (width, height),
                                                         borderMode=cv.BORDER_REPLICATE)

                        features = hog(translated_image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                       cells_per_block=(2, 2), feature_vector=True)
                        positive_descriptors.append(features)
                        labels.append(int(files[i][-10]))

        # Return positive feature descriptors and their associated labels
        return np.array(positive_descriptors), np.array(labels)

    # Function that creates the feature vectors for negative examples
    def get_negative_descriptors(self):
        negative_descriptors = []

        # Load negative example images
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        random.shuffle(files)

        for i in range(self.params.number_negative_examples):
            print('Processing negative example number %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            # Extract HOG features for negative examples
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(features)

        return np.array(negative_descriptors)

    # Function that trains the MLP model for the detection task
    def train_mlp_detection(self, training_examples, train_labels):
        # Path to save the trained MLP classifier for detection
        file_mlp_detection = os.path.join(self.params.dir_save_files, 'mlp_detection_%d_%d' %
                                           (self.params.number_negative_examples, self.params.number_positive_examples_det))

        # Initialize the MLP classifier for detection
        model = MLPClassifier(activation='relu', solver='adam', learning_rate='adaptive', max_iter=1500,
                              verbose=True, warm_start=True, n_iter_no_change=5)

        # Fit the model 
        model.fit(training_examples, train_labels)
        # Save the model
        pickle.dump(model, open(file_mlp_detection, 'wb'))

        # Evaluate the model
        score = model.score(training_examples, train_labels)
        print(score)

        self.best_model_detection = model

        # Predict labels and build the confusion matrix
        predict_labels = model.predict(training_examples)

        conf_mat = np.zeros((2, 2))
        for i in range(len(predict_labels)):
            conf_mat[int(train_labels[i]), int(predict_labels[i])] += 1

        print(conf_mat)

    # Function that loads the trained MLP model if it exists
    def get_mlp_detection(self):
        file_mlp_detection = os.path.join(self.params.dir_save_files, 'mlp_detection_%d_%d' %
                                           (self.params.number_negative_examples, self.params.number_positive_examples_det))

        if os.path.exists(file_mlp_detection):
            self.best_model_detection = pickle.load(open(file_mlp_detection, 'rb'))
            return

    # Function that computes the Intersection over Union (IoU) for two bounding boxes
    def intersection_over_union(self, bbox_a, bbox_b):
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

    # Function that performs Non-Maximum Suppression to filter overlapping detections
    def non_maximal_suppression(self, image_detections, image_scores, image_descriptors, image_size):
        # Adjust detections that go out of image bounds
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)

        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]

        # Sort detections by their scores
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]
        sorted_descriptors = image_descriptors[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0

        # Remove redundant detections based on IoU threshold
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i]:
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j]:
                        if self.intersection_over_union(sorted_image_detections[i], sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_descriptors[is_maximal]

    # Main function to run the detection task
    def run_detection(self):
        # Load images
        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)

        detections = None # List of ounding box coordinates
        scores = np.array([])  # List of confidence scores
        file_names = np.array([]) # List of file names

        num_test_images = len(test_files)
        descriptors_to_return = []

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Processing test image %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i])
            img_gray = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)

            image_scores = [] # Confidence scores for each detection in the current image
            image_detections = [] # Bounding box coordinates for each detection in the current image
            image_descriptors = [] # Feature vectors for each detection in the current image

            # Define scales for image resizing in order to use a multiscale sliding window approach
            scales = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

            for scale in scales:
                # Iterate over a list of x and y scale pairs, representing different aspect ratios for the sliding window
                for xy_scales in [[0.81, 1], [0.96, 1], [1.08, 1], [1.25, 1], [1, 1.22], [1, 1.04], [1, 0.92], [1, 0.8]]:
                    # Resize image 
                    img_scaled = cv.resize(img, (0, 0), fx=scale * xy_scales[0], fy=scale * xy_scales[1])
                    img_scaled_gray = cv.resize(img_gray, (0, 0), fx=scale * xy_scales[0], fy=scale * xy_scales[1])

                    # Extract HOG features 
                    hog_descriptors = hog(img_scaled_gray, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                          cells_per_block=(2, 2), feature_vector=False)

                    num_cols = img_scaled_gray.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = img_scaled_gray.shape[0] // self.params.dim_hog_cell - 1
                    num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                    for y in range(0, num_rows - num_cell_in_template):
                        for x in range(0, num_cols - num_cell_in_template):
                            # Determine the feature vector for the current patch
                            descriptor = hog_descriptors[y: y + num_cell_in_template, x: x + num_cell_in_template].flatten()

                            # Save the current patch in order to apply color filtering for skin detection
                            window_patch = img_scaled[y * self.params.dim_hog_cell: y * self.params.dim_hog_cell + self.params.dim_window,
                                                      x * self.params.dim_hog_cell: x * self.params.dim_hog_cell + self.params.dim_window]
                            
                            # Split the saved patch into its blue, green and red color channels 
                            blue_frame, green_frame, red_frame = cv.split(window_patch.copy())

                            # Find the maximum and minimum intensity values among the three channels
                            bgr_max = np.maximum.reduce([blue_frame, green_frame, red_frame])
                            bgr_min = np.minimum.reduce([blue_frame, green_frame, red_frame])

                            # Define conditions for detecting skin tones under normal lighting conditions
                            case_normal_light = np.logical_and.reduce(
                                [red_frame > 75, green_frame > 40, blue_frame > 20,
                                 bgr_max - bgr_min > 5, abs(red_frame - green_frame) > 5,
                                 red_frame > green_frame, red_frame > blue_frame])
                            # Define conditions for detecting skin tones under artificial lighting conditions
                            case_artificial_light = np.logical_and.reduce(
                                [red_frame > 220, green_frame > 210, blue_frame > 170,
                                 abs(red_frame - green_frame) <= 15,
                                 red_frame > blue_frame,
                                 green_frame > blue_frame])

                            # Combine the results of normal and artificial light detection to create an RGB skin mask
                            rgb_mask = np.logical_or(case_normal_light, case_artificial_light)

                            # Create an HSV mask for detecting skin tones based on the hue values
                            hsv_frame = cv.cvtColor(window_patch.copy(), cv.COLOR_BGR2HSV)
                            hsv_mask = np.logical_or(hsv_frame[..., 0] < 50, hsv_frame[..., 0] > 150)

                            # Combine the RGB and HSV masks to form a final mask for skin detection
                            final_skin_mask = np.logical_and.reduce([rgb_mask, hsv_mask]).astype(np.uint8) * 255

                            # If the average skin mask value is greater than 50, pass the feature vector to the classifier
                            if np.mean(final_skin_mask) > 50:
                                predictions = self.best_model_detection.predict_log_proba([descriptor])[0]
                                prediction = np.argmax(predictions) # Class label
                                score = np.max(predictions) # Confidence score

                                # If classified as a face with a confidence score above a given threshold, save the coordinates of the bounding box
                                if prediction == 1 and score >= self.params.threshold:
                                    # Convert coordinates back to original scale
                                    x_min = int((x * self.params.dim_hog_cell) / scale / xy_scales[0])
                                    y_min = int((y * self.params.dim_hog_cell) / scale / xy_scales[1])
                                    x_max = int((x * self.params.dim_hog_cell + self.params.dim_window) / scale / xy_scales[0])
                                    y_max = int((y * self.params.dim_hog_cell + self.params.dim_window) / scale / xy_scales[1])

                                    # Save the bounding box coordinates, confidence score and feature vector
                                    image_detections.append([x_min, y_min, x_max, y_max])
                                    image_scores.append(score)
                                    image_descriptors.append(descriptor)

            # Apply Non-Maximum Suppression if there are detections in the current image
            if len(image_scores) > 0:
                image_detections, image_scores, image_descriptors = self.non_maximal_suppression(
                    np.array(image_detections), np.array(image_scores), np.array(image_descriptors), img.shape)
            if len(image_scores) > 0:
                # Save the data for all detected faces in the current image
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))

                scores = np.append(scores, image_scores)
                short_name = ntpath.basename(test_files[i])
                image_names = [short_name for _ in range(len(image_scores))]
                file_names = np.append(file_names, image_names)

                for d in image_descriptors:
                    descriptors_to_return.append(d)

            end_time = timeit.default_timer()
            print('Processing time for test image %d/%d is %f sec.'
                  % (i, num_test_images, end_time - start_time))

        # Save the results to .npy files
        np.save(os.path.join(self.params.dir_sol_task1_folder, 'detections_all_faces.npy'), detections)
        np.save(os.path.join(self.params.dir_sol_task1_folder, 'scores_all_faces.npy'), scores)
        np.save(os.path.join(self.params.dir_sol_task1_folder, 'file_names_all_faces.npy'), file_names)

        return detections, scores, file_names, descriptors_to_return