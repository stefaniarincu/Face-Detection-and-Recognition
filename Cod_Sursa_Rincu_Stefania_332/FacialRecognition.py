import random  
from Parameters import * # Import the parameters module
import numpy as np  
import glob  
import cv2 as cv  
import pickle  
from skimage.feature import hog 
from sklearn.neural_network import MLPClassifier  

class FacialRecognition:
    def __init__(self, params: Parameters):
        self.params = params
        # Initialize placeholders for the best models for each character
        self.best_model_recognition_0 = None
        self.best_model_recognition_1 = None
        self.best_model_recognition_2 = None
        self.best_model_recognition_3 = None

    # Function that extracts the feature vectors for positive examples    
    def get_positive_descriptors(self):
        positive_descriptors, labels = [], []

        # Load positive example images
        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        random.shuffle(files)
        num_images = len(files)

        for i in range(num_images):
            print(f'Recognitie -- Procesam exemplul pozitiv numarul {i}...')

            # Read the image in grayscale and create a flipped version
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            img_flipped = np.fliplr(img).copy()

            # Get the image dimensions and calculate the center for rotation
            width, height = np.shape(img)
            center = (width // 2, height // 2)

            # Apply transformations (rotation, translation) to both original and flipped images
            for current_image in [img, img_flipped]:
                for angle in self.params.angles:
                    # Rotate the image by a specified angle
                    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
                    rotated_image = cv.warpAffine(current_image, rotation_matrix, (width, height), borderMode=cv.BORDER_REPLICATE)

                    # Translate the image based on the predefined coordinates
                    for c in self.params.coord_recognition:
                        translation_matrix = np.float32([[1, 0, c[0]], [0, 1, c[1]]])
                        translated_image = cv.warpAffine(rotated_image, translation_matrix, (width, height),
                                                         borderMode=cv.BORDER_REPLICATE)

                        # Extract HOG features from the translated image
                        features = hog(translated_image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                       cells_per_block=(2, 2), feature_vector=True)
                        # Append the features and corresponding label
                        positive_descriptors.append(features)
                        labels.append(int(files[i][-10]))  # Assuming label is part of the filename

        # Return positive feature descriptors and their associated labels
        return np.array(positive_descriptors), np.array(labels)

    # Function that prepares training examples for each character using both positive and negative features
    def get_training_examples_each_character(self, positive_features, labels, negative_features):
        print("Recognition -- creating training examples for each character")

        # Initialize dictionaries to store feature vectors and labels for each character
        dict_features_each_character = {0: [], 1: [], 2: [], 3: [], 4: []}
        dict_train_each = {0: [], 1: [], 2: [], 3: []}
        dict_labels_each = {0: [], 1: [], 2: [], 3: []}

        # Categorize features by the label (character) they belong to
        for i in range(len(labels)):
            val = labels[i]
            if val in dict_features_each_character:
                dict_features_each_character[val].append(positive_features[i])

        # Prepare training sets for each character by combining a mix of positive and negative examples
        # Barney (label 0) 
        barney_faces_nr = len(dict_features_each_character[0])

        random.shuffle(dict_features_each_character[1])
        random.shuffle(dict_features_each_character[2])
        random.shuffle(dict_features_each_character[4])
        random.shuffle(negative_features)

        dict_train_each[0] = np.concatenate((np.squeeze(dict_features_each_character[0]),
                                             np.squeeze(dict_features_each_character[2][:int(barney_faces_nr * 0.4)]),
                                             np.squeeze(dict_features_each_character[1][:int(barney_faces_nr * 0.1)]),
                                             np.squeeze(dict_features_each_character[4][:int(barney_faces_nr * 0.2)]),
                                             np.squeeze(negative_features[:int(barney_faces_nr * 0.3)])), axis=0)
        dict_labels_each[0] = np.concatenate(
            (np.ones(barney_faces_nr), np.zeros(int(barney_faces_nr * 0.4) + int(barney_faces_nr * 0.1) +
                                               int(barney_faces_nr * 0.2) + int(barney_faces_nr * 0.3))))

        # Betty (label 1)
        betty_faces_nr = len(dict_features_each_character[1])

        random.shuffle(dict_features_each_character[3])
        random.shuffle(dict_features_each_character[4])
        random.shuffle(negative_features)
        
        dict_train_each[1] = np.concatenate((np.squeeze(dict_features_each_character[1]),
                                             np.squeeze(dict_features_each_character[3][:int(betty_faces_nr * 0.4)]),
                                             np.squeeze(dict_features_each_character[4][:int(betty_faces_nr * 0.4)]),
                                             np.squeeze(negative_features[:int(betty_faces_nr * 0.2)])), axis=0)
        dict_labels_each[1] = np.concatenate(
            (np.ones(betty_faces_nr), np.zeros(int(betty_faces_nr * 0.4) + int(betty_faces_nr * 0.4) +
                                               int(betty_faces_nr * 0.2))))

        # Fred (label 2)
        fred_faces_nr = len(dict_features_each_character[2])

        random.shuffle(dict_features_each_character[3])
        random.shuffle(dict_features_each_character[4])
        random.shuffle(dict_features_each_character[0])
        random.shuffle(negative_features)
        
        dict_train_each[2] = np.concatenate((np.squeeze(dict_features_each_character[2]),
                                             np.squeeze(dict_features_each_character[4][:int(fred_faces_nr * 0.3)]),
                                             np.squeeze(dict_features_each_character[0][:int(fred_faces_nr * 0.4)]),
                                             np.squeeze(negative_features[:int(fred_faces_nr * 0.3)])), axis=0)
        dict_labels_each[2] = np.concatenate(
            (np.ones(fred_faces_nr), np.zeros(int(fred_faces_nr * 0.3) + int(fred_faces_nr * 0.4) +
                                              int(fred_faces_nr * 0.3))))

        # Wilma (label 3)
        wilma_faces_nr = len(dict_features_each_character[3])

        random.shuffle(dict_features_each_character[2])
        random.shuffle(dict_features_each_character[1])
        random.shuffle(dict_features_each_character[4])
        random.shuffle(negative_features)

        dict_train_each[3] = np.concatenate((np.squeeze(dict_features_each_character[3]),
                                             np.squeeze(dict_features_each_character[2][:int(wilma_faces_nr * 0.3)]),
                                             np.squeeze(dict_features_each_character[1][:int(wilma_faces_nr * 0.3)]),
                                             np.squeeze(dict_features_each_character[4][:int(wilma_faces_nr * 0.2)]),
                                             np.squeeze(negative_features[:int(wilma_faces_nr * 0.2)])), axis=0)
        dict_labels_each[3] = np.concatenate(
            (np.ones(wilma_faces_nr), np.zeros(int(wilma_faces_nr * 0.3) + int(wilma_faces_nr * 0.3) +
                                               int(wilma_faces_nr * 0.2) + int(wilma_faces_nr * 0.2))))

        # Return the training sets and corresponding labels for each character
        return dict_train_each, dict_labels_each

    # Function that trains a MLP classifier for a given character
    def train_mlp_recognition_each(self, train_examples, train_labels, character):
        # Path to save the trained MLP classifier for the specified character
        file_mlp_recognition = os.path.join(self.params.dir_save_models, f'mlp_recognition_{character}')

        # Initialize the MLP classifier 
        model = MLPClassifier(activation='relu', solver='adam', learning_rate='adaptive', max_iter=1000,
                              verbose=True, warm_start=True, n_iter_no_change=5)

        # Train the model on the given examples and labels
        model.fit(train_examples, train_labels)
        # Save the model 
        pickle.dump(model, open(file_mlp_recognition, 'wb'))

        # Print model statistics
        print(f"Statistics of the model for character -- {character}:")
        score = model.score(train_examples, train_labels)
        print(score)

        # Save the model as the best model for the specified character
        if character == 0:
            self.best_model_recognition_0 = model
        elif character == 1:
            self.best_model_recognition_1 = model
        elif character == 2:
            self.best_model_recognition_2 = model
        elif character == 3:
            self.best_model_recognition_3 = model

        # Predict the labels for the training examples to test the model
        predicted_labels = model.predict(train_examples)

        # Compute the confusion matrix in order to observes which characters are often misclassified
        conf_mat = np.zeros((2, 2))
        for i in range(len(predicted_labels)):
            conf_mat[int(train_labels[i]), int(predicted_labels[i])] += 1

        print(conf_mat)

    # Function that loads the best MLP model for a given character
    def get_mlp_recognition_each(self, character):
        # Path to the saved MLP classifier for the specified character
        file_mlp_recognition = os.path.join(self.params.dir_save_models, f'mlp_recognition_{character}')

        # Check if the model file exists
        if os.path.exists(file_mlp_recognition):
            # Load the saved model based on the character ID 
            if character == 0:  # Barney
                self.best_model_recognition_0 = pickle.load(open(file_mlp_recognition, 'rb'))
                return
            elif character == 1:  # Betty
                self.best_model_recognition_1 = pickle.load(open(file_mlp_recognition, 'rb'))
                return
            elif character == 2:  # Fred
                self.best_model_recognition_2 = pickle.load(open(file_mlp_recognition, 'rb'))
                return
            else:  # Wilma
                self.best_model_recognition_3 = pickle.load(open(file_mlp_recognition, 'rb'))
                return

    # Function that performs the facial recognition task using the loaded models and input descriptors
    def run_recognition(self, detections, descriptors, file_names):
        print("Passed to Face Recognition") 

        # Initialize lists to store detection results (coordinates, scores, filename) for each character
        barney_all = [[], [], []]  
        betty_all = [[], [], []]  
        fred_all = [[], [], []]   
        wilma_all = [[], [], []]  

        # Loop over all the descriptors (features) passed for recognition
        for i in range(len(descriptors)):
            # Retrieve the log-probability predictions for each model (Barney, Betty, Fred, Wilma)
            pred_class_0 = self.best_model_recognition_0.predict_log_proba([descriptors[i]])[0]
            pred_class_1 = self.best_model_recognition_1.predict_log_proba([descriptors[i]])[0]
            pred_class_2 = self.best_model_recognition_2.predict_log_proba([descriptors[i]])[0]
            pred_class_3 = self.best_model_recognition_3.predict_log_proba([descriptors[i]])[0]

            # Convert log-probabilities to actual probabilities and create a prediction list
            predictions = [np.argmax(pred_class_0) * np.exp(np.max(pred_class_0)), 
                        np.argmax(pred_class_1) * np.exp(np.max(pred_class_1)),
                        np.argmax(pred_class_2) * np.exp(np.max(pred_class_2)), 
                        np.argmax(pred_class_3) * np.exp(np.max(pred_class_3))]

            # If the sum of the predictions is greater than zero, get the highest probability prediction
            if sum(predictions) > 0:
                final_prediction_index = np.argmax(predictions)  # Determine which character is most likely
            else:
                final_prediction_index = 4  # If no prediction is strong enough, assign consider it character 'Unknown' 

            # Depending on the final prediction, add results to the corresponding character's list
            if final_prediction_index == 0: 
                barney_all[0].append(detections[i])
                barney_all[1].append(np.max(pred_class_0))
                barney_all[2].append(file_names[i])
            elif final_prediction_index == 1:
                betty_all[0].append(detections[i])
                betty_all[1].append(np.max(pred_class_1))
                betty_all[2].append(file_names[i])
            elif final_prediction_index == 2:
                fred_all[0].append(detections[i])
                fred_all[1].append(np.max(pred_class_2))
                fred_all[2].append(file_names[i])
            elif final_prediction_index == 3:
                wilma_all[0].append(detections[i])
                wilma_all[1].append(np.max(pred_class_3))
                wilma_all[2].append(file_names[i])

        # Save the results (detections, scores, filenames) for each character into .npy files
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'detections_barney.npy'), barney_all[0])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'scores_barney.npy'), barney_all[1])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'file_names_barney.npy'), barney_all[2])

        np.save(os.path.join(self.params.dir_sol_task2_folder, 'detections_betty.npy'), betty_all[0])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'scores_betty.npy'), betty_all[1])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'file_names_betty.npy'), betty_all[2])

        np.save(os.path.join(self.params.dir_sol_task2_folder, 'detections_fred.npy'), fred_all[0])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'scores_fred.npy'), fred_all[1])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'file_names_fred.npy'), fred_all[2])

        np.save(os.path.join(self.params.dir_sol_task2_folder, 'detections_wilma.npy'), wilma_all[0])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'scores_wilma.npy'), wilma_all[1])
        np.save(os.path.join(self.params.dir_sol_task2_folder, 'file_names_wilma.npy'), wilma_all[2])
