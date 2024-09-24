from FacialRecognition import *  # Import the facial recognition module
from FacialDetector import *  # Import the facial detection module
from Parameters import *  # Import the parameters module

# Initialize parameters
params: Parameters = Parameters()
params.dim_window = 64  # Window size - positive examples (faces) are 64x64 pixels
params.dim_hog_cell = 8  # Size of the HOG cell
params.overlap = 0.3  # Maximum overlap of 30% allowed between sliding windows

# Translation coordinates applied to face images (positive examples) during detection an recognition for data augmentation
params.coord_detection = [[-12.5, 0],  [12.5, 0], [0, 0], [0, 12.5], [0, -12.5]]  # Detection 
params.coord_recognition = [[-12.5, 0],  [12.5, 0], [0, 0], [0, 12.5], [0, -12.5]]  # Recognition 

# Rotation angles applied to face images in both detection and recognition for robustness
params.angles = [-12.5, 0, 12.5]

# Set the number of positive and negative examples
params.number_positive_examples_det = 6920 * len(params.angles) * len(params.coord_detection)  # For detection
params.number_positive_examples_rec = 6920 * len(params.angles) * len(params.coord_recognition) * 2  # For recognition
params.number_negative_examples = 207906  

params.threshold = -1e-11 # Confidence threshold for face detection (I am using log probabilities, which are negative)
params.use_flip_images = True  # Use flipped images to augment data

params.training = True  # Set to False if testing
params.training_detection = True  # Set to True if training detection
params.training_recognition = True  # Set to True for training recognition

# If flipped images are used, double the number of positive detection examples
if params.use_flip_images:
    params.number_positive_examples_det *= 2

# Initialize the facial detector and recognizer with the parameters
facial_detector: FacialDetector = FacialDetector(params)
facial_recognition: FacialRecognition = FacialRecognition(params)

# This block is skipped when testing
if params.training: 
    # Path to save/load negative feature descriptors
    negative_features_path = os.path.join(params.dir_save_training_descr,
                                          'negativeExamplesDescriptors_' + str(params.dim_hog_cell) + '_' +
                                          str(params.number_negative_examples) + '.npy')
                                          
    # Check if negative descriptors already exist
    if os.path.exists(negative_features_path):
        # Load negative descriptors
        negative_features = np.load(negative_features_path)
        print('Loaded negative feature descriptors')
    else:
        print('Building negative feature descriptors:')
        # Generate negative descriptors if not present
        negative_features = facial_detector.get_negative_descriptors()
        # Save the generated descriptors
        np.save(negative_features_path, negative_features)
        print(f'Saved negative feature descriptors to {negative_features_path}')

    # If training for detection is enabled
    if params.training_detection:
        print("DETECTION TRAINING")
        
        # Path to save/load positive feature descriptors and labels for detection
        positive_features_path_detection = os.path.join(params.dir_save_training_descr,
                                              'positiveExamplesDescriptors_' + str(params.dim_hog_cell) + '_' +
                                                        str(params.number_positive_examples_det) + '.npy')                     
        labels_path_detection = os.path.join(params.dir_save_training_descr, 'positiveExamplesLabels_' + str(params.dim_hog_cell) + '_' +
                                             str(params.number_positive_examples_det) + '.npy')   

        # Check if positive descriptors for detection already exist                              
        if os.path.exists(positive_features_path_detection):
            # Load positive descriptors and labels
            positive_features_det = np.load(positive_features_path_detection)
            labels_det = np.load(labels_path_detection)
            print('Detection -- Loaded positive descriptors and labels')
        else:
            print('Building positive feature descriptors for detection:')
            # Generate positive descriptors for detection if not present
            positive_features_det, labels_det = facial_detector.get_positive_descriptors()
            # Save the generated descriptors and labels
            np.save(positive_features_path_detection, positive_features_det)
            np.save(labels_path_detection, labels_det)
            print(f'Detection -- Saved positive descriptors and labels to {positive_features_path_detection}')

        # Combine positive and negative examples to train the MLP classifier for detection
        training_examples = np.concatenate((np.squeeze(positive_features_det), np.squeeze(negative_features)), axis=0)
        train_labels = np.concatenate((np.ones(params.number_positive_examples_det), np.zeros(negative_features.shape[0])))
        # Train the detection model using the combined training data
        facial_detector.train_mlp_detection(training_examples, train_labels)

    # If training for recognition is enabled
    if params.training_recognition:
        print("RECOGNITION TRAINING")
        
        # Path to save/load positive feature descriptors and labels for recognition
        positive_features_path_recognition = os.path.join(params.dir_save_training_descr,
                                                        'positiveExamplesDescriptors_Rec_' + str(params.dim_hog_cell) + '_' +
                                                        str(params.number_positive_examples_rec) + '.npy')      
        labels_path_recognition = os.path.join(params.dir_save_training_descr,
                                             'positiveExamplesLabels_Rec_' + str(params.dim_hog_cell) + '_' +
                                             str(params.number_positive_examples_rec) + '.npy')

        # Check if positive descriptors for recognition already exist                           
        if os.path.exists(positive_features_path_recognition):
            # Load positive descriptors and labels
            positive_features_rec = np.load(positive_features_path_recognition)
            labels_rec = np.load(labels_path_recognition)
            print('Recognition -- Loaded positive descriptors and labels')
        else:
            print('Building positive feature descriptors for recognition:')
            # Generate positive descriptors for recognition if not present
            positive_features_rec, labels_rec = facial_recognition.get_positive_descriptors()
            # Save the generated descriptors and labels
            np.save(positive_features_path_recognition, positive_features_rec)
            np.save(labels_path_recognition, labels_rec)
            print(f'Recognition -- Saved positive descriptors and labels to {positive_features_path_recognition}')

        # Get training examples for each character and train MLP models for each character
        dict_features_each, dict_labels_each = facial_recognition.get_training_examples_each_character(positive_features_rec, labels_rec, negative_features)
        for i in range(4):  # Barney, Betty, Fred, Wilma
            facial_recognition.train_mlp_recognition_each(dict_features_each[i], dict_labels_each[i], i)

# Load saved models for detection and recognition and start solving the tasks
facial_detector.get_mlp_detection()  # Load the detection model
for i in range(4):
    facial_recognition.get_mlp_recognition_each(i)  # Load recognition models for each character

# Perform face detection
detections, scores, file_names, descriptors = facial_detector.run_detection()

# Classify and recognize the detected faces using the recognition model
facial_recognition.run_recognition(detections, descriptors, file_names)

print("Finished! =)")