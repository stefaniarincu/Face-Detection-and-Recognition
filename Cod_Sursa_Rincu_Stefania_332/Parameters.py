import os

class Parameters:
    def __init__(self):
        # Define the base directory where all the files are located
        self.base_dir = './'
        
        # Paths for the training (positive and negative examples) and test images
        self.dir_pos_examples = os.path.join(self.base_dir, 'data/antrenare/positiveExamples64')  # Path for positive training examples (64x64 pixel images)
        self.dir_neg_examples = os.path.join(self.base_dir, 'data/antrenare/negativeExamples64')  # Path for negative training examples (64x64 pixel images)
        self.dir_test_examples = os.path.join(self.base_dir, 'data/testare/testare')  # Path for test images
        # self.dir_test_examples = os.path.join(self.base_dir, 'data/validare/validare')  # Path for validation images
        self.path_annotations = os.path.join(self.base_dir, 'data/testare/ground-truth-test') # Path to the correct solution files for test dataset in order to test the performance of the algorithm
        # self.path_annotations = os.path.join(self.base_dir, 'data/validare/ground-truth-validare') # Path to the correct solution files for validation dataset in order to test the performance of the algorithm

        # Path for saving files
        self.dir_save_files = './saved_files'  # Relative path for saving files
        self.dir_save_training_descr =  os.path.join(self.dir_save_files, "descriptors")  # Directory for saving descriptors and labels during training
        self.dir_save_models =  os.path.join(self.dir_save_files, "models")  # Directory for models for detection and recognition tasks
        self.dir_sol_task1_folder = 'evaluare/fisiere_solutie/332_Rincu_Stefania/task1_test/'  # Directory for solution files for task 1 for test dataset
        self.dir_sol_task2_folder = 'evaluare/fisiere_solutie/332_Rincu_Stefania/task2_test/'  # Directory for solution files for task 2 for test dataset

        # self.dir_sol_task1_folder = 'evaluare/fisiere_solutie/332_Rincu_Stefania/task1_validare/'  # Directory for solution files for task 1 for validation dataset
        # self.dir_sol_task2_folder = 'evaluare/fisiere_solutie/332_Rincu_Stefania/task2_validare/'  # Directory for solution files for task 2 for validation dataset
        
        # Ensure the save directory exists; if not, create it
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # Ensure the directory for saving descriptors and labels during training exists; if not, create it
        if not os.path.exists(self.dir_save_training_descr):
            os.makedirs(self.dir_save_training_descr)
            print('directory created: {} '.format(self.dir_save_training_descr))
        else:
            print('directory {} exists '.format(self.dir_save_training_descr))

        # Ensure the directory for saving models during training exists; if not, create it
        if not os.path.exists(self.dir_save_models):
            os.makedirs(self.dir_save_models)
            print('directory created: {} '.format(self.dir_save_models))
        else:
            print('directory {} exists '.format(self.dir_save_models))

        # Ensure the solution directory for task 1 exists; if not, create it
        if not os.path.exists(self.dir_sol_task1_folder):
            os.makedirs(self.dir_sol_task1_folder)
            print('directory created: {} '.format(self.dir_sol_task1_folder))
        else:
            print('directory {} exists '.format(self.dir_sol_task1_folder))
        
        # Ensure the solution directory for task 2 exists; if not, create it
        if not os.path.exists(self.dir_sol_task2_folder):
            os.makedirs(self.dir_sol_task2_folder)
            print('directory created: {} '.format(self.dir_sol_task2_folder))
        else:
            print('directory {} exists '.format(self.dir_sol_task2_folder))
        
        # Window size for the training examples (positive examples are 64x64 pixels)
        self.dim_window = 64

        # HOG (Histogram of Oriented Gradients) parameters
        self.dim_hog_cell = 8  # Size of the HOG cell
        self.dim_descriptor_cell = 64  # Descriptor size for each HOG cell
        
        # Maximum permited overlap ratio for the sliding window approach (30% overlap)
        self.overlap = 0.3
        
        # Number of examples used for detection and recognition tasks
        self.number_positive_examples_det = 6920  # Number of positive examples for detection
        self.number_positive_examples_rec = 6920  # Number of positive examples for recognition
        self.number_negative_examples = 207906  # Number of negative examples

        # Option to enable the use of horizontally flipped face images to enhance training diversity
        self.use_flip_images = True

        # Confidence threshold for face detection (I am using log probabilities, which are negative)
        self.threshold = -1e-11 
        
        # Translation coordinates applied to face images (positive examples) during detection for data augmentation
        self.coord_detection = [[-12.5, 0], [12.5, 0], [0, 0], [0, 12.5], [0, -12.5]]
        
        # Translation coordinates applied to face images during (positive examples) recognition for further data augmentation
        self.coord_recognition = [[-12.5, 0], [12.5, 0], [0, 0], [0, 12.5], [0, -12.5], 
                                  [-12.5, 12.5], [12.5, 12.5], [-12.5, -12.5], [12.5, -12.5]]
        
        # Rotation angles applied to face images in both detection and recognition for robustness
        self.angles = [-12.5, 0, 12.5]

        self.training = True  # Set to False if testing
        self.training_detection = True  # Set to True if training for detection task
        self.training_recognition = True  # Set to True if training for recognition task