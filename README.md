# Face Detection and Recognition of *The Flintstones* characters

This project implements a system for detecting and recognizing characters' faces from *"The Flintstones"* animated series using computer vision techniques.

## Objectives

The project focuses on two main tasks:

1. **Facial Detection**: detect all faces in the images
2. **Facial Recognition**: recognize the faces of the four main characters (Fred, Barney, Betty and Wilma)

## Data

The dataset consists of images divided into training, validation and testing sets. In the training set, for each of the four main characters (Fred, Barney, Betty and Wilma) is present one folder that contains 1000 images along with a specific annotation file. For each folder of images for a character, the corresponding annotation file contains the name of the image, the bounding box coordinates for a face and the character recognized.

- **Training Set**: 4000 images (1000 per character)
- **Validation Set**: 200 images
- **Test Set**: 200 images

## Choosing the Positive and Negative Examples for Training

The first step in solving the detection task was identifying the positive and negative examples to use for training the classifier, which will differentiate between faces and non-faces.

For both selecting positive and negative examples, I used the data available in the [`data/antrenare`](data/antrenare) directory, which contains training images for each of the four main characters (Fred, Wilma, Barney and Betty). There are 1,000 images per character, and each image is annotated with all the faces of the characters present.

### Color Filters

To improve the face detection process in the first task and to extract the examples used during training, I applied a strategy based on **color filters**. The goal of this approach is to isolate regions in an image that contain skin tones or shades close to skin color. This way, the area of interest (faces) is significantly reduced.

In the algorithm, I used two color filters based on different approaches.

The first filter is applied in the **HSV** (Hue, Saturation, Value) space, in order to highlight the regions in the original image with skin-like tones such as red, orange, and yellow. By selecting pixels with moderate to high saturation and brightness, it also adapts to various lighting conditions.

```python
skin_patch_hsv = cv.cvtColor(negative_example.copy(), cv.COLOR_BGR2HSV)
skin_patch = cv.inRange(skin_patch_hsv, (0, 20, 70), (20, 255, 255))
```

As seen in the image on the left, the filter is effective at detecting skin-colored regions. However, in some cases, the background may have shades close to skin color, leading to over-selection, as shown on the right.

The second filter is based on several results from papers on skin detection in images. It is more complex and uses both the RGB and HSV color spaces.

In the first stage, the color channels (blue, green, red) are extracted to analyze their distribution. Criteria for skin region detection under various lighting conditions were adapted from [this article](https://medium.com/swlh/human-skin-color-classification-using-the-threshold-classifier-rgb-ycbcr-hsv-python-code-d34d51febdf8), adjusting the thresholds to better match the training and validation images. In the second stage, I defined a mask in the HSV space that retains pixels with a hue value below 50 or above 150.

Since the first filter was sensitive to lighting variations, many faces in darker images were not detected. As seen in Figure 4, the second filter is more permissive than the first one, which is why I use it in the detection stage.

## Positive Examples

The set of positive examples includes **6920** images of size **64x64** that contain faces. The extraction process was done using the script `getPositiveExamples.py`, where I used the four `.txt` annotation  files to extract all the bounding boxes containing faces from each image.

To **label** each character from the positive examples and simplify the recognition task later, I introduced a digit-based encoding system:
- **Barney - 0**
- **Betty - 1**
- **Fred - 2**
- **Wilma - 3**
- **Unknown - 4** (faces that do not correspond to one of the four main characters)

Before saving the positive examples, I applied the second color filter mentioned earlier to maintain a dataset with specific properties. This process removed some faces belonging to unknown characters (green colored faces). Additionally, I manualy removed a few face images that were blurred to refine the dataset.

## Negative Examples

The set of negative examples was formed of **207906** images of size **64x64** that do not contain faces. The extraction process was done using the script `getNegativeExamples.py`, by randomly selecting regions from the training images. Additionally, I had to handle cases where these regions overlapped with the faces of the characters.

I scaled each image from the [`data/antrenare`](data/antrenare) directory to generate a diverse dataset. For **each scale**, I randomly selected a maximum of **40** negative examples. To ensure the regions selected mostly contained skin-like colors, I applied the first color filter described above, which is more restrictive. If the average value of the extracted patch after applying this filter exceeded a threshold, I checked if the patch **did not overlap with any detected faces** in the image and whether the sum of overlaps with already extracted negative examples was small enough. If these conditions were met, I saved the negative example.

## Determining Positive Feature Vectors

In the `get_positive_descriptors` method, found in both `FacialDetector.py` and `FacialRecognition.py`, I generated the positive descriptors needed to train classifiers specialized in facial detection and recognition. To expand the set of positive examples, in addition to **mirroring** the images (horizontal flip used in the lab), I introduced two additional augmentation techniques: **rotation** and **translation** of the images.

In the final version, I kept only **translations up, down, left, and right**, without including diagonal movements. For each augmented image, I applied the **Histogram of Oriented Gradients (HOG)** algorithm to extract the relevant features.

## Determining Negative Feature Vectors

In the `get_negative_descriptors` method, found in `FacialDetector.py`, I extracted the negative examples from the specified directory and stored their associated HOG descriptors without augmenting the dataset.

## Multi-scale Paradigm

To account for variability in distance from the *camera* (foreground) and to handle cases with faces that are either closer or farther away, I defined a vector containing values between **1.5** and **0.4**, which I used to resize the initial image. Using this paradigm, I generated a pyramid of resized images, as illustrated in the image below. This improved the classifier’s ability to detect faces in various conditions and distances.

## Sliding Window

To implement a solution based on the **sliding window** paradigm for face detection in this project, I used an approach that selects successive sections of the image with a window of size **64x64**. 

To determine a pattern for the shape and size of the bounding boxes used to detect character's faces, I calculated the **average height, width, and aspect ratio** for all the faces extracted as positive examples. Based on these results, I included a vector to resize the original image horizontally or vertically to simulate **rectangular bounding boxes**.

## Face Detection Procedure

### Step 1: Retrieving the Dataset Used for Model Training

As a dataset for training the detection classifier, I used the HOG feature vectors extracted, as detailed in sections [Determining Positive Feature Vectors](#determining-positive-feature-vectors) and [Determining Negative Feature Vectors](#determining-negative-feature-vectors).

### Step 2: Training the Classifier

For solving the detection task, I chose to use an **MLPClassifier (Multi-Layer Perceptron Classifier)**, imported from the `sklearn.neural_network` library. For the activation function, I used **ReLU (Rectified Linear Unit)**, knowing that it removes negative values, speeding up learning and reducing training time. As a solver, I chose **Adam** to optimize the weights. Based on past projects, I knew that a constant learning rate does not favor the model’s learning, so I set it to **adaptive**. The maximum number of epochs is set to **1500**, allowing the model to learn over a sufficient number of iterations. Additionally, I configured the model to stop training if it does not observe an improvement within **5** consecutive epochs.
The best model for detection is saved in the file [`mlp_detection_207906_207600.npy`](saved_files/models/mlp_detection_207906_207600.npy).

### Step 3: Face Detection

In the `run_detection` method implemented in the [`FacialDetector.py`](Cod_Sursa_Rincu_Stefania_332/FacialDetector.py) script, I process the test images one by one. The detection process is based on the **multi-scale sliding window** paradigm mentioned earlier. Each image is resized equally both horizontally and vertically to create a pyramid of images. However, to also detect rectangular windows, I make slight adjustments to the heights or widths separately (see sections [Multi-scale Paradigm](#multi-scale-paradigm) and [Sliding Window](#sliding-window)).

I computed the HOG descriptors for each region in the image and also extracted the corresponding patch from the color image. Then I applied the two color filters mentioned above in order to select only the regions that contain skin-tones. 

As a **confidence score**, the MLP classifier returns the logarithms of the predicted probabilities for each class, which are values between 0 and 1. These scores are returned as **negative values** (logarithmic). To determine if a patch is a face or non-face, I use this confidence score, choosing the class with the highest score. Additionally, to eliminate false positives, I introduced a threshold for the score.

Finally, the function returns the coordinates of the detections (positions of the bounding boxes for faces found in the image), the associated scores, and the names of the corresponding test image files.

## Face Recognition Procedure

### Step 1: Retrieving the Dataset Used for Model Training

The first step is similar to the one presented in section above. In addition to this, I implemented the `get_training_examples_each_character` function implemented in the [`FacialRecognition.py`](Cod_Sursa_Rincu_Stefania_332/FacialRecognition.py) script, which retrieves the datasets necessary for individually training the classifiers for each character (Fred, Barney, Betty and Wilma).

To determine the best distribution of the training data, I searched for the situations where there are the most frequent **confusions** between characters. For instance, I observed that Barney is often confused with Fred and Betty with Wilma.

To improve the model accuracy, I **adjusted the proportions** of the datasets used for training each classifier. Thus, I allocated more positive and negative examples to the characters that caused **significant confusions**. Additionally, I shuffled the feature vectors at every step to avoid always retrieving the same features in the same order.

### Step 2: Training the Classifiers

For solving the recognition task, I chose to use one model for each character. In the `train_mlp_recognition_each` function implemented in the [`FacialRecognition.py`](Cod_Sursa_Rincu_Stefania_332/FacialRecognition.py) script, I defined a basic **MLP (Multi-Layer Perceptron Classifier)** model, which I train using the datasets corresponding to each character.

The model parameters are almost identical to those used in the detection task, presented in section above. After the training, each model is saved with a file name indicating the character it was trained for.

### Step 3: Face Recognition

In the `run_recognition` function implemented in the [`FacialRecognition.py`](Cod_Sursa_Rincu_Stefania_332/FacialRecognition.py) script, the face recognition process is carried out based on the descriptors obtained in the previous face detection step. These descriptors are passed to each of the four specialized classifiers in order to identify the main characters: Barney, Betty, Fred and Wilma.

Based on the scores (probabilities) returned by each classifier, I implemented a **weighted voting system**. This determines which character has the highest confidence score, taking multiple results into account. If all four models classify a descriptor into class 0, it should indicate that the person is either unknown or part of the background.

The final results are organized and stored separately for each character.
