# Face Detection and Recognition of The Flintstones characters

This project implements a system for detecting and recognizing characters' faces from *"The Flintstones"* animated series using computer vision techniques.

## Objectives

The project focuses on two main tasks:

1. **Facial Detection**: detect all faces in the images
2. **Facial Recognition**: recognize the faces of the four main characters (Fred, Barney, Betty and Wilma)

## Data

The dataset, located in the [`data`](data) folder, is divided into training, validation and testing sets. In the training set, each of the four main characters (Fred, Barney, Betty and Wilma) has its own folder containing 1000 images, accompanied by a corresponding annotation file. Each line in the annotation file includes the image name, the bounding box coordinates of a detected face and the recognized character.

- **Training Set**: 4000 images (1000 per character)
- **Validation Set**: 200 images
- **Test Set**: 200 images

## Positive and Negative Examples for Training

The first step in solving the detection task involved identifying positive and negative examples for training the classifier to distinguish between faces and non-faces. To select these examples, I used the data available in the [`data/antrenare`](data/antrenare) directory.

### Color Filters

To improve the face detection process and extract the examples used during training, I applied a strategy based on **color filters**. This approach aims to isolate regions in an image that contain skin tones or shades close to skin color. This way, the area of interest (faces) is significantly reduced. In the algorithm, I implemented two color filters based on different approaches. 

The first filter is applied in the **HSV** (Hue, Saturation, Value) space, with ranges of [0, 20] for hue, [20, 255] for saturation and [70, 255] for brightness, to emphasize regions in the original image with skin-like tones, specifically red, orange, and yellow. By targeting pixels with moderate to high saturation and brightness, this filter effectively adapts to various lighting conditions.

The second filter is based on various studies on skin detection in images and employs both the RGB and HSV color spaces. Initially, the color channels (blue, green, red) are extracted to analyze their distribution, adapting skin region detection criteria for different lighting conditions based on insights from this article [this article](https://medium.com/swlh/human-skin-color-classification-using-the-threshold-classifier-rgb-ycbcr-hsv-python-code-d34d51febdf8). Thresholds were adjusted to align more closely with the training and validation images. In the second stage, a mask is defined in the HSV space, retaining pixels with a hue value below 50 or above 150.

Since the first filter was sensitive to lighting variations, many faces in darker images were not detected. The second filter is more permissive than the first one, which is why I use it in the detection stage.

### Positive Examples

The set of positive examples includes **6920** images of size **64x64** that contain faces. The extraction process was done using the script [`getPositiveExamples.py`](Cod_Sursa_Rincu_Stefania_332/getPositiveExamples.py), where I used the four `.txt` annotation  files to extract all the bounding boxes containing faces from each image.

To **label** each character from the positive examples and simplify the recognition task later, I introduced a digit-based encoding system:
- **Barney - 0**
- **Betty - 1**
- **Fred - 2**
- **Wilma - 3**
- **Unknown - 4** (faces that do not correspond to one of the four main characters)

Before saving the positive examples, I applied the second color filter mentioned earlier to ensure the dataset retained specific properties. This filtering process eliminated some faces belonging to unknown characters. Additionally, I manualy removed several blurred face images to further refine the dataset.

### Negative Examples

The set of negative examples was formed of **207906** images of size **64x64** that do not contain faces. The extraction process was done using the script [`getNegativeExamples.py`](Cod_Sursa_Rincu_Stefania_332/getNegativeExamples.py), by randomly selecting regions from the training images. 

I **scaled** each image from the [`data/antrenare`](data/antrenare) directory to create a diverse dataset. For each scale, I randomly selected up to **40** negative examples. To ensure that the selected regions contained skin-like colors, I applied the first, more restrictive color filter described above. If the average value of the extracted patch after applying this filter exceeded a certain threshold, I checked if the patch **did not overlap with any detected faces** in the image and that its overlap with already extracted negative examples was minimal.
