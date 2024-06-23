# EmoSense

EmoSense is a real-time emotion detection tool that can recognize emotions from live video feeds, images, and videos. The project utilizes `OpenCV` for capturing and processing frames, creating the user interface, and drawing on faces, `Haar cascades` to efficiently detect and extract faces and a TensorFlow-trained `CNN model` to detect facial emotions with high accuracy.

<p align="center">
  <img src=./output_samples/team0_result.jpg alt="team img"/>
</p>

# Installation

### Prerequisites

- Python 3.6 or higher
- `pip` (Python package installer)

### Setup Instructions

#### 1. Clone the repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/Emo-Sense/EmoSense.git

```
Navigate to the EmoSense folder:
```sh
cd EmoSense
```

#### 2. Create a virtual environment (Optional but Recommended)

It is recommended to create a virtual environment to keep dependencies isolated:
```sh
python -m venv env
```

#### 3. Activate the virtual environment (Only if you create virtual environment)

```sh
# On Windows
.\env\Scripts\activate

# On macOS and Linux
source env/bin/activate
```

#### 4. Install the requirements

```sh
pip install -r requirements.txt
```

#### 5. Download the model

The best model exceeds the GiHub storage limit so it is stored in a Google Drive folder. You can download and move it to the `models/` folder using :

```sh
gdown https://drive.google.com/uc?id=1NnJ5ObGiN-_LbTUohvjxES3koHjPgxsH -O models/emosense_finalmodel_82.h5
```

# Usage

### Real-time Emotion Detection
To start real-time emotion detection using your webcam, run the following command:

```sh
python recognition.py
```
### Emotion Detection in Images
To detect emotions in a specific image, use the following command:

```sh
python recognition.py path/to/your/image.jpg
```
### Emotion Detection in Videos
To detect emotions in a specific video, use the following command:

```sh
python recognition.py path/to/your/video.mp4
```


# Model Description

## Data Preparation
The project utilizes the [FER 2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset with the Microsoft released [FER +](https://github.com/microsoft/FERPlus) labels.

The FER 2013 dataset comprises 35,887 grayscale images, each with a resolution of 48x48 pixels, and originally labeled with one of seven emotions: neutral, happiness, surprise, sadness, anger, disgust, and fear. The FER+, an enhanced version of FER 2013, addresses labeling errors present in the original dataset, providing a more accurate representation of the emotions. 

The FER+ dataset also includes additional labels for contempt, unknown, and NF, which are not part of the original seven emotions. These labels were removed from both the image data and the labels to focus exclusively on the seven primary emotions.

## Model Architecture
The CNN architecture that was used as the design to model the Neural network for the FER2013 dataset was VGG-19.
 
**Input Layer:** The default or standard input dimension for the VGG-19 architecture is (224,224,64) but in our case, the dataset provides us with the image dimension (48,48,1), so it is modified.

**Convolutional Layers:** VGG-19 consists of 16 convolutional layers, each followed by a rectified linear unit (ReLU) activation function. These convolutional layers were arranged in a sequential manner, with 3x3 filters and a stride of 1 pixel. The number of filters gradually increased with the depth of the network (Shown in figure). 

**Pooling Layers:** After some of the convolutional layers, max-pooling layers are applied to reduce the spatial dimensions of the feature maps while preserving the most important information, i.e. the maximum value in the feature matrix in this case. In VGG-19, max-pooling was performed over a 2x2 pixel window with a stride of 2 pixels.

**Fully Connected Layers:** Towards the end of the network, there were two fully connected layers, each followed by a ReLU activation function. These layers were responsible for combining the features learned by the convolutional layers and making the final predictions. 

**Output Layer:** The output layer of VGG-19 depends on the specific task it's trained for. For image classification tasks such as ours, it used a softmax activation function to produce probabilities for each class i.e. each emotion.

<p align="center">
  <img src=./readme_images/arch.png alt="architecture"/>
</p>

## Results
The model obtained a test accuracy of 82%.
<p align="center">
  <img src=./readme_images/graph.png alt="graph"/>
  <img src=./readme_images/confusion.png alt="matrix" width=600 /> 
</p>

# The Team

- [Swastik Aryal](https://github.com/Swastik-Aryal)
- [Bhisma Prasad Bhandari](https://github.com/DontHash)
- [Rikesh Panta](https://github.com/RiCEmare)
- [Asim Poudel](https://www.github.com/octokatherine)
