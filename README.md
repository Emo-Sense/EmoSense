# EmoSense

EmoSense is a real-time emotion detection tool that can recognize emotions from live video feeds, images, and videos. The project utilizes a TensorFlow-trained CNN model to detect facial emotions with high accuracy.

## Installation

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

The best model exceeds the GiHub storage limit so it is stored in a Google Drive folder. You can download and move it to the ```models/``` folder using :

```sh
gdown https://drive.google.com/uc?id=1zfGYRncWUr5Rhs8VdBlWTqbA-uYZVOwC -O models/emosense_model_82.h5
```

## Usage

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


# The Team

- [Swastik Aryal](https://github.com/Swastik-Aryal)
- [Bhisma Prasad Bhandari](https://github.com/DontHash)
- [Rikesh Panta](https://github.com/RiCEmare)
- [Asim Poudel](https://www.github.com/octokatherine)


