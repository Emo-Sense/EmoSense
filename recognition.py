import sys
import os
import cv2 as cv
import numpy as np
import tensorflow as tf

# Removes unnecessary warnings
tf.get_logger().setLevel("ERROR")


output_directory = "output_samples"
model = tf.keras.models.load_model(r"models\emosense_finalmodel_82.h5")

# Emotion Classes are from the fer+ labels used with the fer2013 dataset
# emotions = ["Neutral","Happy","Surprise","Sadness","Anger","Disgust","Fear","Contempt","Unknown","NF",]
# Contempt, Unknown and Not a Face (NF) removed during training
emotions = [
    "Neutral",
    "Happy",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
]


# Separates face from frame
def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier(
        r"haar_cascades\haarcascade_frontalface_alt.xml"
    )
    # scaleFactor : (the amount the image is scaled i.e. reduced ) lower =  more accurate but slower processing
    # minNeighbors : (specifies how many neighbors each face rectangle should have for it to be a face) higher = more accurate
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    return faces


# Main detection code
def detect_emotions_in_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_detect(frame)
    for x, y, w, h in faces:

        # Image preprocessing
        face_roi = gray[y : y + h, x : x + w]
        resized_face = cv.resize(face_roi, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

        predictions = model.predict(reshaped_face)
        probabilities = [round(pred * 100, 2) for pred in predictions.tolist()[0]]
        
        max_index = np.argmax(predictions)
        emotion = f"{emotions[max_index]} {max(probabilities)}%"
        
        cv.putText(
            frame,
            emotion,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0, 0),
            4,
            lineType=cv.LINE_AA
        )
        cv.putText(
            frame,
            emotion,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            lineType=cv.LINE_AA
        )
        
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


# Only for process image and process video windows
def resize_frame(image_path):
    resized_image = image_path
    if image_path.shape[1] > 800 or image_path.shape[1] < 400:
        new_width = 800
        new_height = int((image_path.shape[0] / image_path.shape[1]) * new_width)
        dim = (new_width, new_height)
        resized_image = cv.resize(image_path, dim, interpolation=cv.INTER_AREA)
    return resized_image


# For image input
def process_image(image_path):
    image = cv.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    print("Processing Image..... ")

    result_image = detect_emotions_in_frame(resize_frame(image))

    filename = os.path.basename(image_path)
    file_root, file_ext = os.path.splitext(filename)
    output_path = os.path.join(output_directory, f"{file_root}_result{file_ext}")

    cv.imwrite(output_path, result_image)

    print(f"Processed image saved as {output_path}")
    cv.imshow("Processed Image", result_image)

    while True:
        if cv.getWindowProperty("Processed Image", cv.WND_PROP_VISIBLE) < 1:
            break
        cv.waitKey(100)

    cv.destroyAllWindows()


# For Video Input
def process_video(video_path):
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    filename = os.path.basename(video_path)
    file_root, file_ext = os.path.splitext(filename)
    output_path = os.path.join(output_directory, f"{file_root}_result{file_ext}")

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(
        output_path,
        fourcc,
        video.get(cv.CAP_PROP_FPS),
        (int(video.get(3)), int(video.get(4))),
    )
    print("Processing video ....")

    # Displaying the process
    while True:
        ret, frame = video.read()
        if not ret:
            break
        result_frame = detect_emotions_in_frame(frame)
        out.write(result_frame)
        cv.imshow("Video Emotion Detection", result_frame)
        if cv.waitKey(1) == ord("q") or (
            cv.getWindowProperty("Video Emotion Detection", cv.WND_PROP_VISIBLE) < 1
        ):
            break

    video.release()
    out.release()
    print(f"Processed video saved as {output_path}")


def realtime_emotion_detection():
    video = cv.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        result_frame = detect_emotions_in_frame(frame)
        cv.imshow("Real-time Emotion Detection", result_frame)
        if cv.waitKey(1) == ord("q") or (
            cv.getWindowProperty("Real-time Emotion Detection", cv.WND_PROP_VISIBLE) < 1
        ):
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        realtime_emotion_detection()
    elif len(sys.argv) == 2:
        input_path = sys.argv[1]
        if input_path.lower().endswith(("jpg", "jpeg", "png")):
            process_image(input_path)
        elif input_path.lower().endswith(("mp4", "mov", "avi", "mkv")):
            process_video(input_path)
        else:
            print("Unsupported file format. Please provide an image or video file.")
    else:
        print("Usage:")
        print(
            "  python recognition.py                        # For real-time emotion detection"
        )
        print(
            "  python recognition.py image.jpg/png/jpeg     # For emotion detection in an image"
        )
        print(
            "  python recognition.py video.mp4/mov/mkv/avi  # For emotion detection in a video"
        )
        print("  Note: Enclose filepaths within quotes")
        print("  Eg : python recognition.py input_samples/sample1.jpg")
