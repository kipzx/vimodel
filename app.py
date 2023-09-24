import os
import cv2
import numpy as np
import tensorflow as tf
from data_processing import load_video, preprocess_frames


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def enhance_video(input_video_path, output_video_path, model):
    frames, fps = load_video(input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (128, 128))
        improved_frame = model.predict(np.expand_dims(frame, axis=0))[0]
        improved_frame = cv2.resize(improved_frame, (frame_width, frame_height))
        out.write(improved_frame.astype(np.uint8))

    cap.release()
    out.release()

    print(f"Обработано {frame_count} кадров.")


if __name__ == "__main__":
    model_path = 'vimodel.keras'
    input_folder = 'C:/Users/Python/PycharmProjects/vimodel/data/input'
    output_folder = 'C:/Users/Python/PycharmProjects/vimodel/data/output'

    model = load_model(model_path)
    print(model.summary())

    os.makedirs(output_folder, exist_ok=True)
    print("Все видео обработаны.")
