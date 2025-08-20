import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

class UCF50DataPreprocessor:
    def __init__(self, data_dir='data/UCF50', image_height=64, image_width=64, sequence_length=20):
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.sequence_length = sequence_length

    def download_instructions(self):
        print("Please download UCF50 dataset manually from:")
        print("https://www.crcv.ucf.edu/data/UCF50.php")
        print("Extract it to:", self.data_dir)
        print("Ensure structure: data/UCF50/<ClassName>/<video files>")

    def extract_frames(self, video_path, sequence_length=None):
        if sequence_length is None:
            sequence_length = self.sequence_length

        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames >= sequence_length:
            frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        else:
            frame_indices = list(range(total_frames))

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (self.image_width, self.image_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        while len(frames) < sequence_length:
            frames.append(frames[-1])
        if len(frames) > sequence_length:
            frames = frames[:sequence_length]

        return np.array(frames)

    def load_data(self, max_videos_per_class=None):
        if not os.path.exists(self.data_dir):
            print(f"Dataset directory {self.data_dir} not found!")
            return None, None, None

        class_names = [d for d in sorted(os.listdir(self.data_dir))
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        print(f"Found {len(class_names)} classes.")

        features = []
        labels = []

        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(self.data_dir, class_name)
            video_files = [f for f in os.listdir(class_path)
                           if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))]
            if max_videos_per_class is not None:
                video_files = video_files[:max_videos_per_class]

            print(f"Processing class '{class_name}' with {len(video_files)} videos...")
            for video_file in tqdm(video_files, desc=f"Class {class_name}"):
                video_path = os.path.join(class_path, video_file)
                frames = self.extract_frames(video_path)
                if frames is None:
                    continue
                features.append(frames)
                labels.append(class_idx)

        if len(features) == 0:
            print("No data processed.")
            return None, None, None

        features = np.array(features, dtype=np.float32)
        labels = to_categorical(np.array(labels), num_classes=len(class_names))
        print(f"Dataset shape: {features.shape}, Labels shape: {labels.shape}")
        return features, labels, class_names

    def split_data(self, features, labels, test_size=0.25, random_state=42):
        y_int = np.argmax(labels, axis=1)
        return train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=y_int)

    def save_preprocessed_data(self, features, labels, class_names, filepath='preprocessed_data.pkl'):
        data = {
            'features': features,
            'labels': labels,
            'class_names': class_names,
            'image_height': self.image_height,
            'image_width': self.image_width,
            'sequence_length': self.sequence_length
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Preprocessed data saved to {filepath}")

    def load_preprocessed_data(self, filepath='preprocessed_data.pkl'):
        if not os.path.exists(filepath):
            print(f"Preprocessed file not found: {filepath}")
            return None, None, None
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['features'], data['labels'], data['class_names']


if __name__ == "__main__":
    pre = UCF50DataPreprocessor()
    if not os.path.exists('preprocessed_data.pkl'):
        X, y, classes = pre.load_data(max_videos_per_class=10)  # set to None for full dataset
        if X is not None:
            pre.save_preprocessed_data(X, y, classes)
            X_train, X_test, y_train, y_test = pre.split_data(X, y)
            print("Train:", X_train.shape, y_train.shape)
            print("Test:", X_test.shape, y_test.shape)
    else:
        X, y, classes = pre.load_preprocessed_data()
        X_train, X_test, y_train, y_test = pre.split_data(X, y)
        print("Train:", X_train.shape, y_train.shape)
        print("Test:", X_test.shape, y_test.shape)
