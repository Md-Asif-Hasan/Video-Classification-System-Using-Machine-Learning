import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
import pickle

from model import LRCNModel  # only used for reference; we load the saved model directly
from data_preprocessing import UCF50DataPreprocessor

class LRCNInference:
    def __init__(self, model_path, preproc_config_path='preprocessed_data.pkl'):
        self.model = tf.keras.models.load_model(model_path)
        self.config = {'image_height': 64, 'image_width': 64, 'sequence_length': 20}
        self.class_names = None

        if os.path.exists(preproc_config_path):
            with open(preproc_config_path, 'rb') as f:
                data = pickle.load(f)
            self.class_names = data.get('class_names', None)
            self.config['image_height'] = data.get('image_height', 64)
            self.config['image_width'] = data.get('image_width', 64)
            self.config['sequence_length'] = data.get('sequence_length', 20)

        self.pre = UCF50DataPreprocessor(image_height=self.config['image_height'],
                                         image_width=self.config['image_width'],
                                         sequence_length=self.config['sequence_length'])

        if self.class_names is None:
            # Default UCF50 class names (50)
            self.class_names = [
                'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards',
                'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing',
                'GolfSwing', 'HighJump', 'HorseRiding', 'IceDancing', 'JavelinThrow',
                'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Lifting',
                'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks',
                'PizzaTossing', 'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin',
                'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps',
                'RockClimbingIndoor', 'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding',
                'Skiing', 'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi',
                'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'
            ]

    def preprocess_video(self, video_path):
        frames = self.pre.extract_frames(video_path, sequence_length=self.config['sequence_length'])
        if frames is None:
            raise ValueError(f"Could not read frames from {video_path}")
        return np.expand_dims(frames, axis=0)

    def predict_video(self, video_path, top_k=5):
        batch = self.preprocess_video(video_path)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        return {
            'predicted_class': self.class_names[idx],
            'confidence': float(probs[idx]),
            'top_k': [(self.class_names[i], float(probs[i])) for i in top_indices]
        }

    def predict_webcam(self, duration_seconds=5, device_index=0):
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(fps * duration_seconds)
        step = max(1, total_frames // self.config['sequence_length'])
        collected = []
        count = 0

        while len(collected) < self.config['sequence_length']:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame = cv2.resize(frame, (self.config['image_width'], self.config['image_height']))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)/255.0
                collected.append(frame)
            count += 1

        cap.release()

        while len(collected) < self.config['sequence_length']:
            collected.append(collected[-1] if collected else np.zeros((self.config['image_height'], self.config['image_width'], 3), dtype=np.float32))

        batch = np.expand_dims(np.array(collected), axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(probs))
        return {
            'predicted_class': self.class_names[idx],
            'confidence': float(probs[idx])
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5)')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--duration', type=int, default=5, help='Webcam duration seconds')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    inf = LRCNInference(args.model)

    if args.webcam:
        res = inf.predict_webcam(duration_seconds=args.duration)
        print("Prediction:", res['predicted_class'], f"(conf: {res['confidence']:.3f})")
    elif args.video:
        res = inf.predict_video(args.video, top_k=args.top_k)
        print("Predicted:", res['predicted_class'], f"(conf: {res['confidence']:.3f})")
        print("Top-k:")
        for i, (c, p) in enumerate(res['top_k'], 1):
            print(f"{i}. {c}: {p:.3f}")
    else:
        print("Provide --video or --webcam")
