# LRCN Video Classification (UCF50)

Implements Long-term Recurrent Convolutional Networks (LRCN) for human action recognition on the UCF50 dataset using CNN+LSTM with Keras/TensorFlow.

## Features
- Frame extraction, resize to 64x64, normalization
- TimeDistributed CNN (VGG16 pretrained or custom)
- LSTM temporal modeling (default 256 units)
- Adam optimizer, categorical cross-entropy
- 75/25 train-test split
- Inference on files or webcam

## Setup
1) Create environment and install dependencies
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

text
2) Download UCF50 from the official site and extract to:
data/UCF50/<ClassName>/<video files>

text

## Preprocess and Train
Preprocess (saves preprocessed_data.pkl)
python data_preprocessing.py

Train
python train.py --epochs 50 --batch_size 8 --learning_rate 0.0001

text

## Inference
On a video file
python inference.py --model models/best_lrcn_model.h5 --video path/to/video.mp4 --top_k 5

From webcam (5 seconds)
python inference.py --model models/best_lrcn_model.h5 --webcam --duration 5

text

## Notes
- Adjust sequence length, image size, and model hyperparameters in the scripts as needed.
- For faster experiments, set `--max_videos_per_class` in train.py.
- GPU training recommended.
How to use

Place all files in a project folder.

Ensure the dataset is at data/UCF50 with class subfolders.

Run preprocessing, then training, then inference as shown in README.