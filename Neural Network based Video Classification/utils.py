import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 1)
    }
    cap.release()
    return info

def save_training_curves(history, out_png='results/training_history.png'):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend(['train','val']); plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(history.history['loss']); plt.plot(history.history['val_loss'])
    plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(['train','val']); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=200)
    print(f"Saved curves to {out_png}")
