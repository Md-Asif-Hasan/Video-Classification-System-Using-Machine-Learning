import os
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data_preprocessing import UCF50DataPreprocessor
from model import LRCNModel

def create_training_config():
    return {
        'data_dir': 'data/UCF50',
        'preprocessed_data_path': 'preprocessed_data.pkl',
        'image_height': 64,
        'image_width': 64,
        'sequence_length': 20,
        'test_split': 0.25,
        'random_seed': 42,
        'max_videos_per_class': None,  # set to an int to limit
        'lstm_units': 256,
        'dropout_rate': 0.3,
        'use_pretrained_cnn': True,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 1e-4,
        'patience': 10,
        'min_lr': 1e-7,
        'model_checkpoint_path': 'models/best_lrcn_model.h5',
        'final_model_path': 'models/final_lrcn_model.h5',
        'results_dir': 'results',
        'verbose': True
    }

def plot_history(history, out_png):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Acc'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"Saved training plot to {out_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_videos_per_class', type=int, default=None)
    args = parser.parse_args()

    cfg = create_training_config()
    cfg['epochs'] = args.epochs
    cfg['batch_size'] = args.batch_size
    cfg['learning_rate'] = args.learning_rate
    cfg['max_videos_per_class'] = args.max_videos_per_class

    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    pre = UCF50DataPreprocessor(cfg['data_dir'], cfg['image_height'], cfg['image_width'], cfg['sequence_length'])

    if os.path.exists(cfg['preprocessed_data_path']):
        X, y, class_names = pre.load_preprocessed_data(cfg['preprocessed_data_path'])
    else:
        X, y, class_names = pre.load_data(max_videos_per_class=cfg['max_videos_per_class'])
        if X is None:
            print("No data found. Ensure dataset is at data/UCF50")
            return
        pre.save_preprocessed_data(X, y, class_names, cfg['preprocessed_data_path'])

    X_train, X_test, y_train, y_test = pre.split_data(X, y, test_size=cfg['test_split'], random_state=cfg['random_seed'])
    num_classes = len(class_names)
    print(f"Classes: {num_classes}, Train: {X_train.shape}, Test: {X_test.shape}")

    lrcn = LRCNModel(sequence_length=cfg['sequence_length'],
                     image_height=cfg['image_height'],
                     image_width=cfg['image_width'],
                     num_classes=num_classes,
                     lstm_units=cfg['lstm_units'],
                     dropout_rate=cfg['dropout_rate'])
    model = lrcn.build_lrcn_model(pretrained_cnn=cfg['use_pretrained_cnn'])
    lrcn.compile_model(learning_rate=cfg['learning_rate'])
    callbacks = lrcn.get_callbacks(cfg['model_checkpoint_path'], patience=cfg['patience'], min_lr=cfg['min_lr'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=cfg['epochs'],
                        batch_size=cfg['batch_size'],
                        callbacks=callbacks,
                        verbose=1 if cfg['verbose'] else 2)

    plot_history(history, os.path.join(cfg['results_dir'], 'training_history.png'))

    print("Evaluating...")
    test_loss, test_acc, test_top5 = model.evaluate(X_test, y_test, batch_size=cfg['batch_size'], verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Top-5: {test_top5:.4f}")

    y_pred = np.argmax(model.predict(X_test, batch_size=cfg['batch_size']), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['results_dir'], 'confusion_matrix.png'), dpi=200)
    print("Saved confusion matrix.")

    lrcn.save_model(cfg['final_model_path'])

if __name__ == "__main__":
    main()
