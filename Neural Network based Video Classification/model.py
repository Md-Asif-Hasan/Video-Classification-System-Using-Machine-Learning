import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

class LRCNModel:
    def __init__(self, sequence_length=20, image_height=64, image_width=64, num_classes=50,
                 lstm_units=256, dropout_rate=0.3):
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None

    def create_cnn_feature_extractor(self, pretrained=True):
        input_shape = (self.image_height, self.image_width, 3)
        if pretrained:
            base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
            for layer in base.layers:
                layer.trainable = False
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
            cnn = Model(inputs=base.input, outputs=x, name='vgg16_feature_extractor')
        else:
            cnn = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                BatchNormalization(), MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(), MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(), MaxPooling2D((2, 2)),
                GlobalAveragePooling2D(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(self.dropout_rate)
            ], name='custom_cnn')
        return cnn

    def build_lrcn_model(self, pretrained_cnn=True):
        video_input = Input(shape=(self.sequence_length, self.image_height, self.image_width, 3), name='video_input')
        cnn = self.create_cnn_feature_extractor(pretrained=pretrained_cnn)
        features = TimeDistributed(cnn, name='td_cnn')(video_input)

        x = LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate,
                 recurrent_dropout=self.dropout_rate, name='lstm_1')(features)
        x = BatchNormalization(name='bn_lstm_1')(x)
        x = LSTM(self.lstm_units // 2, dropout=self.dropout_rate,
                 recurrent_dropout=self.dropout_rate, name='lstm_2')(x)
        x = BatchNormalization(name='bn_lstm_2')(x)

        x = Dense(256, activation='relu', name='dense_1')(x)
        x = BatchNormalization(name='bn_dense_1')(x)
        x = Dropout(self.dropout_rate, name='dropout_1')(x)

        x = Dense(128, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='bn_dense_2')(x)
        x = Dropout(self.dropout_rate, name='dropout_2')(x)

        output = Dense(self.num_classes, activation='softmax', name='classification_output')(x)
        self.model = Model(inputs=video_input, outputs=output, name='LRCN_Classifier')
        return self.model

    def compile_model(self, learning_rate=1e-3):
        if self.model is None:
            raise ValueError("Build the model first.")
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')])
        print("Model compiled.")

    def get_callbacks(self, model_checkpoint_path='models/best_lrcn_model.h5', patience=10, min_lr=1e-7):
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience,
                                             restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                                 min_lr=min_lr, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor='val_accuracy',
                                               save_best_only=True, verbose=1)
        ]

    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(path)
        print(f"Saved model to {path}")
