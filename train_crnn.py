import os
import cv2
import pickle
import random
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path

from models.crnn import model
from models.loss import CTCLoss
from models.accuracy import WordACC

from config import CRNNConfig

cfg = CRNNConfig()

table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
    cfg.TABLE_PATH,
    tf.string,
    tf.lookup.TextFileIndex.WHOLE_LINE,
    tf.int64,
    tf.lookup.TextFileIndex.LINE_NUMBER), cfg.NUM_CLASSES - 1)

def process_img(img, mode='training'):
    assert mode in ['training', 'validation', 'inference'], 'error'
    img = tf.image.decode_jpeg(img, channels=3)
    if mode == 'train':
        img_shape = (32, 320, 3)
        # 饱和度
        img = tf.image.random_saturation(img, lower=0, upper=3)
        # 色调 randomly picked in the interval [-max_delta, max_delta)
        img = tf.image.random_hue(img, max_delta=0.3)
        # 对比度
        img = tf.image.random_contrast(img, lower=0.5, upper=5)
        # 亮度
        img = tf.image.random_brightness(img, max_delta=0.05)

    elif mode == 'validation':
        img_shape = (32, 320, 3)
    else:
        img_shape = (32, 720, 3)

    img = img / 255
    img -= 0.5
    img /= 0.5

    h, w, c = img_shape
    resized_img = tf.image.resize(img, (h, w), preserve_aspect_ratio=True)
    return tf.image.pad_to_bounding_box(resized_img, 0, 0, h, w)

def load_and_process_img(path, label, mode='training'):
    img = tf.io.read_file(path)
    return process_img(img, mode), label

def load_img(path):
    img = tf.io.read_file(path)
    return tf.image.decode_jpeg(img, channels=3)

def decode_label(img, label):
    chars = tf.strings.unicode_split(label, "UTF-8")
    tokens = tf.ragged.map_flat_values(table.lookup, chars)
    tokens = tokens.to_sparse()
    return img, tokens

def load_dataset():
    """
    获取图片的路径、标签的路径
    """
    dir_path = cfg.TRAIN_DATA_PATH

    if os.path.exists(os.path.join(dir_path, 'dataset.data')):
        with open(os.path.join(dir_path, 'dataset.data'), 'rb') as ds:
            train_all_image_paths, train_all_image_labels, val_all_image_paths, val_all_image_labels = pickle.load(ds)

        print('Loaded! Load dataset from dataset.data.')
        return train_all_image_paths, train_all_image_labels, val_all_image_paths, val_all_image_labels

    img_list = []
    train_all_image_paths = []
    train_all_image_labels = []
    val_all_image_paths = []
    val_all_image_labels = []
    for root, dirs, files in tqdm(os.walk(dir_path)):
        for file in files:
            if '.jpg' in file:
                file_path = os.path.join(root, file)
                label_path = file_path.replace('.jpg', '.txt')
                if Path(file_path.replace('.jpg', '.txt')).exists():
                    with open(label_path) as f:
                        label = f.read().strip()
                    img = cv2.imread(file_path)
                    if img.shape[1] / img.shape[0] <= 10 and len(label) > 0:
                        img_list.append((file_path, label))

    random.shuffle(img_list)
    for img, label in img_list:
        random_num = random.randint(1, 100)
        if random_num == 1:
            val_all_image_paths.append(img)
            val_all_image_labels.append(label)
        else:
            train_all_image_paths.append(img)
            train_all_image_labels.append(label)

    with open(os.path.join(dir_path, 'dataset.data'), 'wb') as ds:
        pickle.dump((train_all_image_paths, train_all_image_labels, val_all_image_paths, val_all_image_labels), ds)

    print('Loaded.')
    return train_all_image_paths, train_all_image_labels, val_all_image_paths, val_all_image_labels


def train():
    train_all_image_paths, train_all_image_labels, val_all_image_paths, val_all_image_labels = load_dataset()

    train_images_num = len(train_all_image_paths)
    train_ds = tf.data.Dataset.from_tensor_slices((train_all_image_paths, train_all_image_labels))
    train_ds = train_ds.map(load_and_process_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=cfg.BUFFER_SIZE)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(cfg.BATCH_SIZE)
    train_ds = train_ds.map(decode_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    val_images_num = len(val_all_image_paths)
    val_ds = tf.data.Dataset.from_tensor_slices((val_all_image_paths, val_all_image_labels))
    val_ds = val_ds.map(load_and_process_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.shuffle(buffer_size=cfg.BUFFER_SIZE)
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(cfg.BATCH_SIZE)
    val_ds = val_ds.map(decode_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    print('Train Set:', train_images_num)
    print('Validation Set:', val_images_num)


    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
                  loss=CTCLoss(),
                  metrics=[WordACC()])

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=cfg.LOG_DIR),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(cfg.CHECKPOINT_DIR, 'output/crnn_{epoch}.h5'),
                                           monitor='val_loss',
                                           verbose=1)
    ]

    model.fit(train_ds,
              epochs=cfg.EPOCHS,
              steps_per_epoch=train_images_num // cfg.BATCH_SIZE,
              validation_data=val_ds,
              validation_steps=val_images_num // cfg.BATCH_SIZE,
              initial_epoch=0,
              callbacks=callbacks)


if __name__ == '__main__':
    # load_dataset()
    train()