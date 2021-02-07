from tensorflow import keras as K
from tensorflow.keras import layers as L
import tensorflow as tf
from models.resnet import ResNet50
from models.loss import db_loss

def DBNet(config, k=50, model='training'):
    assert model in ['training', 'inference'], 'error'

    input_image = L.Input(shape=[None, None, 3], name='input_image')
    backbone_net = ResNet50(inputs=input_image, include_top=False, freeze_bn=True)
    C2, C3, C4, C5 = backbone_net.outputs

    # in2
    in2 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in2 = L.BatchNormalization()(in2)
    in2 = L.ReLU()(in2)
    # in3
    in3 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in3 = L.BatchNormalization()(in3)
    in3 = L.ReLU()(in3)
    # in4
    in4 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in4 = L.BatchNormalization()(in4)
    in4 = L.ReLU()(in4)
    # in5
    in5 = L.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)
    in5 = L.BatchNormalization()(in5)
    in5 = L.ReLU()(in5)

    # P5
    P5 = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5)
    P5 = L.BatchNormalization()(P5)
    P5 = L.ReLU()(P5)
    P5 = L.UpSampling2D(size=(8, 8))(P5)
    # P4
    out4 = L.Add()([in4, L.UpSampling2D(size=(2, 2))(in5)])
    P4 = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4)
    P4 = L.BatchNormalization()(P4)
    P4 = L.ReLU()(P4)
    P4 = L.UpSampling2D(size=(4, 4))(P4)
    # P3
    out3 = L.Add()([in3, L.UpSampling2D(size=(2, 2))(out4)])
    P3 = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3)
    P3 = L.BatchNormalization()(P3)
    P3 = L.ReLU()(P3)
    P3 = L.UpSampling2D(size=(2, 2))(P3)
    # P2
    out2 = L.Add()([in2, L.UpSampling2D(size=(2, 2))(out3)])
    P2 = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out2)
    P2 = L.BatchNormalization()(P2)
    P2 = L.ReLU()(P2)

    fuse = L.Concatenate()([P2, P3, P4, P5])

    # binarize map
    p = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = L.BatchNormalization()(p)
    p = L.ReLU()(p)
    p = L.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = L.BatchNormalization()(p)
    p = L.ReLU()(p)
    binarize_map = L.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                      activation='sigmoid', name='binarize_map')(p)

    # threshold map
    t = L.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    t = L.BatchNormalization()(t)
    t = L.ReLU()(t)
    t = L.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
    t = L.BatchNormalization()(t)
    t = L.ReLU()(t)
    threshold_map = L.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                                       activation='sigmoid', name='threshold_map')(t)

    # thresh binary map
    thresh_binary = L.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([binarize_map, threshold_map])

    if model == 'training':
        input_gt = L.Input(shape=[config.IMAGE_SIZE, config.IMAGE_SIZE], name='input_gt')
        input_mask = L.Input(shape=[config.IMAGE_SIZE, config.IMAGE_SIZE], name='input_mask')
        input_thresh = L.Input(shape=[config.IMAGE_SIZE, config.IMAGE_SIZE], name='input_thresh')
        input_thresh_mask = L.Input(shape=[config.IMAGE_SIZE, config.IMAGE_SIZE], name='input_thresh_mask')

        loss_layer = L.Lambda(db_loss, name='db_loss')(
            [input_gt, input_mask, input_thresh, input_thresh_mask, binarize_map, thresh_binary, threshold_map])

        db_model = K.Model(inputs=[input_image, input_gt, input_mask, input_thresh, input_thresh_mask],
                           outputs=[loss_layer])

        loss_names = ["db_loss"]
        for layer_name in loss_names:
            layer = db_model.get_layer(layer_name)
            db_model.add_loss(layer.output)
            # db_model.add_metric(layer.output, name=layer_name, aggregation="mean")
    else:
        db_model = K.Model(inputs=input_image,
                           outputs=binarize_map)
        """
        db_model = K.Model(inputs=input_image,
                           outputs=thresh_binary)
        """
    return db_model