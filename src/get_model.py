import tensorflow as tf


def _avgpool_dense(image_size: int, num_classes: int, preprocess_input, base_model):
    i = tf.keras.layers.Input([image_size, image_size, 3], dtype=tf.float32, name="input")
    i = preprocess_input(i)
    x = base_model(i)

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return base_model, tf.keras.Model(inputs=[i], outputs=[x], name=base_model.name)


def _get_weight(trainable: bool):
    return None if trainable else "imagenet"


# 模型有丶大
def get_vgg19(image_size: int, num_classes: int, trainable: bool = True):
    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights=_get_weight(trainable), input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = trainable

    i = tf.keras.layers.Input([image_size, image_size, 3], dtype=tf.float32, name="input")
    i = tf.keras.applications.vgg19.preprocess_input(i)
    x = base_model(i)

    # x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    # x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    # x = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(4096, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dense(4096, activation="relu", name="fc2")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return base_model, tf.keras.Model(inputs=[i], outputs=[x], name="vgg19")


# 96x96 2369x20x50
def get_NASNetMobile(image_size: int, num_classes: int):
    base_model = tf.keras.applications.nasnet.NASNetMobile(
        include_top=False, weights=None, input_shape=(image_size, image_size, 3)
    )

    return _avgpool_dense(image_size, num_classes, tf.keras.applications.nasnet.preprocess_input, base_model)


def get_MobileNetV3Large(image_size: int, num_classes: int, trainable: bool = True):
    def hard_sigmoid(x):
        return tf.keras.layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)

    def hard_swish(x):
        return tf.keras.layers.Multiply()([x, hard_sigmoid(x)])

    base_model = tf.keras.applications.MobileNetV3Large(
        include_top=False, weights=_get_weight(trainable), input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = trainable

    i = tf.keras.layers.Input([image_size, image_size, 3], dtype=tf.float32, name="input")
    i = tf.keras.applications.mobilenet_v3.preprocess_input(i)
    x = base_model(i)

    x = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name="avg_pool")(x)
    x = tf.keras.layers.Conv2D(1280, kernel_size=1, padding="same", use_bias=True, name="Conv_2")(x)
    x = hard_swish(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    x = tf.keras.layers.Conv2D(num_classes, kernel_size=1, padding="same", name="Logits")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Activation("softmax", name="Predictions")(x)
    return base_model, tf.keras.Model(inputs=[i], outputs=[x], name=base_model.name)


# 96x96 2369x20x5
def get_EfficientNetV2B0(image_size: int, num_classes: int, trainable: bool = True):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights=_get_weight(trainable), input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = trainable
    return _avgpool_dense(image_size, num_classes, tf.keras.applications.efficientnet_v2.preprocess_input, base_model)


def get_EfficientNetV2B1(image_size: int, num_classes: int, trainable: bool = True):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(
        include_top=False, weights=_get_weight(trainable), input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = trainable
    return _avgpool_dense(image_size, num_classes, tf.keras.applications.efficientnet_v2.preprocess_input, base_model)


def get_Xception(image_size: int, num_classes: int, trainable: bool = True):
    base_model = tf.keras.applications.xception.Xception(
        include_top=False, weights=_get_weight(trainable), input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = trainable
    return _avgpool_dense(image_size, num_classes, tf.keras.applications.xception.preprocess_input, base_model)


if __name__ == "__main__":
    base_model, model = get_Xception(96, 15, False)
    model.summary()
    base_model.trainable = True
    model.summary()
    # model = tf.keras.applications.MobileNetV3Large(include_top=True, weights=None, input_shape=(96, 96, 3), classes=15,)
    # model.summary()
