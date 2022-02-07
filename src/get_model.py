import tensorflow as tf


# 不知道为啥不行
def get_vgg19(image_size: int, num_classes: int):
    base_model = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3), classes=num_classes,
    )

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

    return tf.keras.Model(inputs=[i], outputs=[x], name="vgg19")


# 96x96 2369x20x50
def get_NASNetMobile(image_size: int, num_classes: int):
    base_model = tf.keras.applications.nasnet.NASNetMobile(
        include_top=False, weights=None, input_shape=(image_size, image_size, 3), classes=num_classes,
    )

    i = tf.keras.layers.Input([image_size, image_size, 3], dtype=tf.float32, name="input")
    i = tf.keras.applications.nasnet.preprocess_input(i)
    x = base_model(i)

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=[i], outputs=[x], name="nasnetMobile")


def get_MobileNetV3Large(image_size: int, num_classes: int):
    base_model = tf.keras.applications.MobileNetV3Large(
        include_top=True, weights=None, input_shape=(image_size, image_size, 3), classes=num_classes,
    )
    return base_model


# 96x96 2369x20x5
def get_EfficientNetV2B0(image_size: int, num_classes: int):
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights=None, input_shape=(image_size, image_size, 3), classes=num_classes,
    )

    i = tf.keras.layers.Input([image_size, image_size, 3], dtype=tf.float32, name="input")
    i = tf.keras.applications.efficientnet_v2.preprocess_input(i)
    x = base_model(i)

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="top_dropout")(x)
    x = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=[i], outputs=[x], name="EfficientNetV2B0")
