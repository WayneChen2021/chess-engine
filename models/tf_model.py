import tensorflow as tf
import os
import json


class PolicyHeadTF(tf.keras.layers):
    def __init__(self):
        super(PolicyHeadTF, self).__init__(name="policy")

    def call(self, inputs):
        pass


class ValueHeadTF(tf.keras.layers):
    def __init__(self):
        super(ValueHeadTF, self).__init__(name="value")

    def call(self, inputs):
        pass


class MainLayerTF(tf.keras.layers):
    def __init__(self):
        super(MainLayerTF, self).__init__()

    def call(self, inputs):
        pass


def model_tf():
    info = json.load("../config.json")
    model_data = info["model"]
    policy_class = getattr(model_data["policy"]["name"])
    value_class = getattr(model_data["value"]["name"])
    main_class = getattr(model_data["main"]["name"])

    policyhead = policy_class(**info["policy"]["args"])
    valuehead = value_class(**info["value"]["args"])
    mainlayer = main_class(**info["main"]["args"])

    board_input = tf.keras.Input(shape=(64, None,), name="board_input")
    main_output = mainlayer(board_input)
    value_output = valuehead(main_output)
    policy_output = policyhead(main_output)

    return tf.keras.Model(inputs=[board_input], outputs=[policy_output, value_output])


def save_model_tf(path: str) -> None:
    best_model = model_tf()
    tf.saved_model.save(best_model, os.path.join(path, "0"))
