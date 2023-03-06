import tensorflow as tf
import json
import os

import tf_model


def train(input_data: list[float], train_policy: list[float],
          train_values: list[float]) -> None:
    info = json.load("../config.json")
    optimizer_class = getattr(
        info["training_params"]["py"]["optimizer"]["name"])
    optimizer = optimizer_class(
        **info["training_params"]["py"]["optimizer"]["args"])
    batch_size = info["training_params"]["cpp"]["batch_size"]
    l2_weight = info["training_params"]["py"]["l2_weight"]
    mse = tf.keras.losses.MeanSquaredError()
    cross_ent = tf.keras.losses.CategoricalCrossentropy()

    best_model = tf_model.model_tf()
    input_data = tf.reshape(tf.convert_to_tensor(
        input_data), (-1, batch_size, 64, 119))
    train_policy = tf.reshape(tf.convert_to_tensor(
        train_policy), (-1, batch_size, 64, 73))
    train_values = tf.reshape(tf.convert_to_tensor(
        train_values), (-1, batch_size))

    for _ in range(info["training_params"]["py"]["epochs"]):
        for (input, policy, value) in zip(input_data, train_policy, train_values):
            train_step(input, policy, value)

    tf.keras.models.save_model(best_model, os.path.join(
        "data", str(int(os.listdir("data")[-1]) + 1)))

    @tf.function
    def train_step(x, policy_true, value_true):
        with tf.GradientTape() as tape:
            policy_pred, value_pred = best_model(x)
            loss_value = mse(value_pred, value_true) + cross_ent(policy_pred, policy_true) + \
                l2_weight * tf.add_n([tf.nn.l2_loss(v)
                                     for v in best_model.trainable_weights])

        grads = tape.gradient(loss_value, best_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, best_model.trainable_weights))
