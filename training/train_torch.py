import torch
import json
import os

import torch_model


def train(input_data: list[float], train_policy: list[float],
          train_values: list[float]) -> None:
    info = json.load("../config.json")
    optimizer_class = getattr(
        info["training_params"]["py"]["optimizer"]["name"])
    optimizer = optimizer_class(
        **info["training_params"]["py"]["optimizer"]["args"])
    l2_weight = info["training_params"]["py"]["l2_weight"]
    batch_size = info["training_params"]["cpp"]["batch_size"]
    mse = torch.nn.MSELoss()
    cross_ent = torch.nn.CrossEntropyLoss()

    best_model = torch_model.model_torch()
    input_data = torch.reshape(torch.tensor(
        input_data), (-1, batch_size, 64, 119))
    train_policy = torch.reshape(torch.tensor(
        train_policy), (-1, batch_size, 64, 73))
    train_values = torch.reshape(torch.tensor(train_values), (-1, batch_size))

    for _ in range(info["training_params"]["py"]["epochs"]):
        for (input, policy_true, value_true) in zip(input_data, train_policy, train_values):
            policy_pred, value_pred = best_model(input)
            loss = mse(value_pred, value_true) + \
                cross_ent(policy_pred, policy_true)
            for p in best_model.parameters():
                loss += l2_weight * p.abs().sum()
            loss.backward()
            optimizer.step()

    model = torch.jit.script(best_model)
    model.save(os.path.join("data", str(int(os.listdir("data")[-1]) + 1)))
