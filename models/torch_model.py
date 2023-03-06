import torch
import os
import json


class PolicyHeadTorch(torch.nn.Module):
    def __init__(self):
        super(PolicyHeadTorch, self).__init__()

    def forward(self, input):
        pass


class ValueHeadTorch(torch.nn.Module):
    def __init__(self):
        super(ValueHeadTorch, self).__init__()

    def forward(self, input):
        pass


class MainLayerTorch(torch.nn.Module):
    def __init__(self):
        super(MainLayerTorch, self).__init__()

    def forward(self, input):
        pass


class FullModelTorch(torch.nn.Module):
    def __init__(self, policy_head, value_head, main_layer):
        super(FullModelTorch, self).__init__()
        self.policy_head = policy_head
        self.value_head = value_head
        self.main_layer = main_layer

    def forward(self, input):
        main_out = self.main_layer(input)
        return self.policy_head(main_out), self.value_head(main_out)


def model_torch():
    info = json.load("../config.json")
    model_data = info["model"]
    policy_class = getattr(model_data["policy"]["name"])
    value_class = getattr(model_data["value"]["name"])
    main_class = getattr(model_data["main"]["name"])

    policyhead = policy_class(**info["policy"]["args"])
    valuehead = value_class(**info["value"]["args"])
    mainlayer = main_class(**info["main"]["args"])
    return FullModelTorch(policyhead, valuehead, mainlayer)


def save_model_torch(path: str) -> None:
    best_model = model_torch()
    model = torch.jit.script(best_model)
    model.save(os.path.join(path, "0"))
