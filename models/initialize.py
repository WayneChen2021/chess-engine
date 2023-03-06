from tf_model import *
from torch_model import *
import json


def initialize() -> None:
    info = json.load("../config.json")
    if info["model"]["tf"]:
        save_model_tf("../data")
    else:
        save_model_torch("../data")
