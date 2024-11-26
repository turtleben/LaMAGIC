import copy
import os

import numpy as np
import torch.nn as nn
import pickle
import transformers
from transformers import Trainers

class AnalogRegressionTrainer(Trainers):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        mse_loss = nn.MSELoss()
        loss = mse_loss(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss
        
        
