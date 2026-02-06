import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from training.geopose.model import GeoPoseNet

model = GeoPoseNet()
print(model)
print(f"Layer 1 type: {type(model.conv1)}")
