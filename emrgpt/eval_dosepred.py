"""
Similar to mederrors eval, try to determine when wrong med is being given

Instead of predicting physiology, predict expected dose
given a token sequence that terminates with the medication token
"""

import torch
from torch.utils.data import Dataset, DataLoader
