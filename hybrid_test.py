import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import load_data, data_info
import torch


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    
