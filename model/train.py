import os
import time
import math
import json
import torch
import random
import pandas as pd
from copy import deepcopy
from shutil import copyfile
import sys
sys.path.insert(0, os.getcwd())
from chess import Chess
from model import TransformerModel

