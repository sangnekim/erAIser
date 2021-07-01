import glob
from vos.test import *

import os, sys
import os.path
import torch
import numpy as np
import cv2
import pickle

from torch.utils import data
from torch.autograd import Variable
from vi.model import generate_model
