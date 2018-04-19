''' Rewrite of dqn.py to check for logical errors '''

import os
import random
import copy

import tensorflow as tf
import numpy as np

import chess

from util import build_input, get_input
