import numpy as np
import pandas as pd

import scikit_posthocs as sp
import scipy.stats as stats
from Orange.evaluation import compute_CD, graph_ranks

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('ps', fonttype=42)

filename = '/content/drive/My Drive/results_RF.xlsx'

df = pd.read_excel(filename,
												 usecols=range(1,9),
												 index_col=0,
												 dtype={'classifier': str,
												        'None': float,
																'ros': float,
																'smote': float,
																'borderline': float,
																'adasyn': float,
																'smote-enn': float,
																'smote-tomek': float,
																}
												)