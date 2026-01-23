# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 120)
print("Deep Parameter Analysis")
print("=" * 120)

# Read data
df = pd.read_csv('final_data_with_gold_signal.csv', encoding='utf-8-sig')
