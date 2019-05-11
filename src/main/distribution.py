import pickle
from time import sleep

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.main.population import populate
from scipy.stats import zscore


_, _, scores = populate(max_results=100000)
# column = 2
# for a in x:
#     column_data.append(a[column])
#
sns.distplot(scores)
plt.show()