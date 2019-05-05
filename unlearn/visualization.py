import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('output_matrix.csv')

time_mean = df.groupby("n").mean().values
ns = df.groupby("n").mean().index.get_level_values(0)

plt.plot(ns,time_mean[:,0],label="native learning")
plt.plot(ns,time_mean[:,1],label="unlearn supported learning")
plt.plot(ns,time_mean[:,2],label="unlearn")
plt.legend()
plt.xlabel("Number of Rating")
plt.ylabel("Time in second")
plt.title("Time Cost, Vectorized Unlearn supported Learning, Nested Loop Unlearn with CSR Indexing")
plt.show()