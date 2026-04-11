import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

df = pd.read_pickle("data/interim/01_data_processed.pkl")


# ---------------------------------------------------------------------------
# Plot single columns
# ---------------------------------------------------------------------------

set_df = df[df["set"] == 1]

plt.plot(set_df['acc_y'])

plt.plot(set_df['acc_y'].reset_index(drop=True))


# --------------------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------------------

for label in df['label'].unique():
    subset = df[df['label'] == label]
    display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
    
    
for label in df['label'].unique():
    subset = df[df['label'] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()



# -------------------------------------------------------------------------
# Adjust Plot Settings
# -------------------------------------------------------------------------








