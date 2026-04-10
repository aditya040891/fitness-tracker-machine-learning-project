import pandas as pd
from glob import glob
pd.set_option('display.max_columns', None)

# -------------------------------------------------------------------------
# Read single csv file
# -------------------------------------------------------------------------

single_file_acc = pd.read_csv('data/raw/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

single_file_gyr = pd.read_csv('data/raw/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')


# -------------------------------------------------------------------------
# List all data in data/raw
# -------------------------------------------------------------------------

files = glob("data/raw/*.csv")
len(files)

# -------------------------------------------------------------------------
# Extract features from filename
# -------------------------------------------------------------------------

data_path = "data/raw/"
f = files[0]

df = pd.read_csv(f)

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123_MetaWear_2019")

df["participant"] = participant
df["label"] = label
df["category"] = category


# -------------------------------------------------------------------------
# Real all files
# ------------------------------------------------------------------------- 

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123_MetaWear_2019")
    
    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df['set'] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df], axis=0)
    if "Gyroscope" in f:
        df['set'] = gyr_set
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df], axis=0)

    











