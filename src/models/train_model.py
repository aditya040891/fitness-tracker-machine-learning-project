import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from LearningAlgorithms import ClassificationAlgorithms
pd.set_option('display.max_columns', None)


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features.pkl")

# -----------------------------------------------------------------------------
# Create a training and test set
# -----------------------------------------------------------------------------

df_train = df.drop(['participant', 'category', 'set', 'duration'], axis=1)

X = df_train.drop("label", axis=1).copy()

y = df_train['label'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=0.75, 
                                                    random_state=42, 
                                                    stratify=y)

fig, ax = plt.subplots(figsize=(10,5))
df_train["label"].value_counts().plot(
        kind='bar', 
        ax=ax,
        color='lightblue',
        label='Total'
    )

y_train.value_counts().plot(kind='bar', ax=ax, color='dodgerblue', label='Train')
y_test.value_counts().plot(kind='bar', ax=ax, color='royalblue', label='Test')
plt.legend()
plt.show()


# -----------------------------------------------------------------------------
# Split feature subsets
# -----------------------------------------------------------------------------

basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
square_features = ['acc_r', 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
time_features = [f for f in df_train.columns if "temp" in f]
frequency_features = [f for f in df_train.columns if ("freq" in f) or ("pse" in f)]
cluster_features = ["cluster"]


print("Basic Features:", len(basic_features))
print("Square Features:", len(square_features))
print("PCA Features:", len(pca_features))
print("Time Features:", len(time_features))
print("Frequency Features:", len(frequency_features))
print("Cluster Features:", len(cluster_features))


feature_set_1 = basic_features
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# -----------------------------------------------------------------------------
# Perform forward feature selection using single decision tree
# -----------------------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10

# selected_features, ordered_features, ordered_scores = learner.forward_selection(
#     max_features, X_train, y_train
# )

plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
plt.xlabel('Number of features')
plt.ylabel('Accuracy')
plt.xticks(np.arange(1, max_features + 1, 1))
plt.show()

selected_features = ['pca_1',
 'acc_z_freq_0.0_Hz_ws_14',
 'acc_x_freq_0.0_Hz_ws_14',
 'gyr_r_freq_0.0_Hz_ws_14',
 'acc_r_freq_0.357_Hz_ws_14',
 'acc_r',
 'acc_x',
 'acc_z_freq_weighted',
 'gyr_z_max_freq',
 'gyr_y_freq_1.786_Hz_ws_14']


# -----------------------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# -----------------------------------------------------------------------------

possible_feature_sets = [
        feature_set_1,
        feature_set_2,
        feature_set_3,
        feature_set_4,
        selected_features
    ]

feature_names = [
        'Feature Set 1',
        'Feature Set 2',
        'Feature Set 3',
        'Feature Set 4',
        'Selected Features'
    ]


