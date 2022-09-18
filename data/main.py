import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn import over_sampling

sns.set(rc={'figure.figsize': (16, 8)})
sns.set_style("whitegrid")
sns.color_palette("dark")
plt.style.use("fivethirtyeight")

data = pd.read_csv("./final.csv")

features = ["title_length", "view_count", "like_count", "comment_count", "video_length", "tag_count", "category_id",
            "channel_subscriber_count"]

data_missing_value = data.isnull().sum().reset_index()
print(data_missing_value)
print(data.describe())

# Data cleaning

data_clean = data.copy()
data_clean = data_clean.dropna()

