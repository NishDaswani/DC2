# import pandas as pd

# df = pd.read_csv("data/01_final_data/Burglary Classified Data.csv")

# df.drop(columns=["Last outcome category", "Context"], inplace=True)

# print(df[df["classification"] == "residential"].count())

# df = df[df["classification"] == "residential"]
# print(df["Falls within"].value_counts())

# # df.drop(columns=["classification", "Crime ID", "Reported by", "Falls within"], inplace=True)


# df = df[df["Falls within"] == "Metropolitan Police Service"]

# print(df.head())

# print(df.columns)
# df.drop(columns=["Falls within", "Crime ID", "Reported by", "source_file", "month_num", "month_str", "classification"], inplace=True)

# print(df.head())

# df.to_csv("data/01_final_data/Burglary-Classified-Data.csv", index=False)

import pandas as pd

# df = pd.read_csv("data/processed/final_processed_data_new.csv")
df2 = pd.read_csv("data/01_final_data/final_data.csv")
df = pd.read_csv("data/00_new/final_data.csv")
df3 = pd.read_csv("data/00_new/final_data_with_features.csv")
print(df3.isnull().sum())

# print(df.isnull().sum())
# print(df.count())