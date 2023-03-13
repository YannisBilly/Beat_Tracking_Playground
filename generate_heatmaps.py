import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

type_of_feature = "mwmd"

df = pd.read_csv(f"general_results_{type_of_feature}.csv")

albums = df["Album"].unique()

df["Accuracy"] = df["True Positives"] / df["Total beats"]
df["F1-Score"] = 2*df["True Positives"] / (2*df["True Positives"] + df["False Negatives"] + df["False Positives"])
df["False Positives"] = df["False Positives"] / 100
df = df.rename({"False Positives" : "False Positives (*1e-2)"}, axis = 1)


df = df.drop(["Total beats", "Total predicted beats", "True Positives", 
              "False Negatives", "Drag no", "Rush no", 
              "Drag std", "Rush std",
              "Drag mean", "Rush mean"], axis = 1)

plt.figure(figsize=(15,15))
sns.heatmap(df.groupby(["Album"]).mean(), robust=True, annot=True, vmin = 0, vmax = 1)
plt.savefig(f"plots/{type_of_feature}_per_album.png")