import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

type_of_feature = "mwmd"

# per album heatmap
mwmd = pd.read_csv(f"general_results_mwmd.csv")
mse = pd.read_csv(f"general_results_mse.csv")
spec_flux = pd.read_csv(f"general_results_spectral_flux.csv")

feature_to_plot = "False Negatives"

mwmd["Accuracy"] = mwmd["True Positives"] / mwmd["Total beats"]
mwmd["F1-Score"] = 2*mwmd["True Positives"] / (2*mwmd["True Positives"] + mwmd["False Negatives"] + mwmd["False Positives"])
mwmd["Rush plus drag"] = mwmd["Drag mean"] + mwmd["Rush mean"]

mse["Accuracy"] = mse["True Positives"] / mse["Total beats"]
mse["F1-Score"] = 2*mse["True Positives"] / (2*mse["True Positives"] + mse["False Negatives"] + mse["False Positives"])
mse["Rush plus drag"] = mse["Drag mean"] + mse["Rush mean"]

spec_flux["Accuracy"] = spec_flux["True Positives"] / spec_flux["Total beats"]
spec_flux["F1-Score"] = 2*spec_flux["True Positives"] / (2*spec_flux["True Positives"] + spec_flux["False Negatives"] + spec_flux["False Positives"])
spec_flux["Rush plus drag"] = spec_flux["Drag mean"] + spec_flux["Rush mean"]

names = ["MSE", "Spectral Flux", "MWMD"]

mwmd_val = mwmd[feature_to_plot].mean()
mse_val = mse[feature_to_plot].mean()
spec_flux_val = spec_flux[feature_to_plot].mean()

plt.plot(names, [mse_val, spec_flux_val, mwmd_val])
plt.ylabel(feature_to_plot)
plt.savefig(f"plots/{feature_to_plot}_comparative.png")
plt.close()