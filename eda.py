import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

# Load the cleaned dataset
data = pd.read_csv('cleaned_data.csv')

# Univariate Analysis
cols = ['AT', 'V', 'AP', 'RH', 'PE']
length = len(cols)
cs = ["b", "r", "g", "c", "m", "k", "lime", "c"]
fig, axes = plt.subplots(4, 2, figsize=(13, 25))

for i, (ax, color) in enumerate(zip(axes.flatten(), cs)):
    if i < length:
        sns.histplot(data[cols[i]], ax=ax, color=color, kde=True)
        sns.rugplot(data[cols[i]], ax=ax, color=color)
        ax.set_facecolor("w")
        ax.axvline(data[cols[i]].mean(), linestyle="dashed", label="mean", color="k")
        ax.legend(loc="best")
        ax.set_title(f'{cols[i]} distribution', color="navy")
        ax.set_xlabel("")
    else:
        fig.delaxes(ax)  # Remove empty subplots

plt.tight_layout()
plt.show()

# Multivariate Analysis
# Pairplot
sns.pairplot(data[cols], diag_kind='kde', corner=True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = data[cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
