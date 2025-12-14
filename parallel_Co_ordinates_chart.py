import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# ------------------------------------------------------------
# Data: VGG16 Quantization Sweep Results
# ------------------------------------------------------------
data = [
    ["FP32 Baseline", 32, 32, 92.22, 99.82, 9.2],
    ["w8_a8", 8, 8, 83, 99.1, 2.35],
    ["w8_a4", 8, 4, 82, 99.0, 2.25],
    ["w6_a6", 6, 6, 87.39, 99.43, 1.8],
    ["w6_a2", 6, 2, 83, 99.0, 1.7],
    ["w4_a32", 4, 32, 87.39, 99.43, 1.28],
    ["w4_a8", 4, 8, 80, 98.9, 1.22],
    ["w4_a4", 4, 4, 80, 98.8, 1.16],
    ["w4_a2", 4, 2, 76, 98.4, 1.12],
    ["w3_a3", 3, 3, 76, 98.5, 0.91],
    ["w2_a6", 2, 6, 63, 96.5, 0.67],
    ["w2_a4", 2, 4, 63, 96.3, 0.64],
    ["w2_a2", 2, 2, 62, 95.5, 0.61],
]

columns = [
    "Run",
    "Weight Bits",
    "Activation Bits",
    "Val Top-1 Acc",
    "Val Top-5 Acc",
    "Model Size (MB)"
]

df = pd.DataFrame(data, columns=columns)

# ------------------------------------------------------------
# Explicit axis ranges (as required)
# ------------------------------------------------------------
ranges = {
    "Weight Bits": (0, 32),
    "Activation Bits": (0, 32),
    "Val Top-1 Acc": (60, 100),
    "Val Top-5 Acc": (90, 100),
    "Model Size (MB)": (0.5, 10),
}

# ------------------------------------------------------------
# Normalize data to [0, 1] using fixed ranges
# ------------------------------------------------------------
df_norm = df.copy()
for col, (min_val, max_val) in ranges.items():
    df_norm[col] = (df[col] - min_val) / (max_val - min_val)

# ------------------------------------------------------------
# Plot Parallel Coordinates
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

parallel_coordinates(
    df_norm,
    class_column="Run",
    ax=ax,
    linewidth=1.5,
    alpha=0.9
)

ax.set_title(
    "Parallel Coordinates Plot (Normalized with Scale Labels): VGG16 Quantization Sweep",
    fontsize=14
)
ax.set_ylabel("Normalized Value")

# ------------------------------------------------------------
# Add explicit scale labels for each axis
# ------------------------------------------------------------
for i, (label, (min_val, max_val)) in enumerate(ranges.items()):
    # Minimum value at bottom
    ax.text(
        i, -0.12, f"{min_val}",
        ha="center", va="top", fontsize=9,
        transform=ax.get_xaxis_transform()
    )
    # Maximum value at top
    ax.text(
        i, 1.05, f"{max_val}",
        ha="center", va="bottom", fontsize=9,
        transform=ax.get_xaxis_transform()
    )

ax.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()

plt.show()
