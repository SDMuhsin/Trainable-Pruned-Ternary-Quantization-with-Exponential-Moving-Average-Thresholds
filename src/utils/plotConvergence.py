import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# 1. Define the data
#-----------------------------------------------------------------------------

datasets = [
    "MNIST\n(2DCNN)",
    "FMNIST\n(ResNet18)",
    "KMNIST\n(ResNet18)",
    "EMNIST\n(ResNet18)",
    "SVHN\n(ResNet18)",
    "CIFAR10\n(ResNet50)",
    "CIFAR100\n(ResNet50)",
    "STL10\n(ResNet50)"
]

# Techniques in the exact order: 
# 1) FP (baseline), 2) DoReFa, 3) TTQ, 4) pTTQ, 5) EMA(k=1)
fp_data    = [47.80, 27.00, 38.60, 16.70, 32.90, 45.10, 36.70, 41.60]
dorefa     = [35.40, 45.90, 44.60, 40.50, 48.10, 44.30, 43.00, 39.60]
ttq        = [17.10, 31.10, 40.60, 16.70, 34.70, 38.00, 33.20, 37.70]
pttq       = [ 6.40,  4.80,  3.30,  2.60,  2.30,  4.40,  5.30, 35.90]
ema_k1     = [22.90,  6.50,  2.90,  3.10,  3.00,  4.80,  4.20, 39.90]

# Combine them in a list of lists for convenience
all_techniques = [fp_data, dorefa, ttq, pttq, ema_k1]
tech_labels    = ["FP (Baseline)", "DoReFa [36]", "TTQ [15]", "pTTQ [16]", "EMA(k=1) [ours]"]

#-----------------------------------------------------------------------------
# 2. Create the figure & grouped bars
#-----------------------------------------------------------------------------

x = np.arange(len(datasets))   # the label locations for each dataset
width = 0.15                   # the width of each bar

fig, ax = plt.subplots(figsize=(10, 5))

# For each technique, we shift its bars on the x-axis
for i, technique_data in enumerate(all_techniques):
    # Offset each group of bars by i * width from the central position
    offset = (i - 2) * width  # shift them around center; (i - 2) centers 5 groups
    bars = ax.bar(
        x + offset, 
        technique_data, 
        width, 
        label=tech_labels[i]
    )

#-----------------------------------------------------------------------------
# 3. Annotate and finalize
#-----------------------------------------------------------------------------

ax.set_ylabel('Convergence Epoch')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=0)  # rotate if needed for readability
ax.set_title('Convergence Epochs for Different Compression Techniques')
ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)

# Optionally adjust layout
plt.tight_layout()

# Show the plot
plt.savefig("./results/convergence.png")

