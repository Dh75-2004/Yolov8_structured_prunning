import os
import math
import torch
import pandas as pd


def get_pruning_ratio(layer_drop):

    if layer_drop < 1:
        return 0.80

    elif layer_drop < 2:
        return 0.70

    elif layer_drop < 4:
        return 0.60

    elif layer_drop <= 5:
        return 0.50

    else:
        return 0.0
    

LAYER_REPORT = "sensitivity_results/layer_sensitivity.csv"

CHANNEL_REPORT = "channel_sensitivity_results/channel_pruning_dictionary.pt"

OUTPUT_DIR = "pruning_plan"

os.makedirs(OUTPUT_DIR, exist_ok=True)

layer_df = pd.read_csv(LAYER_REPORT)#layer sensitivity report

print(layer_df.head())

channel_dict = torch.load(CHANNEL_REPORT)#channel sensitivity dictionary

# Making pruning plan

pruning_plan = {}

summary = []

for _, row in layer_df.iterrows():

    layer_name = row["Layer_Name"]

    layer_drop = float(row["Accuracy_Drop"])

    if layer_drop > 5:#Ignoring Highly sensitive layers
        print(f"Skipping {layer_name}")
        continue

    if layer_name not in channel_dict:#Checking if layers are analysed during channel sensitivity analysis
        print(f"{layer_name} not found")
        continue
    channels = channel_dict[layer_name]#List of channels in layer
    total_channels = len(channels)

    ratio = get_pruning_ratio(layer_drop)#getting prunning ratio value can be found through multiple iterations
    prune_count = math.floor(total_channels * ratio)
    prune_count = min(prune_count, total_channels-1)#ensuring pruning count doesn't exceeds

    prune_channels = [

        item["channel"]

        for item in channels[:prune_count]

    ]#selecting channels to prune
    pruning_plan[layer_name] = {

        "layer_drop": layer_drop,

        "ratio": ratio,

        "total_channels": total_channels,

        "prune_channels": prune_channels,

        "keep_channels":[

            item["channel"]

            for item in channels[prune_count:]

        ]

    }
    summary.append({

        "Layer":layer_name,

        "Layer_Drop":layer_drop,

        "Ratio":ratio,

        "Total_Channels":total_channels,

        "Pruned":len(prune_channels),

        "Remaining":total_channels-len(prune_channels)

    })
    
# Converting Summary to DataFrame inorder to save as csv/excel

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(
    by="Layer_Drop",
    ascending=True
).reset_index(drop=True)

print(summary_df)

# Save CSV
csv_path = os.path.join(
    OUTPUT_DIR,
    "pruning_plan.csv"
)

summary_df.to_csv(
    csv_path,
    index=False
)

# Save Excel

excel_path = os.path.join(
    OUTPUT_DIR,
    "pruning_plan.xlsx"
    )

summary_df.to_excel(
    excel_path,
    index=False
    )

print("Excel Saved")

# Save prunning plan dictionary in torch file which will be used for prunning

pt_path = os.path.join(
    OUTPUT_DIR,
    "pruning_plan.pt"
)
torch.save(
    pruning_plan,
    pt_path
)
print("Torch File Saved")

# Saving info in text file for further references
report_path = os.path.join(
    OUTPUT_DIR,
    "pruning_report.txt"
)

with open(report_path, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("YOLOv8 Channel Pruning Plan\n")
    f.write("=" * 70 + "\n\n")

    for layer_name, info in pruning_plan.items():

        f.write(f"Layer : {layer_name}\n")
        f.write(f"Layer Sensitivity : {info['layer_drop']:.4f}\n")
        f.write(f"Pruning Ratio : {info['ratio']:.2f}\n")
        f.write(f"Total Channels : {info['total_channels']}\n")
        f.write(f"Channels Pruned : {len(info['prune_channels'])}\n")
        f.write(f"Channels Kept : {len(info['keep_channels'])}\n")
        f.write("\n")

print("Text Report Saved")

    