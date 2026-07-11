import os
import copy
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationValidator

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "best.pt"
    DATASET = "fashion_dataset"
    LAYER_REPORT = "sensitivity_results/layer_sensitivity.csv"
    OUTPUT_DIR = "channel_sensitivity_results"
    LAYER_THRESHOLD = 0.5
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yolo = YOLO(MODEL_PATH)
    model = yolo.model.to(DEVICE)
    model.eval()

    print(model.__class__.__name__)
    validator = ClassificationValidator(
        args=dict(
            data=DATASET,
            split="val",
            imgsz=28,
            batch=64,
            workers=0,
            device=DEVICE,
            verbose=False,
        )
    )#Validator

    print("Validator Created")
    # Read Layer Sensitivity Report and selecting laying to be pruned
    layer_df = pd.read_csv(LAYER_REPORT)
    print(layer_df.head())

    selected_layers = layer_df[
        layer_df["Accuracy_Drop"] < LAYER_THRESHOLD
    ].copy()

    selected_layers.reset_index(drop=True, inplace=True)

    print(selected_layers[["Layer_Name", "Accuracy_Drop"]])

    print(
        f"Selected {len(selected_layers)} Layers "
        f"having Accuracy Drop < {LAYER_THRESHOLD}"
    )

    metrics = validator(model=model)#Accuracy of original model

    baseline_acc = metrics['metrics/accuracy_top1']

    print(f"Baseline Accuracy : {baseline_acc:.4f}")

    # Collect Conv Layers

    print("point1")

    conv_layers = {}

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            conv_layers[name] = module


    print(f"Total Conv Layers : {len(conv_layers)}")

    # Verify Selected Layers

    valid_layers = []
    for _, row in selected_layers.iterrows():

        layer_name = row["Layer_Name"]

        if layer_name in conv_layers:

            layer = conv_layers[layer_name]

            print(
                f"[FOUND] {layer_name}"
                f"  Out Channels={layer.out_channels}"
            )

            valid_layers.append(
                (
                    layer_name,
                    layer
                )
            )

        else:

            print(
                f"[SKIPPED] {layer_name}"
            )

    print(
        f"Layers Selected for Channel Sensitivity : "
        f"{len(valid_layers)}"
    )

    # Channel Sensitivity Analysis (Forward Hook Version) for selected layers
    print("Point2")
    channel_results = []

    total_layers = len(valid_layers)

    for layer_idx, (layer_name, layer) in enumerate(valid_layers):


        print(f"\n[{layer_idx+1}/{total_layers}] Layer : {layer_name}")

        out_channels = layer.out_channels

        print(f"Output Channels : {out_channels}")

        # Test every output channel

        for ch in range(out_channels):
            print("point3")

            print(
                f"   Testing Channel {ch+1}/{out_channels}",
                end="\r"
            )
            # Forward Hook
            def make_hook(channel_idx):

                def hook(module, input, output):

                    output = output.clone()

                    output[:, channel_idx, :, :] = 0

                    return output

                return hook

            handle = layer.register_forward_hook(
                make_hook(ch)
            )
            metrics = validator(model=model)#validation
            acc = metrics["metrics/accuracy_top1"]
            drop = baseline_acc - acc
            handle.remove()
            channel_results.append({

                "Layer_Index": layer_idx,

                "Layer_Name": layer_name,

                "Channel_Index": ch,

                "Top1_Accuracy": float(acc),

                "Accuracy_Drop": float(drop),

                "Input_Channels": layer.in_channels,

                "Output_Channels": layer.out_channels,

                "Kernel_Size": list(layer.kernel_size),

                "Stride": list(layer.stride),

                "Padding": list(layer.padding),

                "Groups": layer.groups,

                "Bias": layer.bias is not None,

                "Weight_Shape": list(layer.weight.shape)

            })

        print(f"Completed Layer : {layer_name}")

    print("\nChannel Sensitivity Analysis Finished")

    # Converting Results to DataFrame
    df = pd.DataFrame(channel_results)
    df_sorted = df.sort_values(
        by=["Layer_Name", "Accuracy_Drop"],
        ascending=[True, False]
    ).reset_index(drop=True)
    print(df_sorted.head())

    # Save CSV

    csv_path = os.path.join(
        OUTPUT_DIR,
        "channel_sensitivity.csv"
    )

    df_sorted.to_csv(
        csv_path,
        index=False
    )

    print("CSV Saved")
    # Save Excel

    excel_path = os.path.join(
        OUTPUT_DIR,
        "channel_sensitivity.xlsx"
        )
    df_sorted.to_excel(
        excel_path,
        index=False
        )
    print("Excel Saved")

    # Save Torch File

    pt_path = os.path.join(
        OUTPUT_DIR,
        "channel_sensitivity.pt"
    )

    torch.save(
    {
        "baseline_accuracy": baseline_acc,

        "layer_threshold": LAYER_THRESHOLD,

        "results": channel_results

    },
    pt_path
    )
    print("Torch File Saved")

    # Build Pruning Dictionary

    pruning_dict = {}

    for _, row in df_sorted.iterrows():

        layer = row["Layer_Name"]

        channel = int(row["Channel_Index"])

        drop = float(row["Accuracy_Drop"])

        if layer not in pruning_dict:

            pruning_dict[layer] = []

        pruning_dict[layer].append(
            {
                "channel": channel,
                "accuracy_drop": drop
            }
        )

    # Sort Every Layer

    for layer in pruning_dict:

        pruning_dict[layer] = sorted(

            pruning_dict[layer],

            key=lambda x: x["accuracy_drop"]

        )

    # Save Dictionary

    dict_path = os.path.join(

        OUTPUT_DIR,

        "channel_pruning_dictionary.pt"

    )

    torch.save(

        pruning_dict,

        dict_path

    )

    print("Dictionary Saved")

    # Plot Each Layer

    for layer in df_sorted["Layer_Name"].unique():

        temp = df_sorted[
            df_sorted["Layer_Name"] == layer
        ]

        plt.figure(figsize=(12,5))

        plt.bar(

            temp["Channel_Index"],

            temp["Accuracy_Drop"]

        )

        plt.xlabel("Channel")

        plt.ylabel("Accuracy Drop")

        plt.title(layer)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        filename = layer.replace(".", "_")

        plt.savefig(

            os.path.join(

                OUTPUT_DIR,

                filename + ".png"

            ),

            dpi=300

        )

        plt.close()

    # Overall Plot

    plt.figure(figsize=(16,6))

    plt.hist(

        df_sorted["Accuracy_Drop"],

        bins=30

    )

    plt.xlabel("Accuracy Drop")

    plt.ylabel("Frequency")

    plt.title("Distribution of Channel Sensitivity")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            OUTPUT_DIR,

            "channel_sensitivity_distribution.png"

        ),

        dpi=300

    )

    plt.close()

if __name__ == "__main__":

    main()