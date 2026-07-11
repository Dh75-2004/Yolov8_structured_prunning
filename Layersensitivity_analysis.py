import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationValidator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best.pt"
DATASET = "fashion_dataset"

OUTPUT_DIR = "sensitivity_results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)# Create Output Directory
    yolo = YOLO(MODEL_PATH)
    model = yolo.model.to(DEVICE)
    model.eval()
    
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
    )# Validator

    metrics = validator(model=model)
    #print(metrics['metrics/accuracy_top1'])
    baseline_acc = metrics['metrics/accuracy_top1']#Basline accuracy/Original model accuracy
    print(f"\nBaseline Accuracy :{baseline_acc}")

    # Finding All Conv Layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    print(f"\nNumber of Convolution Layers : {len(conv_layers)}")

    # Sensitivity Analysis for layers
    results = []
    for idx, (name, layer) in enumerate(conv_layers):
        def hook(module, input, output):
            return torch.zeros_like(output)
        handle = layer.register_forward_hook(hook)
        metrics = validator(model=model)
        acc = metrics['metrics/accuracy_top1']
        drop = baseline_acc - acc
        results.append(
            {
                "Layer_Index": idx,
                "Layer_Name": name,
                "Top1_Accuracy": float(acc),
                "Accuracy_Drop": float(drop),
            }
        )
        print(f"Accuracy = {acc}")
        print(f"Drop     = {drop}")
        handle.remove()

    # convert result list to dataframe
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(
        by="Accuracy_Drop",
        ascending=False
    ).reset_index(drop=True)

    print("Layer Sensitivity Results")
    print(df_sorted)
    csv_path = os.path.join(
        OUTPUT_DIR,
        "layer_sensitivity.csv"
    )

    df_sorted.to_csv(csv_path, index=False)
    excel_path = os.path.join(
        OUTPUT_DIR,
        "layer_sensitivity.xlsx"
    )

    df_sorted.to_excel(excel_path, index=False)

    # Save Torch File
    pt_path = os.path.join(
        OUTPUT_DIR,
        "layer_sensitivity.pt"
    )

    torch.save(
        {
            "baseline_accuracy": baseline_acc,
            "results": results,
        },
        pt_path,
    )

    # Plot
    plt.figure(figsize=(15, 6))

    plt.bar(
        range(len(df_sorted)),
        df_sorted["Accuracy_drop"]
    )

    plt.xticks(
        range(len(df_sorted)),
        df_sorted["Layer_name"],
        rotation=90,
        fontsize=8
    )

    plt.ylabel("Accuracy Drop")
    plt.xlabel("Convolution Layers")
    plt.title("YOLOv8 Layer Sensitivity Analysis")

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        "layer_sensitivity.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    main()