import os
import copy
import math
import time
import torch
import torch.nn as nn
import torch_pruning as tp
import pandas as pd
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationValidator
from graphviz import Digraph
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from classifier import YOLOv8Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "best.pt"

DATASET = "fashion_dataset"

PRUNING_PLAN = "pruning_plan/pruning_plan.pt"

OUTPUT_DIR = "iterative_pruning"

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MAX_ACCURACY_DROP = 5.0    
SAVE_EVERY_LAYER = True

STOP_ON_FAILURE = False

class PruningEngine:

    def __init__(self):

        self.device = DEVICE
        self.model = YOLOv8Classifier(num_classes=10).to(DEVICE)

        self.model.load_state_dict(
        torch.load(
        "native_classifier_weights.pth",
        map_location=DEVICE
        )
        )

        self.model.eval()
        

        self.pruning_plan = None

        self.module_dict = {}

        self.dependency_graph = None

        self.example_inputs = None

        self.layer_sequence = []

        self.current_layer = 0

        self.statistics = []

        self.total_params_before = 0

        self.total_params_after = 0

        self.total_macs_before = 0

        self.total_macs_after = 0
        self.train_epochs = 8
        self.train_lr = 1e-4
        self.batch = 64
        self.imgsz = 28
        self.best_accuracy = 0

    def validator(self):
        transform = transforms.Compose([
        transforms.Resize((32, 32)),      # same size as YOLO validator
        transforms.ToTensor(),
        ])
        dataset = ImageFolder(
        root="fashion_dataset/val",
        transform=transform
        )
        correct_native = 0
        sample_index = 0
        total = 0
        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(self.device)
                labels = labels.to(self.device)
                out1 = self.model(images)
                if isinstance(out1, tuple):
                    out1 = out1[0]
                pred1 = out1.argmax(dim=1)
                correct_native += pred1.eq(labels).sum().item()
                sample_index += labels.size(0)
                total += labels.size(0)
        print(f"Native Accuracy        : {100.0 * correct_native / total:.4f}%")
        accuracy = 100.0 * correct_native / total
        return accuracy
    

    def baseline_accuracy(self):
        self.baseline_accuracy = self.validator()

    def load_pruning_plan(self):

        self.pruning_plan = torch.load(

            PRUNING_PLAN,

            map_location=self.device

        )

        print(

            f"Layers Loaded : "

            f"{len(self.pruning_plan)}"

        )

    def build_module_dictionary(self):

        self.module_dict.clear()
        print("module_dicitionary build up")

        for name, module in self.model.named_modules():
            self.module_dict[name] = module

        print(f"Total Modules : {len(self.module_dict)}")

    def build_layer_sequence(self):

        self.layer_sequence = []
        

        for layer_name, info in self.pruning_plan.items():

            if layer_name not in self.module_dict:

                print(f"[SKIPPED] {layer_name}")

                continue

            module = self.module_dict[layer_name]

            if not isinstance(module, nn.Conv2d):

                print(f"[NOT CONV] {layer_name}")

                continue

            self.layer_sequence.append({

                "layer_name": layer_name,

                "module": module,

                "layer_drop": info["layer_drop"],

                "ratio": info["ratio"],

                "prune_channels": info["prune_channels"],

                "keep_channels": info["keep_channels"]

            })

        self.layer_sequence.sort(

            key=lambda x: x["layer_drop"]

        )

        '''for i, layer in enumerate(self.layer_sequence):

            print(

                f"{i+1:02d}. "

                f"{layer['layer_name']}"

                f"  Drop={layer['layer_drop']:.4f}"

            )'''

    def create_dummy_input(self):

        self.example_inputs = (

            torch.randn(

                1,

                3,

                28,

                28,

                device=self.device

            ),

        )


    def build_dependency_graph(self):

        self.dependency_graph = tp.DependencyGraph()

        self.dependency_graph.build_dependency(

            self.model,

            example_inputs=self.example_inputs

        )

        print("Dependency Graph Ready")
        dot = Digraph()

        for node in self.dependency_graph.module2node.values():

            current = str(id(node))

            dot.node(current, str(node.module))

            for out in node.outputs:

                child = str(id(out))

                dot.node(child, str(out.module))

                dot.edge(current, child)

        dot.render("dependency_graph", format="png")

        '''for node in self.dependency_graph.module2node.values():
            print("Dependancy graph")
            print(node)
            print("Inputs :")
            for inp in node.inputs:
                print("   ", inp)

            print("Outputs :")

            for out in node.outputs:
                print("   ", out)'''

    #rebuilding dependency  graph after each teration
    def rebuild_dependency_graph(self):
        print("Rebuilding Dependency Graph...")
        del self.dependency_graph
        torch.cuda.empty_cache()
        self.dependency_graph = tp.DependencyGraph()
        self.dependency_graph.build_dependency(
            self.model,
            example_inputs=self.example_inputs
        )
        print("Dependency Graph Updated")

    def count_parameters(self):

        total = sum(

            p.numel()

            for p in self.model.parameters()

        )

        trainable = sum(

            p.numel()

            for p in self.model.parameters()

            if p.requires_grad

        )

        return total, trainable

    def count_macs(self):

        try:

            macs, params = tp.utils.count_ops_and_params(

                self.model,

                self.example_inputs[0]

            )

            return macs, params

        except Exception as e:

            print(e)

            return None, None

    def print_statistics(self):

        total, trainable = self.count_parameters()

        macs, params = self.count_macs()

        print(f"Parameters : {total:,}")

        print(f"Trainable  : {trainable:,}")

        if macs is not None:

            print(f"MACs       : {macs:,}")

    def get_module(self, layer_name):

        if layer_name not in self.module_dict:

            return None

        return self.module_dict[layer_name]
    #changes required in below method
    '''def fine_tune(self):
        print("=" * 80)
        print("Fine-tuning Started")
        print("=" * 80)

        # Use current pruned model
        self.yolo.model = self.model

        results = self.yolo.train(
            data=DATASET,
            epochs=self.train_epochs,
            imgsz=self.imgsz,
            batch=self.batch,
            lr0=self.train_lr,
            device=self.device,
            workers=0,
            optimizer="AdamW",
            verbose=False,
            val=True,
            save=False,
            pretrained=False
        )

        # Retrieve updated model after training
        self.model = self.yolo.model.to(self.device)
        self.model.eval()

        self.build_module_dictionary()
        self.rebuild_dependency_graph()

        print("=" * 80)
        print("Fine-tuning Finished")
        print("=" * 80)'''
    def fine_tune(self):
        print("fine tunning started")
        transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        ])

        train_dataset = ImageFolder(
            root="fashion_dataset/train",
            transform=transform
        )

        val_dataset = ImageFolder(
            root="fashion_dataset/val",
            transform=transform
        )
        train_loader = DataLoader(
        train_dataset,
        batch_size=self.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_lr,
            weight_decay=5e-4
        )
        best_acc = 0.0

        self.model.train()
        i = 1

        for epoch in range(self.train_epochs):
            print("epcho no:"+str(i))
            # Train

            self.model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                pred = outputs.argmax(dim=1)

                train_correct += pred.eq(labels).sum().item()

                train_total += labels.size(0)

            train_acc = 100.0 * train_correct / train_total

            self.model.eval()

    def prune_single_layer(self, layer_info):

        layer_name = layer_info["layer_name"]

        prune_channels = sorted(
            list(set(layer_info["prune_channels"]))
        )

        print(f"Pruning Layer : {layer_name}")

        self.build_module_dictionary()

        layer = self.get_module(layer_name)

        if layer is None:

            print("Layer not found.")
            return False

        if not isinstance(layer, nn.Conv2d):

            print("Layer is no longer Conv2d.")
            return False
        current_channels = layer.out_channels

        prune_channels = [

            ch for ch in prune_channels

            if ch < current_channels

        ]

        if len(prune_channels) == 0:

            print("Nothing to prune.")

            return False

        if len(prune_channels) >= current_channels:

            print("Cannot remove every channel.")

            return False

        print(f"Current Output Channels : {current_channels}")
        print(f"Channels To Remove      : {len(prune_channels)}")
        self.rebuild_dependency_graph()
        try:
            group = self.dependency_graph.get_pruning_group(
                layer,
                tp.prune_conv_out_channels,
                idxs=prune_channels
            )

        except Exception as e:
            print(e)
            return False

        if not self.dependency_graph.check_pruning_group(group):
            print("Invalid pruning group.")
            return False
        try:
            group.prune()
            print("Layer pruned successfully.")

        except Exception as e:
            print(e)
            return False

        self.build_module_dictionary()
        layer = self.get_module(layer_name)
        if layer is not None:
            print(
                f"Remaining Output Channels : "
                f"{layer.out_channels}"
            )
        return True


    def validate_model(self):


        self.model.eval()

        accuracy = self.validator()

        drop = self.baseline_accuracy - accuracy

        print(f"Accuracy       : {accuracy:.4f}")

        print(f"Accuracy Drop  : {drop:.4f}")

        return accuracy, drop


    def save_checkpoint(

            self,

            layer_name,

            accuracy,

            drop

    ):

        checkpoint = {

            "model_state_dict": self.model.state_dict(),

            "layer_name": layer_name,

            "accuracy": accuracy,

            "accuracy_drop": drop,

            "baseline_accuracy": self.baseline_accuracy,

            "statistics": self.statistics

        }

        filename = layer_name.replace(".", "_")

        path = os.path.join(

            CHECKPOINT_DIR,

            filename + ".pt"

        )

        torch.save(

            checkpoint,

            path

        )

        print()

        print(f"Checkpoint Saved : {path}")

        return path


    def load_checkpoint(

            self,

            checkpoint_path

    ):


        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device

        )

        self.model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        self.model.eval()

        self.statistics = checkpoint["statistics"]

        print(

            "Checkpoint Restored"

        )

    def save_best_model(self):

        path = os.path.join(

            OUTPUT_DIR,

            "best_pruned_model.pt"

        )

        torch.save(

            {

                "model": self.model,

                "state_dict": self.model.state_dict()

            },

            path

        )

        print()

        print(

            f"Best Model Saved : {path}"

        )

    def log_statistics(

            self,

            layer_name,accuracy_before_ft,

            accuracy,

            drop,params,macs

    ):

        params, trainable = self.count_parameters()

        macs, _ = self.count_macs()

        log = {
        "Layer": layer_name,
        "Accuracy_After_Pruning": accuracy_before_ft,
        "Accuracy_After_Finetune": accuracy,
        "Accuracy_Drop": drop,
        "Parameters": params,
        "MACs": macs
            }

        self.statistics.append(log)

        print()

        print("=" * 80)

        print("Current Statistics")

        print("=" * 80)

        print(log)

    def save_statistics(self):

        if len(self.statistics) == 0:

            return

        df = pd.DataFrame(

            self.statistics

        )

        csv_path = os.path.join(

            OUTPUT_DIR,

            "pruning_statistics.csv"

        )

        df.to_csv(

            csv_path,

            index=False

        )

        print()

        print(

            f"Statistics Saved : {csv_path}"

        )

    def rollback(

            self,

            backup_model

    ):

        print()

        print("=" * 80)

        print("Rolling Back Model")

        print("=" * 80)

        self.model.load_state_dict(

            backup_model.state_dict()

        )

        self.model.eval()

        self.build_module_dictionary()

        self.rebuild_dependency_graph()

        print("Rollback Completed")

    def iterative_prune(self):

        print()
        print("=" * 80)
        print("Starting Iterative Structured Pruning")
        print("=" * 80)

        total_layers = len(self.layer_sequence)

        successful_layers = 0
        failed_layers = 0
        i = 0

        for index, layer_info in enumerate(self.layer_sequence):
            print(i)

            layer_name = layer_info["layer_name"]
            print(
                f"[{index+1}/{total_layers}] {layer_name}"
            )

            backup_model = copy.deepcopy(self.model)# Backup Model

            params_before, _ = self.count_parameters()
            status = self.prune_single_layer(layer_info)
            if not status:
                print("Layer skipped.")
                failed_layers += 1
                continue
            # Validate immediately after pruning

            accuracy_before_ft, drop_before_ft = self.validate_model()

            print(f"Accuracy after pruning : {accuracy_before_ft:.4f}")
            # Fine-tune
            self.fine_tune()
            # Validate after fine-tuning

            accuracy, drop = self.validate_model()

            print(f"Accuracy after fine-tuning : {accuracy:.4f}")
            print(drop)

            if drop > MAX_ACCURACY_DROP:

                print(i)

                print(
                    "Accuracy Drop Exceeded Threshold"
                )

                print(
                    f"{drop:.2f}% > "
                    f"{MAX_ACCURACY_DROP:.2f}%"
                )

                self.rollback(backup_model)

                failed_layers += 1

                if STOP_ON_FAILURE:
                    print(i)
                    break
                continue
            params_after, _ = self.count_parameters()
            removed = params_before - params_after
            mac,param  = self.count_macs()
            print(f"Parameters Removed : {removed:,}")
            self.log_statistics(
                layer_name,accuracy_before_ft,
                accuracy,
                drop,
                params_after,mac
            )
            if SAVE_EVERY_LAYER:

                self.save_checkpoint(

                    layer_name,

                    accuracy,

                    drop

                )
            i = i + 1

            successful_layers += 1
        print("Pruning Finished")
        print(f"Successful Layers : {successful_layers}")
        print(f"Failed Layers     : {failed_layers}")
        self.save_statistics()
        self.save_best_model()

    def summary(self):
        params, trainable = self.count_parameters()
        macs, _ = self.count_macs()
        print(f"Parameters : {params:,}")
        print(f"Trainable  : {trainable:,}")
        if macs is not None:
            print(f"MACs       : {macs:,}")
        print(
            f"Baseline Accuracy : "
            f"{self.baseline_accuracy:.4f}"
        )


def main():
    engine = PruningEngine()
    engine.baseline_accuracy()
    engine.load_pruning_plan()
    engine.build_module_dictionary()
    engine.create_dummy_input()
    engine.build_dependency_graph()
    engine.print_statistics()
    engine.total_params_before, _ = engine.count_parameters()
    macs, _ = engine.count_macs()
    if macs is not None:
        engine.total_macs_before = macs
    engine.build_layer_sequence()
    print(500*'*')
    engine.iterative_prune()
    print(50*'*')
    engine.total_params_after, _ = engine.count_parameters()
    macs, _ = engine.count_macs()
    if macs is not None:
        engine.total_macs_after = macs
    print("=" * 30)
    print("FINAL REPORT")
    print("=" * 30)
    print(f"Baseline Accuracy : {engine.baseline_accuracy:.4f}")
    params_removed = (
        engine.total_params_before
        - engine.total_params_after
    )
    percent_removed = (
        100.0 * params_removed
        / engine.total_params_before
    )
    print(f"Parameters Before : {engine.total_params_before:,}")
    print(f"Parameters After  : {engine.total_params_after:,}")
    print(f"Parameters Removed: {params_removed:,}")
    print(f"Reduction (%)     : {percent_removed:.2f}")
    if engine.total_macs_before != 0 and engine.total_macs_after != 0:

        mac_removed = (
            engine.total_macs_before
            - engine.total_macs_after
        )
        mac_percent = (
            100.0 * mac_removed
            / engine.total_macs_before
        )
        print(f"MACs Before : {engine.total_macs_before:,}")
        print(f"MACs After  : {engine.total_macs_after:,}")
        print(f"MAC Reduction (%) : {mac_percent:.2f}")

    # Save Final Model
    final_model_path = os.path.join(
        OUTPUT_DIR,
        "final_pruned_model.pt"
    )

    torch.save(
        {
            "state_dict": engine.model.state_dict(),
            "model": engine.model
        },
        final_model_path
    )
    print(f"Final Model Saved : {final_model_path}")
    print("Pruning Pipeline Completed Successfully")

if __name__ == "__main__":

    main()