# Yolov8_structured_prunning
***1.Project Overview:***

This project focuses on structured (coarse-grained) pruning of a pre-trained YOLOv8 Classification model to reduce computational complexity while preserving classification accuracy.
The original YOLOv8 classification model was previously trained for a real-time object classification application. The complete data collection, image annotation, data augmentation, and model training pipeline were performed separately and are not part of this project. This repository begins with an already trained model and concentrates entirely on the pruning, fine-tuning, and performance evaluation stages.
Original Model Statistics
The pre-trained YOLOv8 classification model used in this project has the following characteristics:
Top-1 Accuracy:92.03%
Trainable Parameters:1,451,098
MACs (Multiply-Accumulate Operations):3,958,602
Model Type	YOLOv8 Classification

***2.YOLOv8 architecture:***

Before performing structured pruning, it is important to understand the architecture of the YOLOv8 classification model. A clear understanding of each building block and the data flow through the network helps explain how pruning affects the model's parameters, computational complexity, and overall classification performance.
<img src="Yolo_architecture/YOLOv8_class.png" width="700">
The above figure illustrates the architecture of the YOLOv8 Nano classification model. As shown, the network consists of a total of 11 backbone blocks followed by a classification head. The figure below presents the internal architecture of each of these blocks, providing a detailed view of their constituent layers and connections.
<img src="Yolo_architecture/C2F.png" width="700">
In summary, the YOLOv8 Nano classification model contains a total of 26 convolutional layers, each having a different number of output channels. These convolutional layers progressively extract hierarchical features from the input image, producing a rich feature map. The final feature map is then passed to the classification head, which performs global feature aggregation and generates the final class prediction.

###3.Channel Sensitivity Analysis and Structured Pruning Strategy:

For prunning model I did sensitivity analysis of each layer by simply making it's activation zeros to check how much accuracy drop we get while performing this sensitivity analysis only one layer at a time was considered how ever when we will be doing actual prunning we will prune multiple layers so actual loss may be higher alsoa.Once we get sensitivity analysis report we need to select layers upto which we can prune i have selected layers which was giving loss upto 5% was selected by me for prunning so. you can see that in graph of sensitivity analysis report below.
<img src="sensitivity_results/layer_sensitivity.png" width="700">
After analyzing the architecture of the YOLOv8 classification model, it was observed that 12 convolutional layers were suitable candidates for structured channel pruning. These layers were selected because pruning them can significantly reduce the computational complexity of the model while preserving the overall network architecture.
The next step was to perform channel sensitivity analysis on each of these layers. During this process, every output channel of a convolutional layer was temporarily removed, and the resulting degradation in the model's output was measured. Channels that caused only a small change in the model's predictions were considered less important, whereas channels causing a large degradation were considered highly important. The complete sensitivity graphs for all eligible layers are available in the channel_sensitivity_results folder of this repository.

####4.Pruning Plan Generation:

Based on the sensitivity analysis, a structured pruning plan was created. The purpose of this pruning plan was to determine:
1)The order in which layers should be pruned.
2)The percentage of channels to remove from each layer.
3)The specific output channels that should be removed.
The pruning process follows the principle of removing the least sensitive layers first. Therefore, all eligible layers were sorted in ascending order of their layer sensitivity, ensuring that pruning begins with layers having the smallest impact on the model's predictions.
Within each selected layer, the output channels were also sorted according to their individual sensitivity scores. Channels producing the least change in the network output were selected for removal before more important channels.
To control the pruning intensity, different pruning ratios were assigned according to the measured layer sensitivity:
1)Layer sensitivity between 1 and 2: prune 60% of the output channels.
2)Layer sensitivity between 2 and 4: prune 60% of the output channels.
3)Layer sensitivity between 4 and 5: prune 50% of the output channels.

Using these thresholds, an automated pruning plan was generated for all eligible convolutional layers.

###5.Building a Torch-Pruning Compatible Model:

The original YOLOv8 model provided by the Ultralytics framework could not be directly used with the Torch-Pruning library because its internal architecture was not fully supported by the dependency graph generation mechanism.
To overcome this limitation, the complete YOLOv8 classification architecture was reimplemented from scratch using pure PyTorch (nn.Module and nn.Sequential) while preserving the original architecture exactly.
Before using this custom implementation for pruning, several verification steps were performed:
1)Every layer name was matched with the corresponding layer in the original Ultralytics model.
2)All pretrained weights were copied from the original model into the custom implementation.
3)The outputs of every intermediate layer were compared to ensure numerical equivalence.
4)Final classification logits and prediction probabilities were verified to be identical.
5)The total number of trainable parameters and the model architecture were confirmed to match the original implementation exactly.

These validation steps ensured that the custom sequential model behaved identically to the original YOLOv8 classification model while being fully compatible with Torch-Pruning.

###6.Dependency Graph Construction:

Once the sequential model was verified, Torch-Pruning was able to successfully construct the dependency graph.
The dependency graph is a crucial component of structured pruning because convolutional layers are often interconnected through concatenation operations, residual connections, normalization layers, and downstream convolutions. Removing channels from one layer may require corresponding changes in several dependent layers.
Using the dependency graph, Torch-Pruning automatically identifies all layers affected by a pruning operation and propagates the necessary structural modifications throughout the network. This guarantees that the pruned model remains structurally valid and can still perform forward inference correctly after each pruning step.

###7.Iterative Structured Pruning and Fine-Tuning:

After generating the pruning plan and constructing the dependency graph, the actual pruning process was carried out iteratively.
For each eligible convolutional layer:
1)The next layer was selected according to the pruning plan.
2)The specified output channels were removed using Torch-Pruning.
3)The dependency graph automatically updated all connected layers to maintain a valid network architecture.
4)The pruned model was fine-tuned for 8 epochs using a batch size of 64 to recover the loss in classification accuracy caused by pruning.
5)The model's accuracy, parameter count, and computational complexity were evaluated after fine-tuning.
6)After fine-tuning, the model's accuracy was evaluated. If the accuracy dropped below the predefined threshold, the pruning operation was considered unsuccessful, and the model was rolled back to its previous state.
The process was repeated for the next layer until all 12 eligible convolutional layers had been processed.

This iterative prune-and-fine-tune strategy enabled the model to gradually adapt to the reduced architecture, minimizing the loss in classification performance while significantly reducing the model size and computational cost.

###8.Conclusion:
In summary, 12 convolutional layers were identified as candidates for structured pruning based on the layer sensitivity analysis. During iterative pruning, Torch-Pruning successfully pruned 6 of these layers. The remaining 6 layers were skipped because the dependency graph identified them as having structural dependencies (primarily residual and concatenation connections), making channel removal unsafe without breaking the computational graph. As a result, the final pruned model retained only the structurally valid pruning operations while preserving network correctness.

Final statistics of the pruned model are as follows:
Validation Top-1 Accuracy: 92.56%
Original Parameters: 1,451,098
Final Parameters: 1,065,798
Parameters Removed: 385,300
Parameter Reduction: 26.55%
Original MACs: 3,958,602
Final MACs: 2,619,638
MAC Reduction: 33.82%
Successful Pruning Operations: 6
Skipped Pruning Operations: 6
Inference time of original model on cpu:
Inference time of prunned model on cpu:








