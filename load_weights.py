'''import torch
from ultralytics import YOLO

from classifier import YOLOv8Classifier

#checking names and all parameters of all module matches in model made by use and original yolo v8 classifier model

original_model = YOLO("best.pt").model
original_model.eval()



own_model = YOLOv8Classifier(
    num_classes=10
)# model built by us

own_model.eval()



original_state = original_model.state_dict()# Get State Dictionaries

own_state = own_model.state_dict()



print("Original Parameters :", len(original_state))

print("Native Parameters   :", len(native_state))


# Compare Keys



original_keys = list(original_state.keys())

own_keys = list(own_state.keys())

common = 0

for ok, nk in zip(original_keys, own_keys):

    if ok == nk:

        common += 1

        print(f"[MATCH]   {ok}")

    else:

        print()

        print(f"[ORIGINAL] {ok}")

        print(f"[NATIVE ]  {nk}")

        print()

print()

print(f"Matched Keys : {common}/{len(original_keys)}")

# Keys Missing in Native Model

print()

print("=" * 80)
print("Missing in Native Model")
print("=" * 80)

for key in original_keys:

    if key not in own_state:

        print(key)


# Extra Keys

for key in own_keys:

    if key not in original_state:

        print(key)'''

from ultralytics import YOLO
from classifier import YOLOv8Classifier
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

original_model = YOLO("best.pt").model.to(device).eval()
native_model = YOLOv8Classifier(
    num_classes=10
).to(device).eval()
native_model.load_state_dict(
    original_model.state_dict(),
    strict=True
)

print("Weights loaded successfully.")
torch.save(
    native_model.state_dict(),
    "native_classifier_weights.pt"
)

print("Weights saved.")