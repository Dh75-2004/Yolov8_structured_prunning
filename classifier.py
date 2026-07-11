import torch
import torch.nn as nn

from block import Conv
from block import C2f
from block import Classify


class YOLOv8Classifier(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()
        self.model = nn.Sequential(
            Conv(
                c1=3,
                c2=16,
                k=3,
                s=2
            ),
            Conv(
                c1=16,
                c2=32,
                k=3,
                s=2
            ),
            C2f(
                c1=32,
                c2=32,
                n=1
            ),
            Conv(
                c1=32,
                c2=64,
                k=3,
                s=2
            ),
            C2f(
                c1=64,
                c2=64,
                n=2
            ),
            Conv(
                c1=64,
                c2=128,
                k=3,
                s=2
            ),
            C2f(
                c1=128,
                c2=128,
                n=2
            ),
            Conv(
                c1=128,
                c2=256,
                k=3,
                s=2
            ),
            C2f(
                c1=256,
                c2=256,
                n=1
            ),
            Classify(
                c1=256,
                nc=num_classes
            )

        )


    def forward(self,x,augment=False,visualize=False,embed=None,**kwargs):
        return self.model(x)

    '''def forward(self, x):

        return self.model(x)'''
    

    def summary(self):

        print(self.model)