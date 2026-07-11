import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]

    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = tuple(x // 2 for x in k)

    return p


class Conv(nn.Module):

    default_act = nn.SiLU(inplace=True)

    def __init__(self,c1, c2,k=1,s=1,p=None,g=1,d=1,act=True,bias=False,):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()


    def forward(self, x):

        return self.act(
            self.bn(
                self.conv(x)
            )
        )
    

    def forward_fuse(self, x):
        return self.act(
            self.conv(x)
        )


class Bottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,k=(3, 3),e=1.0):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(
            c1=c1,
            c2=c_,
            k=k[0],
            s=1
        )
        self.cv2 = Conv(
            c1=c_,
            c2=c2,
            k=k[1],
            s=1,
            g=g
        )
        self.add = shortcut and (c1 == c2)


    def forward(self, x):
        y = self.cv2(
            self.cv1(x)
        )
        if self.add:
            y = x + y

        return y
    

class Bottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True,g=1,k=((3, 3), (3, 3)),e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(
            c1,
            c_,
            k=k[0],
            s=1
        )
        self.cv2 = Conv(
            c_,
            c2,
            k=k[1],
            s=1,
            g=g
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):

        y = self.cv2(self.cv1(x))

        return x + y if self.add else y
    

class C2f(nn.Module):
    def __init__(self,c1,c2,n=1,shortcut=True,g=1,e=0.5):

        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(
            c1=c1,
            c2=2 * self.c,
            k=1,
            s=1
        )
        self.cv2 = Conv(
            c1=(2 + n) * self.c,
            c2=c2,
            k=1,
            s=1
        )
        self.m = nn.ModuleList(
        Bottleneck(
        self.c,
        self.c,
        shortcut,
        g,
        k=((3, 3), (3, 3)),
        e=1.0
        )
        for _ in range(n)
        )

    def forward(self, x):
        y = list(

            self.cv1(x).chunk(2, dim=1)

        )

        for block in self.m:

            y.append(

                block(

                    y[-1]

                )

            )

        y = torch.cat(

            y,

            dim=1

        )

        y = self.cv2(

            y

        )

        return y


class Classify(nn.Module):

    def __init__(self,c1,nc,dropout=0.0):

        super().__init__()
        self.conv = Conv(
            c1=c1,
            c2=1280,
            k=1,
            s=1
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.drop = nn.Dropout(
            p=dropout,
            inplace=True
        )
        self.linear = nn.Linear(
            1280,
            nc
        )

        self.export = False

    def forward(self, x):

        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        x = self.conv(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.drop(x)

        x = self.linear(x)

        if self.training:

            return x

        y = x.softmax(dim=1)

        if self.export:

            return y

        return y, x
    
