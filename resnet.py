import os
import threading
import time
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef


num_classes = 1000


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self._lock = threading.Lock()

        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer6 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer7 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer8 = torch.nn.ReLU(inplace=True)
        self.layer9 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer10 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer11 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer12 = torch.nn.ReLU(inplace=True)
        self.layer13 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer14 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer15 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer19 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer20 = torch.nn.ReLU(inplace=True)
        self.layer21 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer22 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer23 = torch.nn.ReLU(inplace=True)
        self.layer24 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer25 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer27 = torch.nn.ReLU(inplace=False)
        self.layer28 = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer29 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer30 = torch.nn.ReLU(inplace=True)
        self.layer31 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer32 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer33 = torch.nn.ReLU(inplace=True)
        self.layer34 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer35 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer37 = torch.nn.ReLU(inplace=True)
        self.layer38 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer39 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer40 = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer41 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer42 = torch.nn.ReLU(inplace=True)
        self.layer43 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer44 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer45 = torch.nn.ReLU(inplace=True)
        self.layer46 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:
            out2 = self.layer2(x)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)
            out7 = self.layer7(out6)
            out8 = self.layer8(out7)
            out9 = self.layer9(out8)
            out10 = self.layer10(out5)
            out11 = self.layer11(out9)
            out12 = self.layer12(out11)
            out13 = self.layer13(out12)
            out14 = self.layer14(out10)
            out15 = self.layer15(out13)
            out15 = out15 + out14

            out17 = self.layer17(out15)
            out18 = self.layer18(out17)
            out19 = self.layer19(out18)
            out20 = self.layer20(out19)
            out21 = self.layer21(out20)
            out22 = self.layer22(out21)
            out23 = self.layer23(out22)
            out24 = self.layer24(out23)
            out25 = self.layer25(out24)
            out25 = out25 + out17
            
            out27 = self.layer27(out25)
            out28 = self.layer28(out27)
            out29 = self.layer29(out28)
            out30 = self.layer30(out29)
            out31 = self.layer31(out30)
            out32 = self.layer32(out31)
            out33 = self.layer33(out32)
            out34 = self.layer34(out33)
            out35 = self.layer35(out34)
            out35 = out35 + out27

            out37 = self.layer37(out35)
            out38 = self.layer38(out37)
            out39 = self.layer39(out38)
            out40 = self.layer40(out37)
            out41 = self.layer41(out40)
            out42 = self.layer42(out41)
            out43 = self.layer43(out42)
            out44 = self.layer44(out43)
            out45 = self.layer45(out44)
            out46 = self.layer46(out45)
            out47 = self.layer47(out46)
            out47 = out39 + out47

        print("end of stage 0 forward")
        return out47

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self._lock = threading.Lock()

        self.layer49 = torch.nn.ReLU(inplace=False)
        self.layer50 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer51 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer52 = torch.nn.ReLU(inplace=True)
        self.layer53 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer54 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer55 = torch.nn.ReLU(inplace=True)
        self.layer56 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer57 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer59 = torch.nn.ReLU(inplace=True)
        self.layer60 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer61 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer62 = torch.nn.ReLU(inplace=True)
        self.layer63 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer64 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer65 = torch.nn.ReLU(inplace=True)
        self.layer66 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer67 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self._initialize_weights()
        self.layer068 = torch.nn.ReLU(inplace=True)
        self.layer069 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer070 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer071 = torch.nn.ReLU(inplace=True)
        self.layer072 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer073 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer074 = torch.nn.ReLU(inplace=True)
        self.layer075 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer076 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer1 = torch.nn.ReLU(inplace=False)
        self.layer2 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer6 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer7 = torch.nn.ReLU(inplace=True)
        self.layer8 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer9 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer10 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer11 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.layer25 = torch.nn.ReLU(inplace=True)
        self.layer26 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer27 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer28 = torch.nn.ReLU(inplace=True)
        self.layer29 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer30 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer31 = torch.nn.ReLU(inplace=True)
        self.layer32 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer33 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:

            out49 = self.layer49(x)
            out50 = self.layer50(out49)
            out51 = self.layer51(out50)
            out52 = self.layer52(out51)
            out53 = self.layer53(out52)
            out54 = self.layer54(out53)
            out55 = self.layer55(out54)
            out56 = self.layer56(out55)
            out57 = self.layer57(out56)
            out57 = out57 + out49
            out59 = self.layer59(out57)
            out60 = self.layer60(out59)
            out61 = self.layer61(out60)
            out62 = self.layer62(out61)
            out63 = self.layer63(out62)
            out64 = self.layer64(out63)
            out65 = self.layer65(out64)
            out66 = self.layer66(out65)
            out67 = self.layer67(out66)
            out68 = out59+out67
            out069 = self.layer068(out68)
            out070 = self.layer069(out069)
            out071 = self.layer070(out070)
            out072 = self.layer071(out071)
            out073 = self.layer072(out072)
            out074 = self.layer073(out073)
            out075 = self.layer074(out074)
            out076 = self.layer075(out075)
            out077 = self.layer076(out076)
            out078 = out077 + out069
            
            out1 = self.layer1(out078)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)
            out7 = self.layer7(out6)
            out8 = self.layer8(out7)
            out9 = self.layer9(out8)
            out10 = self.layer10(out1)
            out11 = self.layer11(out10)
            out12 = out11 + out9

            out25 = self.layer25(out12)
            out26 = self.layer26(out25)
            out27 = self.layer27(out26)
            out28 = self.layer28(out27)
            out29 = self.layer29(out28)
            out30 = self.layer30(out29)
            out31 = self.layer31(out30)
            out32 = self.layer32(out31)
            out33 = self.layer33(out32)
            out33 = out33 + out25


        print("end of stage 1 forward")
        return out33

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self._lock = threading.Lock()

        self.layer35 = torch.nn.ReLU(inplace=False)
        self.layer36 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer37 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer38 = torch.nn.ReLU(inplace=True)
        self.layer39 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer40 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer41 = torch.nn.ReLU(inplace=True)
        self.layer42 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer43 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self._initialize_weights()
        self.layer45 = torch.nn.ReLU(inplace=True)
        self.layer46 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer47 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer48 = torch.nn.ReLU(inplace=True)
        self.layer49 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer50 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer51 = torch.nn.ReLU(inplace=True)
        self.layer52 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer53 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self._initialize_weights()
        self.layer54 = torch.nn.ReLU(inplace=True)
        self.layer55 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer56 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer57 = torch.nn.ReLU(inplace=True)
        self.layer58 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer59 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer60 = torch.nn.ReLU(inplace=True)
        self.layer61 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer62 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer064 = torch.nn.ReLU(inplace=False)
        self.layer065 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer066 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer067 = torch.nn.ReLU(inplace=True)
        self.layer068 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer069 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer070 = torch.nn.ReLU(inplace=True)
        self.layer071 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer072 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:

            out35 = self.layer35(x)
            out36 = self.layer36(out35)
            out37 = self.layer37(out36)
            out38 = self.layer38(out37)
            out39 = self.layer39(out38)
            out40 = self.layer40(out39)
            out41 = self.layer41(out40)
            out42 = self.layer42(out41)
            out43 = self.layer43(out42)
            out43 = out43 + out35
            out45 = self.layer45(out43)
            out46 = self.layer46(out45)
            out47 = self.layer47(out46)
            out48 = self.layer48(out47)
            out49 = self.layer49(out48)
            out50 = self.layer50(out49)
            out51 = self.layer51(out50)
            out52 = self.layer52(out51)
            out53 = self.layer53(out52)
            out53 = out53+out45
            out54 = self.layer54(out53)
            out55 = self.layer55(out54)
            out56 = self.layer56(out55)
            out57 = self.layer57(out56)
            out58 = self.layer58(out57)
            out59 = self.layer59(out58)
            out60 = self.layer60(out59)
            out61 = self.layer61(out60)
            out62 = self.layer62(out61)
            out62 = out62 + out54

            out064 = self.layer064(out62)
            out065 = self.layer065(out064)
            out066 = self.layer066(out065)
            out067 = self.layer067(out066)
            out068 = self.layer068(out067)
            out069 = self.layer069(out068)
            out070 = self.layer070(out069)
            out071 = self.layer071(out070)
            out072 = self.layer072(out071)
            out072 = out072 + out064

        print("end of stage 2 forward")
        return out072

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self._lock = threading.Lock()

        self.layer074 = torch.nn.ReLU(inplace=False)
        self.layer075 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer076 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer077 = torch.nn.ReLU(inplace=True)
        self.layer078 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer079 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer080 = torch.nn.ReLU(inplace=True)
        self.layer081 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer082 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer083 = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.layer084 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer36 = torch.nn.ReLU(inplace=False)
        self.layer37 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer38 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer39 = torch.nn.ReLU(inplace=True)
        self.layer40 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer41 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer42 = torch.nn.ReLU(inplace=True)
        self.layer43 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer44 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer46 = torch.nn.ReLU(inplace=True)
        self.layer47 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer48 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer49 = torch.nn.ReLU(inplace=True)
        self.layer50 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.layer51 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer52 = torch.nn.ReLU(inplace=True)
        self.layer53 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer54 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer56 = torch.nn.ReLU(inplace=True)
        self.layer57 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.layer60 = torch.nn.Linear(in_features=8192, out_features=1000, bias=True)
        self._initialize_weights()

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:

            out074 = self.layer074(x)
            out075 = self.layer075(out074)
            out076 = self.layer076(out075)
            out077 = self.layer077(out076)
            out078 = self.layer078(out077)
            out079 = self.layer079(out078)
            out080 = self.layer080(out079)
            out081 = self.layer081(out080)
            out082 = self.layer082(out081)
            out083 = self.layer083(out074)
            out084 = self.layer084(out083)
            out084 = out084 + out082

            out36 = self.layer36(out084)
            out37 = self.layer37(out36)
            out38 = self.layer38(out37)
            out39 = self.layer39(out38)
            out40 = self.layer40(out39)
            out41 = self.layer41(out40)
            out42 = self.layer42(out41)
            out43 = self.layer43(out42)
            out44 = self.layer44(out43)
            out44 = out44 + out36

            out46 = self.layer46(out44)
            out47 = self.layer47(out46)
            out48 = self.layer48(out47)
            out49 = self.layer49(out48)
            out50 = self.layer50(out49)
            out51 = self.layer51(out50)
            out52 = self.layer52(out51)
            out53 = self.layer53(out52)
            out54 = self.layer54(out53)
            out54 = out54 + out46
            out56 = self.layer56(out54)
            out57 = self.layer57(out56)
            out58 = out57.size(0)
            out59 = out57.view(out58, -1)
            out60 = self.layer60(out59)
        
        print("end of stage 3 forward")

        return out60

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class DistResNet(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """
    def __init__(self, workers, *args, **kwargs):
        super(DistResNet, self).__init__()

        # Put the first stage on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            Stage0,
            args = args,
            kwargs = kwargs,
            timeout=0
        )
        # Put the second stage on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            Stage1,
            args = args,
            kwargs = kwargs,
            timeout=0
        )
        # Put the third stage on workers[2]
        self.p3_rref = rpc.remote(
            workers[2],
            Stage2,
            args = args,
            kwargs = kwargs,
            timeout=0
        )
        # Put the fourth stage on workers[3]
        self.p4_rref = rpc.remote(
            workers[3],
            Stage3,
            args = args,
            kwargs = kwargs,
            timeout=0
        )

    def forward(self, x):

        out_futures = []    # List to store futures received from the workers
        tensors = x[0]
        offset = x[1]
        length = x[2]
        index = length + offset

        # Executes each stages depending on the stage each tensor is supposed to be fed into
        if offset < 1:
            rref_1 = RRef(tensors[0])
            p1_out_rref = self.p1_rref.rpc_async().forward(rref_1)
            out_futures.append(p1_out_rref)     # Maintains the order of input list stages
        if index >= 2 and offset < 2:
            rref_2 = RRef(tensors[1 - offset])
            p2_out_rref = self.p2_rref.rpc_async().forward(rref_2) 
            out_futures.append(p2_out_rref)
        if index >= 3 and offset < 3:
            rref_3 = RRef(tensors[2 - offset])
            p3_out_rref = self.p3_rref.rpc_async().forward(rref_3) 
            out_futures.append(p3_out_rref)
        if index >= 4 and offset < 4:
            rref_4 = RRef(tensors[3 - offset])
            p4_out_rref = self.p4_rref.rpc_async().forward(rref_4) 
            out_futures.append(p4_out_rref)

        out_tensors = torch.futures.wait_all(out_futures)   # Collects all the output tensors

        for i in range(len(out_tensors)):
            out_tensors[i] = torch.cat([out_tensors[i]])    # Prevents the problem from each tensor being a leaf tensor

        return out_tensors  # Changed from returning concatenated tensors to returning list of output tensors

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p3_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p4_rref.remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 128
batch_size = 1
image_w = 256
image_h = 256


def run_master():

    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet(["worker1", "worker2", "worker3", "worker4"])
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    # Uses input_list to store the intermediary tensors, and input_list feeds the tensors to the model
    input_list = []
    offset = 0  # Index to track which earlier stages are not executed. Also used to shift the input_list
    # batch_no = 0

    # Implements the Inter-batch parallelism achieved in PipeDream
    # Runs the loop num_batches + 3 times to execute the remaining + 3 stages of forwarding.
    for i in range(num_batches + 3):

        input_col = []
        inputs = ''

        if i < num_batches:
            print(f"Processing batch {i}")
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, one_hot_indices, 1)
            input_list = [inputs] + input_list
            offset = 0
        else:
            offset += 1

        # Append the offset, index, and length data to send them to the model
        input_col.append(input_list)
        input_col.append(offset)
        input_col.append(len(input_list))

        results = model(input_col) # Runs the Stages with the given inputs

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            if offset + len(results) >= 4:
                # print("backward for batch ", batch_no )
                dist_autograd.backward(context_id, [loss_fn(results[-1], labels)])
                opt.step(context_id)
                # batch_no += 1

        input_list = []     # Reset the input_list

        for j in range(0, len(results), 1):
            if j + offset + 1 == 4:     # Drop the last element of results, which is already backward propagated
                break
            input_list.append(results[j])   # Re-insert the results

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=600)

    import psutil
    p = psutil.Process()
    
    if rank == 0:
        p.cpu_affinity([0])
        print(f"Child #{rank}: Set my affinity to {rank}, affinity now {p.cpu_affinity()}", flush=True)

        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master()
    else:
        p.cpu_affinity([rank-1])
        print(f"Child #{rank}: Set my affinity to {rank}, affinity now {p.cpu_affinity()}", flush=True)

        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 5
    tik = time.time()
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    tok = time.time()
    print(f"execution time = {tok - tik}")