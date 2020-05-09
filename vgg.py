"""
    vgg
    ~~~


"""


import torch
import torch.nn as nn

from module import *

cfg = {
    'dfm_vgg16': [64, 64, 64, 'M',\
            128, 128, 128, 'M',\
            256, 256, 256, 'M',\
            512, 512, 512, 'M',\
            512, 512, 512, 'M'],
}


class dfm_vgg16(nn.Module):
    def __init__(self, vgg_name, n_classes, 
                channel_reduction, base_reduction, hidden_reduction, 
                att_tau, p_att, p_cls, dfm_start):
        super(dfm_vgg16, self).__init__()
        self.n_classes = n_classes
        self.features = self._make_layers(cfg = cfg[vgg_name],
                                    att_tau   = att_tau, 
                                    channel_r = channel_reduction,
                                    base_r    = base_reduction, 
                                    hidden_r  = hidden_reduction,
                                    p_att     = p_att,
                                    dfm_start = dfm_start)
        if p_cls == 0:
            self.classifier = nn.Linear(512//channel_reduction, n_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_cls),
                nn.Linear(512//channel_reduction, n_classes),
                )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, att_tau, 
                     channel_r, base_r, hidden_r, p_att, dfm_start):
        layers = []
        c_in = 3
        if 'entire' in dfm_start.lower():
            barrier = 1
        else:
            dfm_count = int(dfm_start)   # (dfm_count)-th "CONV" layer
            barrier = dfm_count - 1 + dfm_count//4  # (barrier)-th layer

        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif i < barrier :
                layers += [nn.Conv2d(c_in, x, 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                c_in = x
                continue
            else:
                x_out = x//channel_r
                layers += [DFMConv2d(\
                              c_in    = c_in,
                              c_out   = x_out,
                              kernel_size = 3,
                              n_base  = x//base_r,
                              hidden  = x//hidden_r,
                              att_tau = att_tau,
                              p_att   = p_att   ),
                    nn.BatchNorm2d(x_out),
                    nn.ReLU(inplace=True)]
                c_in = x_out
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class vgg16(nn.Module):
    '''
    VGG Model
    '''
    def __init__(self, vgg_name, n_classes):
        super(vgg16, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

