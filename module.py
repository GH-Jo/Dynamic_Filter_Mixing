import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision


def soft_criterion(outputs, teacher_outputs, T):
    soft_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                               F.softmax(teacher_outputs/T, dim=1)) * (T*T)
    return soft_loss
    


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DFMConv2d(nn.Module):
    def __init__(self, c_in, c_out, 
                kernel_size, n_base, hidden, att_tau, p_att):
        super(DFMConv2d, self).__init__()
        self.c_out   = c_out
        self.c_in    = c_in
        self.k       = kernel_size
        self.n_base  = n_base
        self.att_tau = att_tau
        
        if p_att == 0:
            self.attention_ = nn.Sequential(
                    Flatten(),
                    nn.Linear(c_in, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, self.c_out * n_base)
                    )
        else:
            self.attention_ = nn.Sequential(
                    Flatten(),
                    nn.Dropout(p=p_att),
                    nn.Linear(c_in, hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=p_att),
                    nn.Linear(hidden, self.c_out * n_base)
                    )

        self.base_filters = nn.Parameter(nn.init.kaiming_normal_(\
                torch.rand(\
                    self.n_base, \
                    self.c_in * self.k * self.k \
                    )))
        
    def forward(self, x):
        # 1. Attention module
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)),
                            stride=(x.size(2), x.size(3)))
        mix_weight = self.attention_(avg_pool)
        mix_weight_2d = mix_weight.reshape((-1, self.c_out,\
                                            self.n_base))
        mix_weight_sfm = F.softmax(mix_weight_2d/self.att_tau, dim=2)

        # 2. Kernel generation (2nd conv2d)
        shp = mix_weight_sfm.shape
        kernel = mix_weight_sfm.reshape(shp[0], shp[1], shp[2], 1, 1)

        # 3. Conv2d (1st with base filters, 2nd with kernel)
        base_filters = self.base_filters.reshape(self.n_base,
                                                self.c_in,
                                                self.k,
                                                self.k)
        x = F.conv2d(x, base_filters, padding = self.k // 2)

        x_one_shape = (1, x.shape[-3], x.shape[-2], x.shape[-1])
        out = F.conv2d(x[0,:,:,:].view(x_one_shape),
                       kernel[0,:,:,:,:], padding=0)
        
        for i in range(1, x.shape[0]):
            out_i = F.conv2d(x[i,:,:,:].view(x_one_shape),
                            kernel[i], padding=0)
            out = torch.cat([out, out_i], dim=0)
        return out


    def tau_decay(self, tau, warmup_epoch):
        self.att_tau = self.att_tau - ((tau-1)/warmup_epoch)


