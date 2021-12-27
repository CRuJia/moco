import torch
import torch.nn as nn

# https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/weight_init.html
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _init_weights(module, init_linear="normal", std=0.01, bias=0.0):
    assert init_linear in ["normal", "kaiming"], "Undefined init_linear: {}".format(
        init_linear
    )
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == "normal":
                normal_init(m, std=std, bias=bias)
            else:  # kaiming init
                kaiming_init(m, mode="fan_in", nonlinearity="relu")
        elif isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DenseCLNeck(nn.Module):
    """
    The non-linear neck in DenseCL.
    Single and dense in parallel: fc-relu-fc, conv-relu-conv
    code reference: https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/necks.py
    """

    def __init__(self, in_channels, hid_channels, out_channels, num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # mlp in moco v2, for global feature
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels),
        )
        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        # mlp2 for dense feature
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1),
        )
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear="normal"):
        _init_weights(self, init_linear)

    def forward(self, x):
        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x)  # sxs
        x = self.mlp2(x)  # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd
        return [avgpooled_x, x, avgpooled_x2]
