import torch
import torch.nn.functional as F

from utils.helper_functions.torch_utils import weights_init


def get_norm_layer(ch, name):
    if name == "batch":
        return torch.nn.BatchNorm2d(ch)
    elif name == "instance":
        return torch.nn.InstanceNorm2d(ch, affine=True)
    elif name == "instancefixed":
        return torch.nn.InstanceNorm2d(ch, affine=False)
    raise KeyError("norm unsupported", name)


class FNet(torch.nn.Module):
    def __init__(self, n_in_channels = 1,  n_out_channels = 1, norm_type="batch"):
        super().__init__()

        mult_chan = 32
        depth = 4
        self.net_recurse = _Net_recurse(
            n_in_channels=n_in_channels, mult_chan=mult_chan, depth=depth, norm_type=norm_type
        )
        self.conv_out = torch.nn.Conv2d(
            mult_chan, n_out_channels, kernel_size=3, padding=1
        )

        self.apply(weights_init)

    def forward(self, x):
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0, norm_type="batch"):
        """Class for recursive definition of U-network.p

        Parameters
        ----------
        in_channels
            Number of channels for input.
        mult_chan
            Factor to determine number of output channels
        depth
            If 0, this subnet will only be convolutions that double the channel
            count.

        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels * mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels, norm_type=norm_type)

        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(
                2 * n_out_channels, n_out_channels, norm_type=norm_type
            )
            self.conv_down = torch.nn.Conv2d(
                n_out_channels, n_out_channels, 2, stride=2
            )
            self.bn0 = get_norm_layer(n_out_channels, norm_type)
            self.relu0 = torch.nn.ReLU()
            self.convt = torch.nn.ConvTranspose2d(
                2 * n_out_channels, n_out_channels, kernel_size=2, stride=2
            )
            self.bn1 = get_norm_layer(n_out_channels, norm_type)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(
                n_out_channels, mult_chan=2, depth=(depth - 1), norm_type=norm_type
            )

    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)

            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)

            # input is CHW
            diffY = x_2conv_more.size()[2] - x_relu1.size()[2]
            diffX = x_2conv_more.size()[3] - x_relu1.size()[3]

            x_relu1 = F.pad(x_relu1, (diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2))

            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less


class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out, norm_type="batch"):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = get_norm_layer(n_out, norm_type)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = get_norm_layer(n_out, norm_type)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.relu2(x)
        return x
