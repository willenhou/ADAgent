import torch
import torch.nn as nn
import os
import random
import nibabel
import numpy as np
from scipy import ndimage
import torch.nn.functional as F
from functools import partial

from einops import rearrange
eps = 1e-5 ### use for finding zero area

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=3):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model

def drop_invalid_range(volume):
    """
    Cut off the invalid area (i.e. zero area)
    """
    #org_z = volume.shape[0] // 2
    zero_value = volume.min()
    non_zeros_idx = np.where((volume - zero_value) > eps)
    
    [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
    
    ret = volume[min_z:max_z, min_h:max_h, min_w:max_w]
    #plt.matshow(ret[(org_z-min_z)])
    #plt.show()

    return ret

def itensity_normalize_one_volume(volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[(volume - volume.min()) > eps]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape) ## strange for padding zero voxels with Gaussian
        out[(volume - volume.min()) <= eps] = out_random[(volume - volume.min()) <= eps]
        return out

def resize_data(data):
    """
    Resize the data to the input size
    """ 
    [depth, height, width] = data.shape
    scale = [128*1.0/depth, 128*1.0/height, 128*1.0/width]  
    data = ndimage.zoom(data, scale, order=0)

    return data

def training_img_process(data): 

    # crop data according net input size
    data = data.get_fdata().transpose(2, 1, 0)

    
    # drop out the invalid range
    data = drop_invalid_range(data)

    # resize data
    data = resize_data(data)
    # label = self.__resize_data__(label)

    # normalization datas
    data = itensity_normalize_one_volume(data)

    return data

def lock_random_seed(seed): 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     dilation=dilation,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class feature_extractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = conv1x1x1(1, 32)
        self.bn1 = nn.InstanceNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1= nn.MaxPool3d(3, stride=2)

        self.conv2 = conv3x3x3(32, 256)
        self.bn2 = nn.InstanceNorm3d(256)
        self.maxpool2= nn.MaxPool3d(3, stride=2)

        self.conv3 = conv3x3x3(256, 512, dilation=2)
        self.bn3 = nn.InstanceNorm3d(512)
        self.maxpool3= nn.MaxPool3d(3, stride=2)

        self.conv4 = conv3x3x3(512, 1024, dilation=5)
        self.bn4 = nn.InstanceNorm3d(1024)
        self.maxpool4= nn.AdaptiveAvgPool3d((4, 4, 4))
        self.transform = nn.Linear(1024, 768)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool4(x)
        
        x = rearrange(x, 'b c h w d -> b c (h w d)')

        x = x.permute(0,2,1)
        x = self.transform(x)

        return x
    
class DotProductAttention(nn.Module) :
    def __init__(self, dropout) :
        super().__init__()
        self.dropout = nn.Dropout(dropout) ### dropout attention weight
    
    def forward(self, Q, K, V) :
        d = Q.shape[-1]
        scores = torch.bmm(Q, K.transpose(-1, -2)) / d ** 0.5 ### \sqrt{d}: scale the variance of <q, k> to 1
        self.attention_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), V)

class MultiHeadAttention(nn.Module) :
    def __init__(self, Q_d, K_d, V_d, hidden_d, num_heads, dropout, bias=True) :
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout=dropout)
        self.W_q = nn.Linear(Q_d, hidden_d, bias=bias) ## hidden_d = num_heads * D
        self.W_k = nn.Linear(K_d, hidden_d, bias=bias)
        self.W_v = nn.Linear(V_d, hidden_d, bias=bias)
        self.W_o = nn.Linear(hidden_d, hidden_d, bias=bias)
    
    def forward(self, Q, K, V) :
        Q = rearrange(self.W_q(Q), 'B L (H D) -> (B H) L D', H=self.num_heads)
        K = rearrange(self.W_k(K), 'B L (H D) -> (B H) L D', H=self.num_heads)
        V = rearrange(self.W_v(V), 'B L (H D) -> (B H) L D', H=self.num_heads)
        o = self.attention(Q, K, V)
        o_concat = rearrange(o, '(B H) L D -> B L (H D)', H=self.num_heads)
        return self.W_o(o_concat)

class PositionWiseFFN(nn.Module) :
    def __init__(self, input_d, hidden_d, output_d, dropout) :
        super().__init__()
        self.fc1 = nn.Linear(input_d, hidden_d)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_d, output_d)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x) :
        x = self.dropout1(self.act(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x

class EncoderBlock(nn.Module) :
    def __init__(self, num_heads=12, hidden_d=768, ffn_d=3072, dropout = 0, attention_dropout=0, norm_layer=partial(nn.LayerNorm, eps=1e-6)) :
        super().__init__()
        self.ln1 = norm_layer(hidden_d)
        self.self_attn = MultiHeadAttention(hidden_d, hidden_d, hidden_d, hidden_d, num_heads, dropout=attention_dropout)
        # self.self_attn = Attention(hidden_d, num_heads, 64)
        # self.self_attn = MultiHeadSelfAttention(hidden_d, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = norm_layer(hidden_d)
        self.ffn = PositionWiseFFN(hidden_d, ffn_d, hidden_d, dropout)

    def forward(self, x, x1) :   ### ViT: Norm Add

        x1 = self.self_attn(x, x1, x1)
        # x1 = self.self_attn(x1)
        x1 = self.dropout1(x1)
        x1 = x1 + x

        x2 = self.ln2(x1)
        x2 = self.ffn(x2)
        return self.dropout2(self.ln1(x1 + x2))
    
class AttenReduction(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.fc1  = nn.Linear(inchannel, inchannel//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2  = nn.Linear(inchannel//2, 1)
    def forward(self, x_ori):
        x = self.fc1(x_ori)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x_ori.size(0), -1)
        x = F.softmax(x, dim = 1)
        x = torch.unsqueeze(x, 2)
        out = x_ori * x

        return torch.sum(out, 1)



class MCAD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mri_extractor = feature_extractor()
        self.pet_extractor = feature_extractor()
        self.cross_attention1 = EncoderBlock()
        self.cross_attention2 = EncoderBlock()
        self.Attenre1 = AttenReduction(768)
        self.Attenre2 = AttenReduction(768)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x_mri, x_pet):
        x_mri = self.mri_extractor(x_mri)
        x_pet = self.pet_extractor(x_pet)

        x_mri1 = self.cross_attention1(x_mri, x_pet)
        x_pet1 = self.cross_attention2(x_pet, x_mri)

        x_mri1 = self.Attenre1(x_mri1)

        x_pet1 = self.Attenre2(x_pet1)

        x_fuse = x_mri1 + x_pet1

        out = self.head(x_fuse)
        #out = torch.sigmoid(out)

        return out, x_mri1, x_pet1




