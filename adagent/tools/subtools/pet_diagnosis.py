import torch
import torch.nn as nn
import os
import random
import nibabel
import numpy as np
from scipy import ndimage
import torch.nn.functional as F
from functools import partial

import sys
import json
from pathlib import Path

# Get project root (go up 3 levels from this file: subtools -> tools -> adagent -> root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
_MODEL_DIR = _PROJECT_ROOT / "model-weights"
_TEMP_DIR = _PROJECT_ROOT / "temp"
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

def subprocess_main():
    raw_input = sys.stdin.read()
    try:
        input_data = json.loads(raw_input)
        lock_random_seed(800)

        MedicalNet = generate_model(50, n_input_channels=1).cuda()
        MedicalNet.load_state_dict(torch.load(str(_MODEL_DIR / "pet_medicalnet_800.pt")), strict=False)
        MedicalNet.eval()

        ResNet50 = generate_model(50, n_input_channels=1).cuda()
        ResNet50.load_state_dict(torch.load(str(_MODEL_DIR / "pet_resnet50_600.pt")), strict=False)
        ResNet50.eval()

        ResNet34 = generate_model(34, n_input_channels=1).cuda()
        ResNet34.load_state_dict(torch.load(str(_MODEL_DIR / "pet_resnet34_800.pt")), strict=False)
        ResNet34.eval()

        ResNet18 = generate_model(18, n_input_channels=1).cuda()
        ResNet18.load_state_dict(torch.load(str(_MODEL_DIR / "pet_resnet18_800.pt")), strict=False)
        ResNet18.eval()

        # if not os.path.isfile(args.pet_path):
        #     print('pet Image not found: ', args.pet_path)
        # assert os.path.isfile(args.pet_path)
        pet_Image = nibabel.load(input_data['pet_path']) 
        #pet_Image = nibabel.load('/home/wlhou/ADAgent/temp/upload_1753281991.nii')
        pet_Image = training_img_process(pet_Image)
        pet_Image = np.resize(pet_Image, [1, 1, 128, 128, 128])
        pet_Image = pet_Image.astype("float32")

        inputs_pet = torch.from_numpy(pet_Image)
        inputs_pet = inputs_pet.to('cuda' if torch.cuda.is_available() else 'cpu')

        MedicalNet_output = nn.functional.softmax(MedicalNet(inputs_pet), dim=1).to('cpu')
        ResNet50_output = nn.functional.softmax(ResNet50(inputs_pet), dim=1).to('cpu')
        ResNet34_output = nn.functional.softmax(ResNet34(inputs_pet), dim=1).to('cpu')
        ResNet18_output = nn.functional.softmax(ResNet18(inputs_pet), dim=1).to('cpu')

        results = {}
        results['MedicalNet'] = MedicalNet_output.detach().numpy()[0].tolist()
        results['ResNet50'] = ResNet50_output.detach().numpy()[0].tolist()
        results['ResNet34'] = ResNet34_output.detach().numpy()[0].tolist()
        results['ResNet18'] = ResNet18_output.detach().numpy()[0].tolist()
        return results, 0

    except json.JSONDecodeError:
        return {"error": "Invalid JSON input"}, 1

if __name__ == "__main__":
    response, exit_code = subprocess_main()
    # 返回 JSON 结果到标准输出
    _TEMP_DIR.mkdir(exist_ok=True)
    with open(str(_TEMP_DIR / "child_output.json"), "w") as f:
        print("success!")
        json.dump(response, f)




