"""Local OSNet x1.0 implementation for the bundled sports ReID checkpoint.

The architecture is adapted from Kaiyang Zhou's torchreid OSNet implementation.
Copyright (c) 2018 Kaiyang Zhou. Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms


class _ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class _LightConv3x3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class _ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        gate_channels = max(channels // reduction, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, gate_channels, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(gate_channels, channels, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        weights = self.global_avgpool(x)
        weights = self.fc2(self.relu(self.fc1(weights)))
        return x * self.gate_activation(weights)


class _OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid_channels = out_channels // reduction
        self.conv1 = _ConvLayer(in_channels, mid_channels, kernel_size=1)
        self.conv2a = self._make_branch(mid_channels, depth=1)
        self.conv2b = self._make_branch(mid_channels, depth=2)
        self.conv2c = self._make_branch(mid_channels, depth=3)
        self.conv2d = self._make_branch(mid_channels, depth=4)
        self.gate = _ChannelGate(mid_channels)
        self.conv3 = _Conv1x1Linear(mid_channels, out_channels)
        self.downsample = (
            _Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        )

    @staticmethod
    def _make_branch(channels, depth):
        return nn.Sequential(*(_LightConv3x3(channels) for _ in range(depth)))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = (
            self.gate(self.conv2a(x))
            + self.gate(self.conv2b(x))
            + self.gate(self.conv2c(x))
            + self.gate(self.conv2d(x))
        )
        x = self.conv3(x)
        return nn.functional.relu(x + identity)


class _OSNetX1_0(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv1 = _ConvLayer(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_stage(64, 256, blocks=2, reduce_spatial_size=True)
        self.conv3 = self._make_stage(256, 384, blocks=2, reduce_spatial_size=True)
        self.conv4 = self._make_stage(384, 512, blocks=2, reduce_spatial_size=False)
        self.conv5 = _ConvLayer(512, 512, kernel_size=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_channels, out_channels, blocks, reduce_spatial_size):
        layers = [_OSBlock(in_channels, out_channels)]
        layers.extend(_OSBlock(out_channels, out_channels) for _ in range(blocks - 1))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    _ConvLayer(out_channels, out_channels, kernel_size=1),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                )
            )
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        features = self.global_avgpool(x).flatten(1)
        features = self.fc(features)
        if self.training:
            return self.classifier(features)
        return features


def _build_osnet_x1_0(num_classes=1):
    return _OSNetX1_0(num_classes=num_classes)


class SportsReIDEncoder:
    """Batched OSNet features from the McByte++ sports ReID checkpoint."""

    def __init__(self, device, weights):
        self.device = torch.device(device)
        self.weights = Path(weights).expanduser()
        self.model = None
        self.error = None
        self.transform = transforms.Compose(
            [
                transforms.Resize([256, 128]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load(self):
        if self.model is not None:
            return True
        if self.error is not None:
            return False
        try:
            if not self.weights.is_file():
                raise FileNotFoundError(f"sports ReID checkpoint not found: {self.weights}")
            self.model = _build_osnet_x1_0(num_classes=1)
            checkpoint = torch.load(
                self.weights,
                map_location="cpu",
                weights_only=False,
            )
            state = checkpoint.get("state_dict", checkpoint)
            model_state = self.model.state_dict()
            compatible = {}
            for key, value in state.items():
                model_key = key.removeprefix("module.")
                if model_key in model_state and model_state[model_key].shape == value.shape:
                    compatible[model_key] = value
            if not compatible:
                raise RuntimeError("sports ReID checkpoint has no compatible OSNet weights")
            model_state.update(compatible)
            self.model.load_state_dict(model_state)
            self.model.eval().to(self.device)
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"warning: OSNet ReID is unavailable; identity matching is disabled: {self.error}")
            self.model = None
            return False

    def encode(self, crops):
        if not crops or not self._load():
            return []
        batch = torch.stack(
            [
                self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                for crop in crops
            ]
        ).to(self.device)
        with torch.inference_mode():
            features = self.model(batch).float()
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return [row.cpu().numpy().astype(np.float32, copy=True) for row in features]
