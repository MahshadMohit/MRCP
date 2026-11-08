import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import (
    resnet,
    densenet,
    EfficientNetBN,
    ViT,
    SwinUNETR
)
from monai.networks.layers import Conv

class ResNet50_3D(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        self.backbone = resnet.resnet50(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes,
            pretrained=pretrained
        )

    def forward(self, x):
        return self.backbone(x)

class DenseNet121_3D(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = densenet.densenet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

class EfficientNet3D(nn.Module):
    def __init__(self, model_name="efficientnet-b0", num_classes=1):
        super().__init__()
        self.backbone = EfficientNetBN(
            model_name=model_name,
            spatial_dims=3,
            in_channels=1,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

class ConvNeXt3D(nn.Module):
    def __init__(self, num_classes=1, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(1, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(
                nn.GroupNorm(1, dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock3D(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.GroupNorm(1, dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x)
        x = x.mean(dim=[2, 3, 4])
        return self.head(x)

class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return residual + x

class ViT3D(nn.Module):
    def __init__(self, img_size=(64, 224, 224), patch_size=(8, 16, 16), num_classes=1):
        super().__init__()
        self.backbone = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            classification=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)[0]

class SwinTransformer3D(nn.Module):
    def __init__(self, img_size=(64, 224, 224), num_classes=1):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=num_classes,
            feature_size=48,
            use_checkpoint=False
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(48 * 16, num_classes)

    def forward(self, x):
        feats = self.backbone.swinViT(x)
        x = feats[-1]
        x = self.pool(x).flatten(1)
        return self.head(x)

class CNN_ViT_Hybrid(nn.Module):
    def __init__(self, num_classes=1, img_size=(64, 224, 224)):
        super().__init__()
        self.cnn = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512,
            pretrained=False
        )
        self.cnn.fc = nn.Identity()
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=(8, 16, 16),
            hidden_size=512,
            mlp_dim=2048,
            num_layers=6,
            num_heads=8,
            classification=True,
            num_classes=512
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)[0]
        if vit_feat.dim() > 2:
            vit_feat = vit_feat.mean(dim=1)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(combined)

class CNN_Swin_Hybrid(nn.Module):
    def __init__(self, num_classes=1, img_size=(64, 224, 224)):
        super().__init__()
        self.cnn = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=512,
            pretrained=False
        )
        self.cnn.fc = nn.Identity()
        self.swin = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=512,
            feature_size=48
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 48 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        swin_feats = self.swin.swinViT(x)
        swin_feat = self.pool(swin_feats[-1]).flatten(1)
        combined = torch.cat([cnn_feat, swin_feat], dim=1)
        return self.classifier(combined)

def get_model(name, num_classes=1, **kwargs):
    name = name.lower()
    if name == "resnet50":
        return ResNet50_3D(num_classes=num_classes)
    elif name == "densenet121":
        return DenseNet121_3D(num_classes=num_classes)
    elif name == "efficientnet":
        return EfficientNet3D(num_classes=num_classes)
    elif name == "convnext":
        return ConvNeXt3D(num_classes=num_classes)
    elif name == "vit":
        return ViT3D(num_classes=num_classes)
    elif name == "swin":
        return SwinTransformer3D(num_classes=num_classes)
    elif name == "cnn_vit":
        return CNN_ViT_Hybrid(num_classes=num_classes)
    elif name == "cnn_swin":
        return CNN_Swin_Hybrid(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")