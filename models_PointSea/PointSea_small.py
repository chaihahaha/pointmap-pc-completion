import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from models_PointSea.model_utils import *
from pytorch3d.ops import knn_points
from torchvision.models import resnet18, ResNet18_Weights
from models_PointSea.mv_utils_zs import euler2mat
from einops import rearrange


def _half_cd(x, y):
    knn = knn_points(x, y, K=1, return_sorted=False)
    return knn.dists[..., 0]


def _scaled_resnet(scale, output_planes):
    """Build a ResNet18-style backbone with channel counts scaled by `scale`,
    and a final 1x1 conv projecting to `output_planes`."""
    import torchvision.models.resnet as _r
    base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # scale the four stage channel counts
    planes = [int(round(p * scale)) for p in [64, 128, 256, 512]]
    planes = [max(p, 8) for p in planes]
    inplanes = resnet18(weights=None).conv1.out_channels
    from torchvision.models.resnet import BasicBlock

    class _ScaledResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, planes[0], 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(planes[0])
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
            self.inplanes = planes[0]
            self.layer1 = self._make_layer(planes[0], 2, stride=1)
            self.layer2 = self._make_layer(planes[1], 2, stride=2)
            self.layer3 = self._make_layer(planes[2], 2, stride=2)
            self.layer4 = self._make_layer(planes[3], 2, stride=2)
            self.head = nn.Conv2d(planes[3], output_planes, 1)
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.eval()

        def _make_layer(self, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                    nn.BatchNorm2d(planes))
            layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes
            for _ in range(1, blocks):
                layers.append(BasicBlock(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.head(x)
            return x

    return _ScaledResNet()


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=128):
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        l0_xyz = point_cloud
        l0_points = point_cloud
        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)
        return l3_points


class ResEncoder(nn.Module):
    def __init__(self, scale=0.5, out_planes=256):
        super(ResEncoder, self).__init__()
        self.net = _scaled_resnet(scale, out_planes)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

    def forward(self, input_view):
        return self.net(input_view)


class SDG(nn.Module):
    def __init__(self, fg_dim=256, ratio=1, channel=64, hidden_dim=192):
        super(SDG, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(fg_dim, channel, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.sa1 = self_attention(channel * 2, hidden_dim, dropout=0.0, nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0, nhead=8)
        self.decoder1 = SDG_Decoder(hidden_dim, channel, ratio)
        self.decoder2 = SDG_Decoder(hidden_dim, channel, ratio)
        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel, kernel_size=1)
        self.conv_ps = nn.Conv1d(hidden_dim, channel * ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=832, layer_dims=[hidden_dim])
        self.sigma_d = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.fusionMlp = MLP_CONV(in_channel=hidden_dim * 2 + channel, layer_dims=[hidden_dim])

    def forward(self, local_feat, coarse, f_g, partial):
        batch_size, _, N = coarse.size()
        F = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(f_g))
        F = torch.cat([F, f_g.repeat(1, 1, F.shape[-1])], dim=1)
        half_cd = _half_cd(coarse.transpose(1, 2).contiguous(), partial.transpose(1, 2).contiguous()) / self.sigma_d
        embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        F_Q = self.sa1(F, embd)
        F_Q_ = self.decoder1(F_Q, embd)
        f_g_current = torch.max(F_Q, 2)[0]
        local_feat = self.mlpp(local_feat)
        F_H = self.cross1(F_Q, local_feat)
        F_H_ = self.decoder2(F_H, embd)
        score = self.fusionMlp(torch.cat([F_Q_ + F_H_, f_g_current.unsqueeze(2).repeat(1, 1, F_Q_.size(2)), f_g.repeat(1, 1, F_Q_.size(2))], 1))
        score = torch.sigmoid(score)
        F_L = score * F_Q_ + (1 - score) * F_H_
        F_L = self.conv_delta(self.conv_ps(F_L).reshape(batch_size, -1, N * self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1, 1, self.ratio) + O_L
        return fine, F_L


class SDG_l(nn.Module):
    def __init__(self, fg_dim=256, ratio=1, channel=64, hidden_dim=128):
        super(SDG_l, self).__init__()
        self.channel = channel
        self.hidden = hidden_dim
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(fg_dim, channel, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)
        self.sa1 = self_attention(channel * 2, hidden_dim, dropout=0.0, nhead=8)
        self.cross1 = cross_attention(hidden_dim, hidden_dim, dropout=0.0, nhead=8)
        self.decoder1 = SDG_Decoder(hidden_dim, channel, ratio)
        self.decoder2 = SDG_Decoder(hidden_dim, channel, ratio)
        self.relu = nn.GELU()
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_delta = nn.Conv1d(channel, channel, kernel_size=1)
        self.conv_ps = nn.Conv1d(hidden_dim, channel * ratio, kernel_size=1)
        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)
        self.mlpp = MLP_CONV(in_channel=832, layer_dims=[hidden_dim])
        self.sigma_d = 0.2
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.fusionMlp = MLP_CONV(in_channel=hidden_dim * 2 + channel * 2, layer_dims=[hidden_dim])

    def forward(self, local_feat, coarse, f_g, partial, F_L_Pre):
        batch_size, _, N = coarse.size()
        F = self.conv_x1(self.relu(self.conv_x(coarse)))
        f_g = self.conv_1(self.relu(f_g))
        F = torch.cat([F, f_g.repeat(1, 1, F.shape[-1])], dim=1)
        half_cd = _half_cd(coarse.transpose(1, 2).contiguous(), partial.transpose(1, 2).contiguous()) / self.sigma_d
        embd = self.embedding(half_cd).reshape(batch_size, self.hidden, -1).permute(2, 0, 1)
        F_Q = self.sa1(F, embd)
        F_Q_ = self.decoder1(F_Q, embd)
        f_g_current = torch.max(F_Q, 2)[0]
        local_feat = self.mlpp(local_feat)
        F_H = self.cross1(F_Q, local_feat)
        F_H_ = self.decoder2(F_H, embd)
        score = self.fusionMlp(torch.cat([F_Q_ + F_H_, F_L_Pre, f_g_current.unsqueeze(2).repeat(1, 1, F_Q_.size(2)), f_g.repeat(1, 1, F_Q_.size(2))], 1))
        score = torch.sigmoid(score)
        F_L = score * F_Q_ + (1 - score) * F_H_
        F_L = self.conv_delta(self.conv_ps(F_L).reshape(batch_size, -1, N * self.ratio))
        O_L = self.conv_out(self.relu(self.conv_out1(F_L)))
        fine = coarse.repeat(1, 1, self.ratio) + O_L
        return fine


class SVFNet(nn.Module):
    def __init__(self, cfg):
        super(SVFNet, self).__init__()
        self.fp = 64           # point feature dim
        self.fv = 128          # image feature dim
        self.channel = 16
        self.point_feature_extractor = FeatureExtractor(out_dim=self.fp)
        self.view_distance = cfg.NETWORK.view_distance
        self.relu = nn.GELU()
        self.sa = self_attention(self.channel * 8, self.channel * 8, dropout=0.0)
        self.viewattn1 = self_attention(self.fv + self.fp, self.fv)
        self.viewattn2 = self_attention(self.fv + self.fp, self.fp)
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(self.fg_dim() + self.channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(self.fg_dim(), self.channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(self.fg_dim() + self.channel, self.channel * 8, kernel_size=1)
        self.img_feature_extractor = ResEncoder(scale=0.25, out_planes=self.fv)
        self.posmlp = MLP_CONV(3, [64, self.fp])

    def fg_dim(self):
        # f_g = concat([f_p (fp), f_v_ (fp after viewattn2 pool)])
        return self.fp + self.fp

    def forward(self, points, depth):
        batch_size, _, N = points.size()
        f_v = self.img_feature_extractor(depth)
        f_v = rearrange(f_v, 'bv c h w -> bv c (h w)')
        f_p = self.point_feature_extractor(points)

        view_point = torch.tensor([0, 0, -self.view_distance, -self.view_distance, 0, 0, 0, self.view_distance, 0],
                                  dtype=torch.float32).view(-1, 3, 3).permute(0, 2, 1).repeat(batch_size, 1, 1).to(depth.device)
        view_feature_1 = self.posmlp(view_point)

        f_v_ = self.viewattn1(torch.cat([f_v, f_p.repeat(3, 1, f_v.size(2))], 1))
        f_v_ = rearrange(f_v_, '(b v) c n -> b c v n', b=batch_size)
        f_v_ = torch.max(f_v_, dim=3)[0]
        f_v_ = self.viewattn2(torch.cat([f_v_, f_p.repeat(1, 1, f_v_.size(2))], dim=1), view_feature_1.permute(2, 0, 1))
        f_v_ = F.adaptive_max_pool1d(f_v_, 1)
        f_g = torch.cat([f_p, f_v_], 1)

        x = self.relu(self.ps(f_g))
        x = self.relu(self.ps_refuse(torch.cat([x, f_g.repeat(1, 1, x.size(2))], 1)))
        x2_d = (self.sa(x)).reshape(batch_size, self.channel * 4, N // 8)
        coarse = self.conv_out(self.relu(self.conv_out1(torch.cat([x2_d, f_g.repeat(1, 1, x2_d.size(2))], 1))))
        return f_g, coarse


class local_encoder(nn.Module):
    def __init__(self, cfg):
        super(local_encoder, self).__init__()
        self.gcn_1 = EdgeConv(3, 64, 16)
        self.gcn_2 = EdgeConv(64, 256, 8)
        self.gcn_3 = EdgeConv(256, 512, 4)
        self.local_number = cfg.NETWORK.local_points

    def forward(self, input):
        x1 = self.gcn_1(input)
        idx = furthest_point_sample(input.transpose(1, 2).contiguous(), self.local_number)
        x1 = gather_points(x1, idx)
        x2 = self.gcn_2(x1)
        x3 = self.gcn_3(x2)
        return torch.cat([x1, x2, x3], 1)


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.encoder = SVFNet(cfg)
        self.localencoder = local_encoder(cfg)
        self.merge_points = cfg.NETWORK.merge_points
        fg_dim = self.encoder.fg_dim()
        self.refine1 = SDG(fg_dim=fg_dim, ratio=cfg.NETWORK.step1, channel=32, hidden_dim=96)
        self.refine2 = SDG_l(fg_dim=fg_dim, ratio=cfg.NETWORK.step2, channel=32, hidden_dim=64)

    def forward(self, partial, depth):
        partial = partial.transpose(1, 2).contiguous()
        feat_g, coarse = self.encoder(partial, depth)
        local_feat = self.localencoder(partial)
        coarse_merge = torch.cat([partial, coarse], dim=2)
        coarse_merge = gather_points(coarse_merge, furthest_point_sample(coarse_merge.transpose(1, 2).contiguous(), self.merge_points))
        fine1, F_L_1 = self.refine1(local_feat, coarse_merge, feat_g, partial)
        fine2 = self.refine2(local_feat, fine1, feat_g, partial, F_L_1)
        return (coarse.transpose(1, 2).contiguous(), fine1.transpose(1, 2).contiguous(), fine2.transpose(1, 2).contiguous())
