import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import furthest_point_sample


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(p, idx):
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(p.size(0), dtype=torch.long)
        .to(p.device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_p = p[batch_indices, idx, :]
    return new_p


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(
        sqrdists,
        nsample,
        dim=-1,
        largest=False,
        sorted=False,
    )
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(
        self,
        channel,
        groups,
        kneighbors,
        normalize="center",
        **kwargs,
    ):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.normalize = normalize
        assert self.normalize in ("center", "anchor")
        self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel]))
        self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel]))

    def forward(self, xyz, p):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, p, xyz]

        fps_idx = furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_p = index_points(p, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_p = index_points(p, idx)  # [B, npoint, k, d]
        if self.normalize == "center":
            mean = torch.mean(grouped_p, dim=2, keepdim=True)
        elif self.normalize == "anchor":
            mean = new_p.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
        std = (
            torch.std((grouped_p - mean).reshape(B, -1), dim=-1, keepdim=True)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1)
        )
        grouped_p = (grouped_p - mean) / (std + 1e-5)
        grouped_p = self.affine_alpha * grouped_p + self.affine_beta

        new_p = torch.cat(
            [
                grouped_p,
                new_p.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1),
            ],
            dim=-1,
        )
        return new_xyz, new_p


class ConvBNReLU1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        bias=True,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channel,
        kernel_size=1,
        res_expansion=1.0,
        bias=True,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channel,
                out_channels=int(channel * res_expansion),
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act,
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(
                in_channels=int(channel * res_expansion),
                out_channels=channel,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(channel),
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        blocks=1,
        res_expansion=1,
        bias=True,
    ):
        super().__init__()
        in_channels = 2 * channels
        self.transfer = ConvBNReLU1D(
            in_channels,
            out_channels,
            bias=bias,
        )
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    out_channels,
                    res_expansion=res_expansion,
                    bias=bias,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        B, N, S, D = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, D, S)
        x = self.transfer(x)
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x = x.reshape(B, N, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels,
        blocks=1,
        res_expansion=1,
        bias=True,
    ):
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    channels,
                    res_expansion=res_expansion,
                    bias=bias,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointMLP(nn.Module):
    def __init__(
        self,
        points=1024,
        class_num=40,
        embed_dim=64,
        res_expansion=1.0,
        bias=False,
        normalize="anchor",
        dim_expansion=(2, 2, 2, 2),
        pre_blocks=(2, 2, 2, 2),
        pos_blocks=(2, 2, 2, 2),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
        **kwargs,
    ):
        super().__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(
            3,
            embed_dim,
            bias=bias,
        )
        assert (
            len(pre_blocks)
            == len(k_neighbors)
            == len(reducers)
            == len(pos_blocks)
            == len(dim_expansion)
        )
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce_ = reducers[i]
            anchor_points = anchor_points // reduce_
            # append local_grouper_list
            local_grouper = LocalGrouper(
                last_channel, anchor_points, kneighbor, normalize
            )  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(
                last_channel,
                out_channel,
                pre_block_num,
                res_expansion=res_expansion,
                bias=bias,
            )
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(
                out_channel,
                pos_block_num,
                res_expansion=res_expansion,
                bias=bias,
            )
            self.pos_blocks_list.append(pos_block_module)
            last_channel = out_channel

        self.act = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num),
        )

    def forward(self, xyz):
        x = xyz.permute(0, 2, 1)
        x = self.embedding(x)  # B,D,N
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d]
            # return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](
                xyz, x.permute(0, 2, 1)
            )  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x
