import torch
from torch import nn
from utils import ops
import math
import torch.nn.functional as F
from utils.pointnet_utils import PointNetEncoder


class EdgeConvBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64, egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_k=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 K=(32, 32, 32), group_type=('center_diff', 'center_diff', 'center_diff'),
                 conv1_channel_in=(3 * 2, 64 * 2, 64 * 2), conv1_channel_out=(64, 64, 64),
                 conv2_channel_in=(64, 64, 64), conv2_channel_out=(64, 64, 64)):
        super(EdgeConvBlock, self).__init__()
        self.embedding_list = nn.ModuleList([EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList([DownSample(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList([DownSampleWithSigma(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for which_ds!')
        self.edgeconv_list = nn.ModuleList([EdgeConv(k, g_type, conv1_in, conv1_out, conv2_in, conv2_out) for k, g_type, conv1_in, conv1_out, conv2_in, conv2_out in zip(K, group_type, conv1_channel_in, conv1_channel_out, conv2_channel_in, conv2_channel_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.edgeconv_list[0](x)
        for i in range(len(self.downsample_list)):
            x = self.downsample_list[i](x)[0][0]
            x = self.edgeconv_list[i+1](x)
        return x


class EdgeConv(nn.Module):
    def __init__(self, K=32, group_type='center_diff', conv1_channel_in=6, conv1_channel_out=64, conv2_channel_in=64, conv2_channel_out=64):

        super(EdgeConv, self).__init__()
        self.K = K
        self.group_type = group_type

        self.conv1 = nn.Sequential(nn.Conv2d(conv1_channel_in, conv1_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv1_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(conv2_channel_in, conv2_channel_out, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(conv2_channel_out),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x.shape == (B, C, N)
        x = ops.group(x, self.K, self.group_type)
        # x.shape == (B, 2C, N, K) or (B, C, N, K)
        x = self.conv1(x)
        # x.shape == (B, C, N, K)
        x = self.conv2(x)
        # x.shape == (B, C, N, K)
        x = x.max(dim=-1, keepdim=False)[0]
        # x.shape == (B, C, N)
        return x


class Neighbor2PointAttentionBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64, egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_k=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 K=(32, 32, 32), group_type=('diff', 'diff', 'diff'), q_in=(64, 64, 64), q_out=(64, 64, 64),
                 k_in=(64, 64, 64), k_out=(64, 64, 64), v_in=(64, 64, 64), v_out=(64, 64, 64), num_heads=(8, 8, 8),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64)):
        super(Neighbor2PointAttentionBlock, self).__init__()
        self.embedding_list = nn.ModuleList([EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList([DownSample(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList([DownSampleWithSigma(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for which_ds!')
        self.neighbor2point_list = nn.ModuleList([Neighbor2PointAttention(k, g_type, q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out)
                                                  for k, g_type, q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                                  in zip(K, group_type, q_in, q_out, k_in, k_out, v_in, v_out, num_heads, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.neighbor2point_list[0](x)
        for i in range(len(self.downsample_list)):
            x = self.downsample_list[i](x)[0][0]
            x = self.neighbor2point_list[i+1](x)
        return x


class Neighbor2PointAttention(nn.Module):
    def __init__(self, K=32, group_type='diff', q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128, ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(Neighbor2PointAttention, self).__init__()
        # check input values
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.K = K
        self.group_type = group_type
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        neighbors = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        x_tmp = x[:, :, :, None]
        # x_tmp.shape == (B, C, N, 1)
        q = self.q_conv(x_tmp)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        x_tmp = (attention @ v)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # x_tmp.shape == (B, N, H, D)
        x_tmp = x_tmp.reshape(x_tmp.shape[0], x_tmp.shape[1], -1).permute(0, 2, 1)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


class Point2PointAttentionBlock(nn.Module):
    def __init__(self, egdeconv_emb_K=40, egdeconv_emb_group_type='center_diff',
                 egdeconv_emb_conv1_in=6, egdeconv_emb_conv1_out=64, egdeconv_emb_conv2_in=64, egdeconv_emb_conv2_out=64,
                 downsample_which='p2p', downsample_k=(1024, 512), downsample_q_in=(64, 64), downsample_q_out=(64, 64),
                 downsample_k_in=(64, 64), downsample_k_out=(64, 64), downsample_v_in=(64, 64),
                 downsample_v_out=(64, 64), downsample_num_heads=(1, 1),
                 q_in=(64, 64, 64), q_out=(64, 64, 64), k_in=(64, 64, 64), k_out=(64, 64, 64), v_in=(64, 64, 64), v_out=(64, 64, 64), num_heads=(8, 8, 8),
                 ff_conv1_channels_in=(64, 64, 64), ff_conv1_channels_out=(128, 128, 128),
                 ff_conv2_channels_in=(128, 128, 128), ff_conv2_channels_out=(64, 64, 64)):
        super(Point2PointAttentionBlock, self).__init__()
        self.embedding_list = nn.ModuleList([EdgeConv(emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out) for emb_k, emb_g_type, emb_conv1_in, emb_conv1_out, emb_conv2_in, emb_conv2_out in zip(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in, egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out)])
        if downsample_which == 'global':
            self.downsample_list = nn.ModuleList([DownSample(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        elif downsample_which == 'local':
            self.downsample_list = nn.ModuleList([DownSampleWithSigma(ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads) for ds_k, ds_q_in, ds_q_out, ds_k_in, ds_k_out, ds_v_in, ds_v_out, ds_heads in zip(downsample_k, downsample_q_in, downsample_q_out, downsample_k_in, downsample_k_out, downsample_v_in, downsample_v_out, downsample_num_heads)])
        else:
            raise ValueError('Only global and local are valid for which_ds!')
        self.point2point_list = nn.ModuleList([Point2PointAttention(q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out)
                                 for q_input, q_output, k_input, k_output, v_input, v_output, heads, ff_conv1_channel_in, ff_conv1_channel_out, ff_conv2_channel_in, ff_conv2_channel_out
                                 in zip(q_in, q_out, k_in, k_out, v_in, v_out, num_heads, ff_conv1_channels_in, ff_conv1_channels_out, ff_conv2_channels_in, ff_conv2_channels_out)])

    def forward(self, x):
        x_list = []
        for embedding in self.embedding_list:
            x = embedding(x)
            x_list.append(x)
        x = torch.cat(x_list, dim=1)
        x = self.point2point_list[0](x)
        for i in range(len(self.downsample_list)):
            x = self.downsample_list[i](x)[0][0]
            x = self.point2point_list[i+1](x)
        return x


class Point2PointAttention(nn.Module):
    def __init__(self, q_in=64, q_out=64, k_in=64, k_out=64, v_in=64, v_out=64, num_heads=8,
                 ff_conv1_channels_in=64, ff_conv1_channels_out=128,
                 ff_conv2_channels_in=128, ff_conv2_channels_out=64):
        super(Point2PointAttention, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')
        if q_in != v_out:
            raise ValueError(f'q_in should be equal to v_out due to ResLink! Got q_in: {q_in}, v_out: {v_out}')

        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.ff = nn.Sequential(nn.Conv1d(ff_conv1_channels_in, ff_conv1_channels_out, 1, bias=False),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(ff_conv2_channels_in, ff_conv2_channels_out, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(v_out)
        self.bn2 = nn.BatchNorm1d(v_out)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        x_tmp = (attention @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # x_tmp.shape == (B, N, H, D)
        x_tmp = x_tmp.reshape(x_tmp.shape[0], x_tmp.shape[1], -1).permute(0, 2, 1)
        # x_tmp.shape == (B, C, N)
        x = self.bn1(x + x_tmp)
        # x.shape == (B, C, N)
        x_tmp = self.ff(x)
        # x_tmp.shape == (B, C, N)
        x = self.bn2(x + x_tmp)
        # x.shape == (B, C, N)
        return x

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x


class DownSample(nn.Module):
    def __init__(self, k, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(DownSample, self).__init__()
        # check input values
        if q_in != k_in or q_in != v_in or k_in != v_in:
            raise ValueError(f'q_in, k_in and v_in should be the same! Got q_in:{q_in}, k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.k = k
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv1d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv1d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv1d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape == (B, C, N)
        q = self.q_conv(x)
        # q.shape == (B, C, N)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, D, N)
        k = self.k_conv(x)
        # k.shape ==  (B, C, N)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, D, N)
        v = self.v_conv(x)
        # v.shape ==  (B, C, N)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, D, N)
        energy = q.permute(0, 1, 3, 2) @ k
        # energy.shape == (B, H, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, N)
        self.idx = torch.sum(attention, dim=-2).topk(self.k, dim=-1)[1]
        # idx.shape == (B, H, K)
        idx_dropped = torch.sum(attention, dim=-2).topk(attention.shape[-1]-self.k, dim=-1, largest=False)[1]
        # idx_dropped.shape == (B, H, N-K)
        attention_down = torch.gather(attention, dim=2, index=self.idx[..., None].expand(-1, -1, -1, q.shape[-1]))
        # attention_down.shape == (B, H, K, N)
        attention_dropped = torch.gather(attention, dim=2, index=idx_dropped[..., None].expand(-1, -1, -1, q.shape[-1]))
        # attention_dropped.shape == (B, H, N-K, N)
        v_down = (attention_down @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # v_down.shape == (B, K, H, D)
        v_dropped = (attention_dropped @ v.permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
        # v_dropped.shape == (B, N-K, H, D)
        v_down = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # v_down.shape == (B, C, K)
        v_dropped = v_dropped.reshape(v_dropped.shape[0], v_dropped.shape[1], -1).permute(0, 2, 1)
        # v_dropped.shape == (B, C, N-K)
        return (v_down, self.idx), (v_dropped, idx_dropped)

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N)
        x = x.view(x.shape[0], heads, depth, x.shape[2])
        # x.shape == (B, H, D, N)
        return x


class DownSampleWithSigma(nn.Module):
    def __init__(self, k, q_in, q_out, k_in, k_out, v_in, v_out, num_heads):
        super(DownSampleWithSigma, self).__init__()
        # check input values
        if k_in != v_in:
            raise ValueError(f'k_in and v_in should be the same! Got k_in:{k_in}, v_in:{v_in}')
        if q_out != k_out:
            raise ValueError('q_out should be equal to k_out!')
        if q_out % num_heads != 0 or k_out % num_heads != 0 or v_out % num_heads != 0:
            raise ValueError('please set another value for num_heads!')

        self.k = k  # number of downsampled points
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.num_heads = num_heads
        self.q_depth = int(q_out / num_heads)
        self.k_depth = int(k_out / num_heads)
        self.v_depth = int(v_out / num_heads)

        self.q_conv = nn.Conv2d(q_in, q_out, 1, bias=False)
        self.k_conv = nn.Conv2d(k_in, k_out, 1, bias=False)
        self.v_conv = nn.Conv2d(v_in, v_out, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape == (B, C, N)
        neighbors = ops.group(x, self.K, self.group_type)
        # neighbors.shape == (B, C, N, K)
        x = x[:, :, :, None]
        # x.shape == (B, C, N, 1)
        q = self.q_conv(x)
        # q.shape == (B, C, N, 1)
        q = self.split_heads(q, self.num_heads, self.q_depth)
        # q.shape == (B, H, N, 1, D)
        k = self.k_conv(neighbors)
        # k.shape ==  (B, C, N, K)
        k = self.split_heads(k, self.num_heads, self.k_depth)
        # k.shape == (B, H, N, K, D)
        v = self.v_conv(neighbors)
        # v.shape ==  (B, C, N, K)
        v = self.split_heads(v, self.num_heads, self.v_depth)
        # v.shape == (B, H, N, K, D)
        k = k.permute(0, 1, 2, 4, 3)
        # k.shape == (B, H, N, D, K)
        energy = q @ k
        # energy.shape == (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)
        # attention.shape == (B, H, N, 1, K)
        self.idx = torch.std(attention, dim=-1, unbiased=False)[:, :, :, 0].topk(self.k, dim=-1)[1]
        # idx.shape == (B, H, M)
        idx_dropped = torch.std(attention, dim=-1, unbiased=False)[:, :, :, 0].topk(attention.shape[-3]-self.k, dim=-1, largest=False)[1]
        # idx_dropped.shape == (B, H, N-M)
        attention_down = torch.gather(attention, dim=2, index=self.idx[..., None, None].expand(-1, -1, -1, -1, k.shape[-1]))
        # attention_down.shape == (B, H, M, 1, K)
        attention_dropped = torch.gather(attention, dim=2, index=idx_dropped[..., None, None].expand(-1, -1, -1, -1, k.shape[-1]))
        # attention_dropped.shape == (B, H, N-M, 1, K)
        v_down = torch.gather(v, dim=2, index=self.idx[..., None, None].expand(-1, -1, -1, k.shape[-1], k.shape[-2]))
        # v_down.shape == (B, H, M, K, D)
        v_dropped = torch.gather(v, dim=2, index=idx_dropped[..., None, None].expand(-1, -1, -1, k.shape[-1], k.shape[-2]))
        # v_dropped.shape == (B, H, N-M, K, D)
        v_down = (attention_down @ v_down)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # v_down.shape == (B, M, H, D)
        v_dropped = (attention_dropped @ v_dropped)[:, :, :, 0, :].permute(0, 2, 1, 3)
        # v_dropped.shape == (B, N-M, H, D)
        v_down = v_down.reshape(v_down.shape[0], v_down.shape[1], -1).permute(0, 2, 1)
        # v_down.shape == (B, C, M)
        v_dropped = v_dropped.reshape(v_dropped.shape[0], v_dropped.shape[1], -1).permute(0, 2, 1)
        # v_dropped.shape == (B, C, N-M)
        return (v_down, self.idx), (v_dropped, idx_dropped)

    def split_heads(self, x, heads, depth):
        # x.shape == (B, C, N, K)
        x = x.view(x.shape[0], heads, depth, x.shape[2], x.shape[3])
        # x.shape == (B, H, D, N, K)
        x = x.permute(0, 1, 3, 4, 2)
        # x.shape == (B, H, N, K, D)
        return x


class PointNet(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(PointNet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x  # x.shape == (B, 40)


class ComparisonModel(nn.Module):
    def __init__(self, neighbor2point_enable, neighbor2point_egdeconv_emb_K, neighbor2point_egdeconv_emb_group_type, neighbor2point_egdeconv_emb_conv1_in,
                 neighbor2point_egdeconv_emb_conv1_out, neighbor2point_egdeconv_emb_conv2_in, neighbor2point_egdeconv_emb_conv2_out,
                 neighbor2point_down_which, neighbor2point_down_k, neighbor2point_down_q_in, neighbor2point_down_q_out, neighbor2point_down_k_in, neighbor2point_down_k_out, neighbor2point_down_v_in, neighbor2point_down_v_out,
                 neighbor2point_down_num_heads, neighbor2point_K, neighbor2point_group_type, neighbor2point_q_in,
                 neighbor2point_q_out, neighbor2point_k_in, neighbor2point_k_out, neighbor2point_v_in, neighbor2point_v_out, neighbor2point_num_heads,
                 neighbor2point_ff_conv1_in, neighbor2point_ff_conv1_out, neighbor2point_ff_conv2_in, neighbor2point_ff_conv2_out,
                 point2point_enable, point2point_egdeconv_emb_K, point2point_egdeconv_emb_group_type,
                 point2point_egdeconv_emb_conv1_in, point2point_egdeconv_emb_conv1_out, point2point_egdeconv_emb_conv2_in, point2point_egdeconv_emb_conv2_out,
                 point2point_down_which, point2point_down_k, point2point_down_q_in, point2point_down_q_out, point2point_down_k_in,
                 point2point_down_k_out, point2point_down_v_in, point2point_down_v_out, point2point_down_num_heads,
                 point2point_q_in, point2point_q_out, point2point_k_in, point2point_k_out, point2point_v_in,
                 point2point_v_out, point2point_num_heads, point2point_ff_conv1_in, point2point_ff_conv1_out, point2point_ff_conv2_in, point2point_ff_conv2_out,
                 edgeconv_enable, egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in,
                 egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out, edgeconv_downsample_which, edgeconv_downsample_k, edgeconv_downsample_q_in, edgeconv_downsample_q_out,
                 edgeconv_downsample_k_in, edgeconv_downsample_k_out, edgeconv_downsample_v_in,
                 edgeconv_downsample_v_out, edgeconv_downsample_num_heads,
                 edgeconv_K, edgeconv_group_type, edgeconv_conv1_channel_in, edgeconv_conv1_channel_out,
                 edgeconv_conv2_channel_in, edgeconv_conv2_channel_out):

        super(ComparisonModel, self).__init__()

        num_enabled_blocks = neighbor2point_enable + point2point_enable + edgeconv_enable
        if num_enabled_blocks != 1:
            raise ValueError(f'Only one of neighbor2point_block, point2point_block and edgecov_block should be enabled, but got {num_enabled_blocks} block(s) enabled!')
        if neighbor2point_enable:
            self.block = Neighbor2PointAttentionBlock(neighbor2point_egdeconv_emb_K, neighbor2point_egdeconv_emb_group_type, neighbor2point_egdeconv_emb_conv1_in,
                                                      neighbor2point_egdeconv_emb_conv1_out, neighbor2point_egdeconv_emb_conv2_in, neighbor2point_egdeconv_emb_conv2_out,
                                                      neighbor2point_down_which, neighbor2point_down_k, neighbor2point_down_q_in, neighbor2point_down_q_out, neighbor2point_down_k_in, neighbor2point_down_k_out, neighbor2point_down_v_in, neighbor2point_down_v_out,
                                                      neighbor2point_down_num_heads, neighbor2point_K, neighbor2point_group_type,
                                                      neighbor2point_q_in, neighbor2point_q_out, neighbor2point_k_in, neighbor2point_k_out,
                                                      neighbor2point_v_in, neighbor2point_v_out, neighbor2point_num_heads, neighbor2point_ff_conv1_in,
                                                      neighbor2point_ff_conv1_out, neighbor2point_ff_conv2_in, neighbor2point_ff_conv2_out)
            nfeat = neighbor2point_ff_conv2_out[-1]
        if point2point_enable:
            self.block = Point2PointAttentionBlock(point2point_egdeconv_emb_K, point2point_egdeconv_emb_group_type,
                                                   point2point_egdeconv_emb_conv1_in, point2point_egdeconv_emb_conv1_out, point2point_egdeconv_emb_conv2_in, point2point_egdeconv_emb_conv2_out,
                                                   point2point_down_which, point2point_down_k, point2point_down_q_in, point2point_down_q_out, point2point_down_k_in,
                                                   point2point_down_k_out, point2point_down_v_in, point2point_down_v_out, point2point_down_num_heads,
                                                   point2point_q_in, point2point_q_out, point2point_k_in, point2point_k_out, point2point_v_in,
                                                   point2point_v_out, point2point_num_heads, point2point_ff_conv1_in, point2point_ff_conv1_out, point2point_ff_conv2_in, point2point_ff_conv2_out)
            nfeat = point2point_ff_conv2_out[-1]
        if edgeconv_enable:
            self.block = EdgeConvBlock(egdeconv_emb_K, egdeconv_emb_group_type, egdeconv_emb_conv1_in,
                                       egdeconv_emb_conv1_out, egdeconv_emb_conv2_in, egdeconv_emb_conv2_out,
                                       edgeconv_downsample_which, edgeconv_downsample_k, edgeconv_downsample_q_in, edgeconv_downsample_q_out,
                                       edgeconv_downsample_k_in, edgeconv_downsample_k_out, edgeconv_downsample_v_in,
                                       edgeconv_downsample_v_out, edgeconv_downsample_num_heads,
                                       edgeconv_K, edgeconv_group_type, edgeconv_conv1_channel_in, edgeconv_conv1_channel_out,
                                       edgeconv_conv2_channel_in, edgeconv_conv2_channel_out)
            nfeat = edgeconv_conv2_channel_out[-1]
        self.conv = nn.Conv1d(nfeat, 3, 1, bias=False)
        self.target_model = PointNet()
        # self.target_model.load_state_dict(torch.load('./artifacts/pointnet_pretrained_model/best_model.pth')['model_state_dict'])
        # for param in self.target_model.parameters():
        #     param.requires_grad = False
        # self.target_model.eval()

    def forward(self, x):
        # x.shape == (B, 3, N)
        x = self.block(x)
        # x.shape == (B, C, M)
        x = self.conv(x)
        # x.shape == (B, 3, M)
        x = self.target_model(x)
        # x.shape == (B, 40)
        return x
