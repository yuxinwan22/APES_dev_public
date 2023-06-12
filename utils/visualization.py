import os
import shutil
import numpy as np
import pkbar
import math
from plyfile import PlyData, PlyElement
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm


def visualize_shapenet_predictions(config, samples, preds, seg_labels, cls_label, shape_ious, index):
    base_path = f'./artifacts/{config.wandb.name}/vis_pred'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select predictions
    samples_tmp = []
    preds_tmp = []
    seg_gts_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.K))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.K))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.K))]
    for cat_id in config.test.visualize_preds.vis_which:
        samples_tmp.append(samples[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        preds_tmp.append(preds[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        seg_gts_tmp.append(seg_labels[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_preds.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_preds.num_vis])
    samples = np.concatenate(samples_tmp)
    preds = np.concatenate(preds_tmp)
    seg_labels = np.concatenate(seg_gts_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized prediction files, please wait...', target=len(samples))
    for i, (sample, pred, seg_gt, category, iou) in enumerate(zip(samples, preds, seg_labels, categories, shape_ious)):
        xyzRGB = []
        xyzRGB_gt = []
        xyzRGB_list = []
        xyzRGB_gt_list = []
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for xyz, p, gt in zip(sample, pred, seg_gt):
            xyzRGB_tmp = []
            xyzRGB_gt_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
            xyzRGB.append(tuple(xyzRGB_tmp))
            xyzRGB_gt_tmp.extend(list(xyz))
            xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
            xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
        xyzRGB_list.append(xyzRGB)
        xyzRGB_gt_list.append(xyzRGB_gt)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            xyzRGB = []
            xyzRGB_gt = []
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer + 1):
                    idx = index[layer - j][i, 0][idx]  # index mapping
            else:
                idx = index[layer][i, 0]
            for xyz, p, gt in zip(sample[idx], pred[idx], seg_gt[idx]):
                xyzRGB_tmp = []
                xyzRGB_gt_tmp = []
                xyzRGB_tmp.extend(list(xyz))
                xyzRGB_tmp.extend(config.datasets.cmap[str(p)])
                xyzRGB.append(tuple(xyzRGB_tmp))
                xyzRGB_gt_tmp.extend(list(xyz))
                xyzRGB_gt_tmp.extend(config.datasets.cmap[str(gt)])
                xyzRGB_gt.append(tuple(xyzRGB_gt_tmp))
            xyzRGB_list.append(xyzRGB)
            xyzRGB_gt_list.append(xyzRGB_gt)
        if config.test.visualize_preds.format == 'ply':
            for which_layer, (xyzRGB, xyzRGB_gt) in enumerate(zip(xyzRGB_list, xyzRGB_gt_list)):
                if which_layer > 0:
                    pred_saved_path = f'{cat_path}/{category}{i}_pred_{math.floor(iou * 1e5)}_dsLayer{which_layer}.ply'
                    gt_saved_path = f'{cat_path}/{category}{i}_gt_dsLayer{which_layer}.ply'
                else:
                    pred_saved_path = f'{cat_path}/{category}{i}_pred_{math.floor(iou * 1e5)}.ply'
                    gt_saved_path = f'{cat_path}/{category}{i}_gt.ply'
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(pred_saved_path)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(gt_saved_path)
        elif config.test.visualize_preds.format == 'png':
            for which_layer, (xyzRGB, xyzRGB_gt) in enumerate(zip(xyzRGB_list, xyzRGB_gt_list)):
                if which_layer > 0:
                    pred_saved_path = f'{cat_path}/{category}{i}_pred_{math.floor(iou * 1e5)}_dsLayer{which_layer}.png'
                    gt_saved_path = f'{cat_path}/{category}{i}_gt_dsLayer{which_layer}.png'
                else:
                    pred_saved_path = f'{cat_path}/{category}{i}_pred_{math.floor(iou * 1e5)}.png'
                    gt_saved_path = f'{cat_path}/{category}{i}_gt.png'
                vertex = np.array(xyzRGB)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(pred_saved_path, bbox_inches='tight')
                plt.close(fig)
                vertex = np.array(xyzRGB_gt)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:] / 255, marker='o', s=1)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(gt_saved_path, bbox_inches='tight')
                plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_preds.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_shapenet_downsampled_points(config, samples, index, cls_label, shape_ious):
    base_path = f'./artifacts/{config.wandb.name}/vis_ds_points'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    category_id_to_hash_code_mapping = {}
    for hash_code in list(config.datasets.mapping.keys()):
        category_id_to_hash_code_mapping[str(config.datasets.mapping[hash_code]['category_id'])] = hash_code
    categories = []
    for cat_id in cls_label:
        hash_code = category_id_to_hash_code_mapping[str(cat_id)]
        categories.append(config.datasets.mapping[hash_code]['category'])
    # select samples
    samples_tmp = []
    categories_tmp = []
    ious_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.K))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.K))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.K))]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_tmp.append(samples[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        ious_tmp.append(np.asarray(shape_ious)[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_label == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    shape_ious = np.concatenate(ious_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    for i, (sample, category, iou) in enumerate(zip(samples, categories, shape_ious)):
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer+1):
                    idx = index[layer-j][i, 0][idx]   # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i}_layer{layer}_{math.floor(iou * 1e5)}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i}_layer{layer}_{math.floor(iou * 1e5)}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=2)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:] / 255, marker='o', s=8)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_downsampled_points(config, samples, index, cls_labels):
    base_path = f'./artifacts/{config.wandb.name}/vis_ds_points'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        idx_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.K))]
    if config.neighbor2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.K))]
    if config.point2point_block.enable:
        idx_tmp = [[] for _ in range(len(config.point2point_block.downsample.K))]
    for cat_id in config.test.visualize_downsampled_points.vis_which:
        samples_tmp.append(samples[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
        for layer, idx in enumerate(index):
            idx_tmp[layer].append(idx[cls_labels == cat_id][:config.test.visualize_downsampled_points.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    index = []
    for each in idx_tmp:
        index.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized downsampled points files, please wait...', target=len(samples))
    for i, (sample, category) in enumerate(zip(samples, categories)):
        xyzRGB = []
        for xyz in sample:
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            xyzRGB_tmp.extend([192, 192, 192])  # gray color
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        for layer, idx in enumerate(index):  # idx.shape == (B, H, K)
            if layer != 0:
                idx = idx[i, 0]  # only visualize the first head
                for j in range(1, layer+1):
                    idx = index[layer-j][i, 0][idx]   # index mapping
            else:
                idx = index[layer][i, 0]
            xyzRGB_tmp = deepcopy(xyzRGB)
            ds_tmp = []
            rst_tmp = []
            for each_idx in idx:
                xyzRGB_tmp[each_idx][3:] = [255, 0, 0]  # red color
                ds_tmp.append(xyzRGB_tmp[each_idx])
            for each in xyzRGB_tmp:
                if each not in ds_tmp:
                    rst_tmp.append(each)
            if config.test.visualize_downsampled_points.format == 'ply':
                saved_path = f'{cat_path}/{category}{i}_layer{layer}.ply'
                vertex = PlyElement.describe(np.array(xyzRGB_tmp, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(saved_path)
            elif config.test.visualize_downsampled_points.format == 'png':
                saved_path = f'{cat_path}/{category}{i}_layer{layer}.png'
                rst_vertex = np.array(rst_tmp)
                ds_vertex = np.array(ds_tmp)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.set_xlim3d(-0.6, 0.6)
                ax.set_ylim3d(-0.6, 0.6)
                ax.set_zlim3d(-0.6, 0.6)
                ax.scatter(rst_vertex[:, 0], rst_vertex[:, 2], rst_vertex[:, 1], c=rst_vertex[:, 3:] / 255, marker='o', s=2)
                ax.scatter(ds_vertex[:, 0], ds_vertex[:, 2], ds_vertex[:, 1], c=ds_vertex[:, 3:]/255, marker='o', s=8)
                plt.axis('off')
                plt.grid('off')
                plt.savefig(saved_path, bbox_inches='tight')
                plt.close(fig)
            else:
                raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')


def visualize_modelnet_heatmap(config, samples, attention_map, cls_labels):
    # this function only generates heatmap for the first downsample layer
    my_cmap = cm.get_cmap('viridis_r', samples.shape[1])
    base_path = f'./artifacts/{config.wandb.name}/vis_heatmap'
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    # get category name
    categories = []
    for cat_id in cls_labels:
        categories.append(config.datasets.mapping[int(cat_id)])
    # select samples
    samples_tmp = []
    categories_tmp = []
    if config.edgeconv_block.enable:
        attention_tmp = [[] for _ in range(len(config.edgeconv_with_ds_block.downsample.K))]
    if config.neighbor2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.neighbor2point_block.downsample.K))]
    if config.point2point_block.enable:
        attention_tmp = [[] for _ in range(len(config.point2point_block.downsample.K))]
    for cat_id in config.test.visualize_attention_heatmap.vis_which:
        samples_tmp.append(samples[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        categories_tmp.append(np.asarray(categories)[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
        for layer, atten in enumerate(attention_map):
            attention_tmp[layer].append(atten[cls_labels == cat_id][:config.test.visualize_attention_heatmap.num_vis])
    samples = np.concatenate(samples_tmp)
    categories = np.concatenate(categories_tmp)
    attention_map = []
    for each in attention_tmp:
        attention_map.append(np.concatenate(each))
    # start visualization
    pbar = pkbar.Pbar(name='Generating visualized heatmap files, please wait...', target=len(samples))
    for i, (sample, category, atten) in enumerate(zip(samples, categories, attention_map[0])):
        xyzRGB = []
        atten = atten[0]
        atten = (atten - np.mean(atten)) / np.std(atten) + 0.5
        for xyz, rgb in zip(sample, atten):
            xyzRGB_tmp = []
            xyzRGB_tmp.extend(list(xyz))
            RGB = 255 * np.asarray(my_cmap(rgb))[:3]
            xyzRGB_tmp.extend(list(RGB))
            xyzRGB.append(xyzRGB_tmp)
        cat_path = f'{base_path}/{category}'
        if not os.path.exists(cat_path):
            os.makedirs(cat_path)
        if config.test.visualize_attention_heatmap.format == 'ply':
            saved_path = f'{cat_path}/{category}{i}.ply'
            vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
            PlyData([vertex]).write(saved_path)
        elif config.test.visualize_attention_heatmap.format == 'png':
            saved_path = f'{cat_path}/{category}{i}.png'
            vertex = np.array(xyzRGB)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlim3d(-0.6, 0.6)
            ax.set_ylim3d(-0.6, 0.6)
            ax.set_zlim3d(-0.6, 0.6)
            ax.scatter(vertex[:, 0], vertex[:, 2], vertex[:, 1], c=vertex[:, 3:]/255, marker='o', s=1)
            plt.axis('off')
            plt.grid('off')
            plt.savefig(saved_path, bbox_inches='tight')
            plt.close(fig)
        else:
            raise ValueError(f'format should be png or ply, but got {config.test.visualize_downsampled_points.format}')
        pbar.update(i)
    print(f'Done! All files are saved in {base_path}')
