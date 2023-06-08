import ssl
import shutil
import wget
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
import h5py
import glob
from utils import data_augmentation
from pytorch3d.ops import sample_farthest_points as fps
import pickle


# ================================================================================
# Yi650M shapenet dataloader

def download_shapenet_Yi650M(url, saved_path):
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # check if dataset already exists
    path = Path(saved_path, 'shapenetcore_partanno_segmentation_benchmark_v0')
    if not path.exists():
        print('Downloading dataset, please wait...')
        wget.download(url=url, out=saved_path)
        print()
        file = str(Path(saved_path, url.split('/')[-1]).resolve())
        print('Unpacking dataset, please wait...')
        shutil.unpack_archive(file, saved_path)
        os.remove(file)


class ShapeNet_Yi650M(torch.utils.data.Dataset):
    def __init__(self, root, json_path, mapping, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
        self.root = root
        self.mapping = mapping
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append([data_augmentation.rotate, [which_axis, angle_range]])
            if translate:
                self.augmentation_list.append([data_augmentation.translate, [x_translate_range, y_translate_range, z_translate_range]])
            if anisotropic_scale:
                self.augmentation_list.append([data_augmentation.anisotropic_scale, [x_scale_range, y_scale_range, z_scale_range]])
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError('At least one kind of data augmentation should be applied!')
            if len(self.augmentation_list) < num_aug:
                raise ValueError(f'num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}')
        self.samples = []
        for each_path in json_path:
            with open(each_path, 'r') as f:
                self.samples.extend(json.load(f))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        category_hash, pcd_hash = sample.split('/')[1:]

        # get point cloud
        pcd_path = os.path.join(self.root, category_hash, 'points', f'{pcd_hash}.pts')
        pcd = np.loadtxt(pcd_path)
        # get a fixed number of points from every point cloud
        if self.fps_enable:
            if self.selected_points <= len(pcd):
                pcd = torch.Tensor(pcd[None, ...]).cuda()  # fps requires batch size dimension
                pcd, indices = fps(pcd, K=self.selected_points, random_start_point=True)
                pcd, indices = pcd[0].cpu().numpy(), indices[0].cpu().numpy()  # squeeze the batch size dimension
            else:
                indices = np.random.choice(len(pcd), self.selected_points, replace=True)
                pcd = pcd[indices]
        else:
            if self.selected_points <= len(pcd):
                indices = np.random.choice(len(pcd), self.selected_points, replace=False)
                pcd = pcd[indices]
            else:
                indices = np.random.choice(len(pcd), self.selected_points, replace=True)
                pcd = pcd[indices]
        if self.augmentation:
            choice = np.random.choice(len(self.augmentation_list), self.num_aug, replace=False)
            for i in choice:
                augmentation, params = self.augmentation_list[i]
                pcd = augmentation(pcd, *params)
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # get point cloud seg label
        parts_id = self.mapping[category_hash]['parts_id']
        seg_label_path = os.path.join(self.root, category_hash, 'points_label', f'{pcd_hash}.seg')
        seg_label = np.loadtxt(seg_label_path).astype('float32')
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        diff = min(parts_id) - 1
        seg_label = seg_label + diff
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # get category one hot
        category_id = self.mapping[category_hash]['category_id']
        category_onehot = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32).permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset_Yi650M(saved_path, mapping, selected_points, fps_enable, augmentation, num_aug, jitter,
                                std, clip, rotate, which_axis, angle_range, translate, x_translate_range,
                                y_translate_range, z_translate_range, anisotropic_scale, x_scale_range, y_scale_range,
                                z_scale_range):
    dataset_path = Path(saved_path, 'shapenetcore_partanno_segmentation_benchmark_v0')
    # get datasets json files
    train_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_train_file_list.json')
    validation_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_val_file_list.json')
    test_json = os.path.join(dataset_path, 'train_test_split', 'shuffled_test_file_list.json')

    # get datasets
    train_set = ShapeNet_Yi650M(dataset_path, [train_json], mapping, selected_points, fps_enable, augmentation, num_aug, jitter,
                                std, clip, rotate, which_axis, angle_range, translate, x_translate_range,
                                y_translate_range, z_translate_range, anisotropic_scale, x_scale_range, y_scale_range,
                                z_scale_range)
    validation_set = ShapeNet_Yi650M(dataset_path, [validation_json], mapping, selected_points, fps_enable, False, num_aug, jitter,
                                     std, clip, rotate, which_axis, angle_range, translate, x_translate_range,
                                     y_translate_range, z_translate_range, anisotropic_scale, x_scale_range, y_scale_range,
                                     z_scale_range)
    trainval_set = ShapeNet_Yi650M(dataset_path, [train_json, validation_json], mapping, selected_points, fps_enable, augmentation, num_aug, jitter,
                                   std, clip, rotate, which_axis, angle_range, translate, x_translate_range,
                                   y_translate_range, z_translate_range, anisotropic_scale, x_scale_range, y_scale_range,
                                   z_scale_range)
    test_set = ShapeNet_Yi650M(dataset_path, [test_json], mapping, selected_points, fps_enable, False, num_aug, jitter,
                               std, clip, rotate, which_axis, angle_range, translate, x_translate_range,
                               y_translate_range, z_translate_range, anisotropic_scale, x_scale_range, y_scale_range,
                               z_scale_range)

    return train_set, validation_set, trainval_set, test_set


# ================================================================================
# AnTao350M shapenet dataloader

def download_shapenet_AnTao350M(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if not os.path.exists(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')):
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


class ShapeNet_AnTao350M(torch.utils.data.Dataset):
    def __init__(self, saved_path, partition, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append([data_augmentation.rotate, [which_axis, angle_range]])
            if translate:
                self.augmentation_list.append([data_augmentation.translate, [x_translate_range, y_translate_range, z_translate_range]])
            if anisotropic_scale:
                self.augmentation_list.append([data_augmentation.anisotropic_scale, [x_scale_range, y_scale_range, z_scale_range]])
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError('At least one kind of data augmentation should be applied!')
            if len(self.augmentation_list) < num_aug:
                raise ValueError(f'num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}')
        self.all_pcd = []
        self.all_cls_label = []
        self.all_seg_label = []
        if partition == 'trainval':
            file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
                   + glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
        else:
            file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*%s*.h5' % partition))
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            pcd = f['data'][:].astype('float32')
            cls_label = f['label'][:].astype('int64')
            seg_label = f['pid'][:].astype('int64')
            f.close()
            self.all_pcd.append(pcd)
            self.all_cls_label.append(cls_label)
            self.all_seg_label.append(seg_label)
        self.all_pcd = np.concatenate(self.all_pcd, axis=0)
        self.all_cls_label = np.concatenate(self.all_cls_label, axis=0)
        self.all_seg_label = np.concatenate(self.all_seg_label, axis=0)

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index, 0]
        category_onehot = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32).permute(1, 0)

        # get point cloud
        pcd = self.all_pcd[index]
        if self.fps_enable:
            pcd = torch.Tensor(pcd[None, ...]).cuda()  # fps requires batch size dimension
            pcd, indices = fps(pcd, K=self.selected_points, random_start_point=True)
            pcd, indices = pcd[0].cpu().numpy(), indices[0].cpu().numpy()  # squeeze the batch size dimension
        else:
            # shuffle points within one point cloud
            indices = np.random.choice(2048, self.selected_points, False)
            pcd = pcd[indices]
        if self.augmentation:
            choice = np.random.choice(len(self.augmentation_list), self.num_aug, replace=False)
            for i in choice:
                augmentation, params = self.augmentation_list[i]
                pcd = augmentation(pcd, *params)
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # get point cloud seg label
        seg_label = self.all_seg_label[index].astype('float32')
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset_AnTao350M(saved_path, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                   angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                   x_scale_range, y_scale_range, z_scale_range):
    # get dataset
    train_set = ShapeNet_AnTao350M(saved_path, 'train', selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                   angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                   x_scale_range, y_scale_range, z_scale_range)
    validation_set = ShapeNet_AnTao350M(saved_path, 'val', selected_points, fps_enable, False, num_aug, jitter, std, clip, rotate, which_axis,
                                        angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                        x_scale_range, y_scale_range, z_scale_range)
    trainval_set = ShapeNet_AnTao350M(saved_path, 'trainval', selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                      angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                      x_scale_range, y_scale_range, z_scale_range)
    test_set = ShapeNet_AnTao350M(saved_path, 'test', selected_points, fps_enable, False, num_aug, jitter, std, clip, rotate, which_axis,
                                  angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                  x_scale_range, y_scale_range, z_scale_range)
    return train_set, validation_set, trainval_set, test_set


# ================================================================================
# AnTao420M modelnet dataloader

def download_modelnet_AnTao420M(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if not os.path.exists(os.path.join(saved_path, 'modelnet40_ply_hdf5_2048')):
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', os.path.join(saved_path, 'modelnet40_ply_hdf5_2048')))
        os.system('rm %s' % (zipfile))


class ModelNet_AnTao420M(torch.utils.data.Dataset):
    def __init__(self, saved_path, partition, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append([data_augmentation.rotate, [which_axis, angle_range]])
            if translate:
                self.augmentation_list.append([data_augmentation.translate, [x_translate_range, y_translate_range, z_translate_range]])
            if anisotropic_scale:
                self.augmentation_list.append([data_augmentation.anisotropic_scale, [x_scale_range, y_scale_range, z_scale_range]])
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError('At least one kind of data augmentation should be applied!')
            if len(self.augmentation_list) < num_aug:
                raise ValueError(f'num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}')
        self.all_pcd = []
        self.all_cls_label = []
        if partition == 'trainval':
            file = glob.glob(os.path.join(saved_path, 'modelnet40_ply_hdf5_2048', '*train*.h5'))
        elif partition == 'test':
            file = glob.glob(os.path.join(saved_path, 'modelnet40_ply_hdf5_2048', '*test*.h5'))
        else:
            raise ValueError('modelnet40 has only train_set and test_set, which means validation_set is included in train_set!')
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            pcd = f['data'][:].astype('float32')
            cls_label = f['label'][:].astype('int64')
            f.close()
            self.all_pcd.append(pcd)
            self.all_cls_label.append(cls_label[:, 0])
        self.all_pcd = np.concatenate(self.all_pcd, axis=0)
        self.all_cls_label = np.concatenate(self.all_cls_label, axis=0)

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index]
        category_onehot = F.one_hot(torch.Tensor([category_id]).long(), 40).to(torch.float32).squeeze()

        # get point cloud
        pcd = self.all_pcd[index]
        if self.fps_enable:
            pcd = torch.Tensor(pcd[None, ...]).cuda()  # fps requires batch size dimension
            pcd, _ = fps(pcd, K=self.selected_points, random_start_point=True)
            pcd = pcd[0].cpu().numpy()  # squeeze the batch size dimension
        else:
            indices = np.random.choice(2048, self.selected_points, False)
            pcd = pcd[indices]
        if self.augmentation:
            choice = np.random.choice(len(self.augmentation_list), self.num_aug, replace=False)
            for i in choice:
                augmentation, params = self.augmentation_list[i]
                pcd = augmentation(pcd, *params)
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # pcd.shape == (C, N)  category_onehot.shape == (40,)
        return pcd, category_onehot


def get_modelnet_dataset_AnTao420M(saved_path, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                   angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                   x_scale_range, y_scale_range, z_scale_range):
    # get dataset
    trainval_set = ModelNet_AnTao420M(saved_path, 'trainval', selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                      angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                      x_scale_range, y_scale_range, z_scale_range)
    test_set = ModelNet_AnTao420M(saved_path, 'test', selected_points, fps_enable, False, num_aug, jitter, std, clip, rotate, which_axis,
                                  angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                  x_scale_range, y_scale_range, z_scale_range)
    return trainval_set, test_set


# ================================================================================
# Alignment1024 modelnet dataloader

def download_modelnet_Alignment1024(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    dataset_path = os.path.join(saved_path, 'modelnet40_normal_resampled')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        os.system('gdown "https://drive.google.com/uc?id=1fq4G5djBblr6FME7TY5WH7Lnz9psVf4i"')
        os.system('gdown "https://drive.google.com/uc?id=1WzcIm2G55yTh-snOrdeiZJrYDBqJeAck"')
        os.system('mv %s %s' % ('modelnet40_train_1024pts_fps.dat', dataset_path))
        os.system('mv %s %s' % ('modelnet40_test_1024pts_fps.dat', dataset_path))


class ModelNet_Alignment1024(torch.utils.data.Dataset):
    def __init__(self, saved_path, partition, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
        super(ModelNet_Alignment1024, self).__init__()
        self.selected_points = selected_points
        self.fps_enable = fps_enable
        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append([data_augmentation.rotate, [which_axis, angle_range]])
            if translate:
                self.augmentation_list.append([data_augmentation.translate, [x_translate_range, y_translate_range, z_translate_range]])
            if anisotropic_scale:
                self.augmentation_list.append([data_augmentation.anisotropic_scale, [x_scale_range, y_scale_range, z_scale_range]])
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError('At least one kind of data augmentation should be applied!')
            if len(self.augmentation_list) < num_aug:
                raise ValueError(f'num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}')
        if partition == 'trainval':
            data_path = os.path.join(saved_path, 'modelnet40_normal_resampled', 'modelnet40_train_1024pts_fps.dat')
        elif partition == 'test':
            data_path = os.path.join(saved_path, 'modelnet40_normal_resampled', 'modelnet40_test_1024pts_fps.dat')
        else:
            raise ValueError('modelnet40 has only train_set and test_set, which means validation_set is included in train_set!')
        with open(data_path, 'rb') as f:
            self.all_pcd, self.all_cls_label = pickle.load(f)
        self.all_pcd = np.stack(self.all_pcd, axis=0)[:, :, :3]
        self.all_cls_label = np.stack(self.all_cls_label, axis=0)[:, 0]

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index]
        category_onehot = F.one_hot(torch.Tensor([category_id]).long(), 40).to(torch.float32).squeeze()

        # get point cloud
        pcd = self.all_pcd[index]
        if self.fps_enable:
            pcd = torch.Tensor(pcd[None, ...]).cuda()  # fps requires batch size dimension
            pcd, _ = fps(pcd, K=self.selected_points, random_start_point=True)
            pcd = pcd[0].cpu().numpy()  # squeeze the batch size dimension
        else:
            indices = np.random.choice(1024, self.selected_points, False)
            pcd = pcd[indices]
        if self.augmentation:
            choice = np.random.choice(len(self.augmentation_list), self.num_aug, replace=False)
            for i in choice:
                augmentation, params = self.augmentation_list[i]
                pcd = augmentation(pcd, *params)
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # pcd.shape == (C, N)  category_onehot.shape == (40,)
        return pcd, category_onehot


def get_modelnet_dataset_Alignment1024(saved_path, selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                       angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                       x_scale_range, y_scale_range, z_scale_range):
    # get dataset
    trainval_set = ModelNet_Alignment1024(saved_path, 'trainval', selected_points, fps_enable, augmentation, num_aug, jitter, std, clip, rotate, which_axis,
                                          angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                          x_scale_range, y_scale_range, z_scale_range)
    test_set = ModelNet_Alignment1024(saved_path, 'test', selected_points, fps_enable, False, num_aug, jitter, std, clip, rotate, which_axis,
                                      angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                                      x_scale_range, y_scale_range, z_scale_range)
    return trainval_set, test_set
