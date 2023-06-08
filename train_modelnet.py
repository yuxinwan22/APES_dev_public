import shutil
from utils import dataloader, lr_scheduler
from models import modelnet_model
from omegaconf import OmegaConf
import hydra
from pathlib import Path
import torch
import pkbar
import wandb
from utils import metrics, debug
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda import amp
import numpy as np
from utils.loss import consistency_loss


@hydra.main(version_base=None, config_path="./configs", config_name="default.yaml")
def main(config):

    # check working directory
    try:
        assert str(Path.cwd().resolve()) == str(Path(__file__).resolve().parents[0])
    except:
        exit(f'Working directory is not the same as project root. Exit.')

    # overwrite the default config with user config
    if config.usr_config:
        usr_config = OmegaConf.load(config.usr_config)
        config = OmegaConf.merge(config, usr_config)

    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        dataloader.download_modelnet_AnTao420M(config.datasets.url, config.datasets.saved_path)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        dataloader.download_modelnet_Alignment1024(config.datasets.url, config.datasets.saved_path)
    else:
        raise ValueError('Not implemented!')

    # multiprocessing for ddp
    if torch.cuda.is_available():
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # read .h5 file using multiprocessing will raise error
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.train.ddp.which_gpu).replace(' ', '').replace('[', '').replace(']', '')
        mp.spawn(train, args=(config,), nprocs=config.train.ddp.nproc_this_node, join=True)
    else:
        exit('It is almost impossible to train this model using CPU. Please use GPU! Exit.')


def train(local_rank, config):  # the first arg must be local rank for the sake of using mp.spawn(...)

    rank = config.train.ddp.rank_starts_from + local_rank

    if config.wandb.enable and rank == 0:
        # initialize wandb
        wandb.login(key=config.wandb.api_key)
        del config.wandb.api_key, config.test
        config_dict = OmegaConf.to_container(config, resolve=True)
        run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config_dict, name=config.wandb.name)
        # cache source code for saving
        OmegaConf.save(config=config, f=f'/tmp/{run.id}_usr_config.yaml', resolve=False)
        os.system(f'cp ./models/modelnet_model.py /tmp/{run.id}_modelnet_model.py')
        os.system(f'cp ./utils/dataloader.py /tmp/{run.id}_dataloader.py')
        os.system(f'cp ./utils/metrics.py /tmp/{run.id}_metrics.py')
        os.system(f'cp ./utils/ops.py /tmp/{run.id}_ops.py')
        os.system(f'cp ./utils/data_augmentation.py /tmp/{run.id}_data_augmentation.py')
        os.system(f'cp ./utils/debug.py /tmp/{run.id}_debug.py')
        os.system(f'cp ./utils/visualization.py /tmp/{run.id}_visualization.py')
        os.system(f'cp ./utils/lr_scheduler.py /tmp/{run.id}_lr_scheduler.py')
        os.system(f'cp ./train_modelnet.py /tmp/{run.id}_train_modelnet.py')
        os.system(f'cp ./test_modelnet.py /tmp/{run.id}_test_modelnet.py')

    # process initialization
    os.environ['MASTER_ADDR'] = str(config.train.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(config.train.ddp.master_port)
    os.environ['WORLD_SIZE'] = str(config.train.ddp.world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # gpu setting
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)  # which gpu is used by current process
    print(f'[init] pid: {os.getpid()} - global rank: {rank} - local rank: {local_rank} - cuda: {config.train.ddp.which_gpu[local_rank]}')

    # create a scaler for amp
    scaler = amp.GradScaler()

    # get dataset
    if config.datasets.dataset_name == 'modelnet_AnTao420M':
        trainval_set, test_set = dataloader.get_modelnet_dataset_AnTao420M(config.datasets.saved_path, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug, config.train.dataloader.data_augmentation.jitter.enable,
                                                                           config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                           config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                           config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                           config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range)
    elif config.datasets.dataset_name == 'modelnet_Alignment1024':
        trainval_set, test_set = dataloader.get_modelnet_dataset_Alignment1024(config.datasets.saved_path, config.train.dataloader.selected_points, config.train.dataloader.fps, config.train.dataloader.data_augmentation.enable, config.train.dataloader.data_augmentation.num_aug, config.train.dataloader.data_augmentation.jitter.enable,
                                                                               config.train.dataloader.data_augmentation.jitter.std, config.train.dataloader.data_augmentation.jitter.clip, config.train.dataloader.data_augmentation.rotate.enable, config.train.dataloader.data_augmentation.rotate.which_axis,
                                                                               config.train.dataloader.data_augmentation.rotate.angle_range, config.train.dataloader.data_augmentation.translate.enable, config.train.dataloader.data_augmentation.translate.x_range,
                                                                               config.train.dataloader.data_augmentation.translate.y_range, config.train.dataloader.data_augmentation.translate.z_range, config.train.dataloader.data_augmentation.anisotropic_scale.enable,
                                                                               config.train.dataloader.data_augmentation.anisotropic_scale.x_range, config.train.dataloader.data_augmentation.anisotropic_scale.y_range, config.train.dataloader.data_augmentation.anisotropic_scale.z_range)
    else:
        raise ValueError('Not implemented!')

    # get sampler
    trainval_sampler = torch.utils.data.distributed.DistributedSampler(trainval_set)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)

    # get dataloader
    trainval_loader = torch.utils.data.DataLoader(trainval_set, config.train.dataloader.batch_size_per_gpu, num_workers=config.train.dataloader.num_workers, drop_last=True, prefetch_factor=config.train.dataloader.prefetch, pin_memory=config.train.dataloader.pin_memory, sampler=trainval_sampler)
    test_loader = torch.utils.data.DataLoader(test_set, config.train.dataloader.batch_size_per_gpu, num_workers=config.train.dataloader.num_workers, drop_last=True, prefetch_factor=config.train.dataloader.prefetch, pin_memory=config.train.dataloader.pin_memory, sampler=test_sampler)

    # if combine train and validation
    if config.train.dataloader.combine_trainval:
        train_sampler = trainval_sampler
        train_loader = trainval_loader
        validation_loader = test_loader
    else:
        raise ValueError('modelnet40 has only train_set and test_set, which means validation_set is included in train_set!')

    # get model
    my_model = modelnet_model.ModelNetModel(config.neighbor2point_block.enable,
                                            config.neighbor2point_block.edgeconv_embedding.K,
                                            config.neighbor2point_block.edgeconv_embedding.group_type,
                                            config.neighbor2point_block.edgeconv_embedding.conv1_in,
                                            config.neighbor2point_block.edgeconv_embedding.conv1_out,
                                            config.neighbor2point_block.edgeconv_embedding.conv2_in,
                                            config.neighbor2point_block.edgeconv_embedding.conv2_out,
                                            config.neighbor2point_block.downsample.which_ds,
                                            config.neighbor2point_block.downsample.K,
                                            config.neighbor2point_block.downsample.q_in,
                                            config.neighbor2point_block.downsample.q_out,
                                            config.neighbor2point_block.downsample.k_in,
                                            config.neighbor2point_block.downsample.k_out,
                                            config.neighbor2point_block.downsample.v_in,
                                            config.neighbor2point_block.downsample.v_out,
                                            config.neighbor2point_block.downsample.num_heads,
                                            config.neighbor2point_block.neighbor2point.K,
                                            config.neighbor2point_block.neighbor2point.group_type,
                                            config.neighbor2point_block.neighbor2point.q_in,
                                            config.neighbor2point_block.neighbor2point.q_out,
                                            config.neighbor2point_block.neighbor2point.k_in,
                                            config.neighbor2point_block.neighbor2point.k_out,
                                            config.neighbor2point_block.neighbor2point.v_in,
                                            config.neighbor2point_block.neighbor2point.v_out,
                                            config.neighbor2point_block.neighbor2point.num_heads,
                                            config.neighbor2point_block.neighbor2point.ff_conv1_channels_in,
                                            config.neighbor2point_block.neighbor2point.ff_conv1_channels_out,
                                            config.neighbor2point_block.neighbor2point.ff_conv2_channels_in,
                                            config.neighbor2point_block.neighbor2point.ff_conv2_channels_out,
                                            config.point2point_block.enable,
                                            config.point2point_block.edgeconv_embedding.K,
                                            config.point2point_block.edgeconv_embedding.group_type,
                                            config.point2point_block.edgeconv_embedding.conv1_in,
                                            config.point2point_block.edgeconv_embedding.conv1_out,
                                            config.point2point_block.edgeconv_embedding.conv2_in,
                                            config.point2point_block.edgeconv_embedding.conv2_out,
                                            config.point2point_block.downsample.which_ds,
                                            config.point2point_block.downsample.K,
                                            config.point2point_block.downsample.q_in,
                                            config.point2point_block.downsample.q_out,
                                            config.point2point_block.downsample.k_in,
                                            config.point2point_block.downsample.k_out,
                                            config.point2point_block.downsample.v_in,
                                            config.point2point_block.downsample.v_out,
                                            config.point2point_block.downsample.num_heads,
                                            config.point2point_block.point2point.q_in,
                                            config.point2point_block.point2point.q_out,
                                            config.point2point_block.point2point.k_in,
                                            config.point2point_block.point2point.k_out,
                                            config.point2point_block.point2point.v_in,
                                            config.point2point_block.point2point.v_out,
                                            config.point2point_block.point2point.num_heads,
                                            config.point2point_block.point2point.ff_conv1_channels_in,
                                            config.point2point_block.point2point.ff_conv1_channels_out,
                                            config.point2point_block.point2point.ff_conv2_channels_in,
                                            config.point2point_block.point2point.ff_conv2_channels_out,
                                            config.edgeconv_block.enable,
                                            config.edgeconv_block.edgeconv_embedding.K,
                                            config.edgeconv_block.edgeconv_embedding.group_type,
                                            config.edgeconv_block.edgeconv_embedding.conv1_in,
                                            config.edgeconv_block.edgeconv_embedding.conv1_out,
                                            config.edgeconv_block.edgeconv_embedding.conv2_in,
                                            config.edgeconv_block.edgeconv_embedding.conv2_out,
                                            config.edgeconv_block.downsample.which_ds,
                                            config.edgeconv_block.downsample.K,
                                            config.edgeconv_block.downsample.q_in,
                                            config.edgeconv_block.downsample.q_out,
                                            config.edgeconv_block.downsample.k_in,
                                            config.edgeconv_block.downsample.k_out,
                                            config.edgeconv_block.downsample.v_in,
                                            config.edgeconv_block.downsample.v_out,
                                            config.edgeconv_block.downsample.num_heads,
                                            config.edgeconv_block.edgeconv.K,
                                            config.edgeconv_block.edgeconv.group_type,
                                            config.edgeconv_block.edgeconv.conv1_in,
                                            config.edgeconv_block.edgeconv.conv1_out,
                                            config.edgeconv_block.edgeconv.conv2_in,
                                            config.edgeconv_block.edgeconv.conv2_out)

    # synchronize bn among gpus
    if config.train.ddp.syn_bn:  #TODO: test performance
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    # get ddp model
    my_model = my_model.to(device)
    my_model = torch.nn.parallel.DistributedDataParallel(my_model)

    # add fp hook and bp hook
    if config.train.debug.enable:
        if len(config.train.ddp.which_gpu) > 1:
            raise ValueError('Please only use 1 GPU when using debug mode!')
        else:
            all_layers = debug.get_layers(my_model)
            debug_log_saved_dir = f'./debug/{config.wandb.name}'
            if os.path.exists(debug_log_saved_dir):
                shutil.rmtree(debug_log_saved_dir)
            os.makedirs(debug_log_saved_dir)
            os.system(f'touch {debug_log_saved_dir}/model_structure.txt')
            with open(f'{debug_log_saved_dir}/model_structure.txt', 'a') as f:
                f.write(my_model.__str__())
            if config.train.debug.check_layer_input_range:
                check_layer_input_range_saved_dir = f'{debug_log_saved_dir}/check_layer_input_range.txt'
                os.system(f'touch {check_layer_input_range_saved_dir}')
                for layer in all_layers:
                    layer.register_forward_hook(debug.check_layer_input_range_fp_hook)
            if config.train.debug.check_layer_output_range:
                check_layer_output_range_saved_dir = f'{debug_log_saved_dir}/check_layer_output_range.txt'
                os.system(f'touch {check_layer_output_range_saved_dir}')
                for layer in all_layers:
                    layer.register_forward_hook(debug.check_layer_output_range_fp_hook)
            if config.train.debug.check_layer_parameter_range:
                check_layer_parameter_range_saved_dir = f'{debug_log_saved_dir}/check_layer_parameter_range.txt'
                os.system(f'touch {check_layer_parameter_range_saved_dir}')
                for layer in all_layers:
                    layer.register_forward_hook(debug.check_layer_parameter_range_fp_hook)
            if config.train.debug.check_gradient_input_range:
                check_gradient_input_range_saved_dir = f'{debug_log_saved_dir}/check_gradient_input_range.txt'
                os.system(f'touch {check_gradient_input_range_saved_dir}')
                for layer in all_layers:
                    layer.register_full_backward_hook(debug.check_gradient_input_range_bp_hook)
            if config.train.debug.check_gradient_output_range:
                check_gradient_output_range_saved_dir = f'{debug_log_saved_dir}/check_gradient_output_range.txt'
                os.system(f'touch {check_gradient_output_range_saved_dir}')
                for layer in all_layers:
                    layer.register_full_backward_hook(debug.check_gradient_output_range_bp_hook)
            if config.train.debug.check_gradient_parameter_range:
                check_gradient_parameter_range_saved_dir = f'{debug_log_saved_dir}/check_gradient_parameter_range.txt'
                os.system(f'touch {check_gradient_parameter_range_saved_dir}')
                for layer in all_layers:
                    layer.register_full_backward_hook(debug.check_gradient_parameter_range_bp_hook)

    # get optimizer
    if config.train.optimizer.which == 'adamw':
        optimizer = torch.optim.AdamW(my_model.parameters(), lr=config.train.lr, weight_decay=config.train.optimizer.weight_decay)
    elif config.train.optimizer.which == 'sgd':
        optimizer = torch.optim.SGD(my_model.parameters(), lr=config.train.lr, weight_decay=config.train.optimizer.weight_decay, momentum=0.9)
    else:
        raise ValueError('Not implemented!')

    # get lr scheduler
    if config.train.lr_scheduler.enable:
        if config.train.lr_scheduler.which == 'stepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_scheduler.stepLR.decay_step, gamma=config.train.lr_scheduler.stepLR.gamma)
        elif config.train.lr_scheduler.which == 'expLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.train.lr_scheduler.expLR.gamma)
        elif config.train.lr_scheduler.which == 'cosLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.lr_scheduler.cosLR.T_max, eta_min=config.train.lr_scheduler.cosLR.eta_min)
        elif config.train.lr_scheduler.which == 'cos_warmupLR':
            scheduler = lr_scheduler.CosineAnnealingWithWarmupLR(optimizer, T_max=config.train.lr_scheduler.cos_warmupLR.T_max, eta_min=config.train.lr_scheduler.cos_warmupLR.eta_min, warmup_init_lr=config.train.lr_scheduler.cos_warmupLR.warmup_init_lr, warmup_epochs=config.train.lr_scheduler.cos_warmupLR.warmup_epochs)
        else:
            raise ValueError('Not implemented!')

    # get loss function
    if config.train.label_smoothing:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.train.epsilon)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    val_acc_list = [0]
    # start training
    for epoch in range(config.train.epochs):
        my_model.train()
        train_sampler.set_epoch(epoch)
        train_loss_list = []
        pred_list = []
        cls_label_list = []
        if rank == 0:
            kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=config.train.epochs, always_stateful=True)
        for i, (samples, cls_labels) in enumerate(train_loader):
            optimizer.zero_grad()
            samples, cls_labels = samples.to(device), cls_labels.to(device)
            if config.train.amp:
                with amp.autocast():
                    preds = my_model(samples)
                    train_loss = loss_fn(preds, cls_labels) + config.train.consistency_loss_factor * consistency_loss(my_model.module.block.res_link_list)
                scaler.scale(train_loss).backward()
                # log debug information
                if config.train.debug.enable:
                    if config.train.debug.check_layer_input_range:
                        debug.log_debug_message(check_layer_input_range_saved_dir, all_layers, 'check_layer_input_range_msg', epoch, i)
                    if config.train.debug.check_layer_output_range:
                        debug.log_debug_message(check_layer_output_range_saved_dir, all_layers, 'check_layer_output_range_msg', epoch, i)
                    if config.train.debug.check_layer_parameter_range:
                        debug.log_debug_message(check_layer_parameter_range_saved_dir, all_layers, 'check_layer_parameter_range_msg', epoch, i)
                    if config.train.debug.check_gradient_input_range:
                        debug.log_debug_message(check_gradient_input_range_saved_dir, all_layers, 'check_gradient_input_range_msg', epoch, i)
                    if config.train.debug.check_gradient_output_range:
                        debug.log_debug_message(check_gradient_output_range_saved_dir, all_layers, 'check_gradient_output_range_msg', epoch, i)
                    if config.train.debug.check_gradient_parameter_range:
                        debug.log_debug_message(check_gradient_parameter_range_saved_dir, all_layers, 'check_gradient_parameter_range_msg', epoch, i)
                if config.train.grad_clip.enable:
                    scaler.unscale_(optimizer)
                    if config.train.grad_clip.mode == 'value':
                        torch.nn.utils.clip_grad_value_(my_model.parameters(), config.train.grad_clip.value)
                    elif config.train.grad_clip.mode == 'norm':
                        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config.train.grad_clip.max_norm)
                    else:
                        raise ValueError('mode should be value or norm!')
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = my_model(samples)
                train_loss = loss_fn(preds, cls_labels)
                train_loss.backward()
                # log debug information
                if config.train.debug.enable:
                    if config.train.debug.check_layer_input_range:
                        debug.log_debug_message(check_layer_input_range_saved_dir, all_layers, 'check_layer_input_range_msg', epoch, i)
                    if config.train.debug.check_layer_output_range:
                        debug.log_debug_message(check_layer_output_range_saved_dir, all_layers, 'check_layer_output_range_msg', epoch, i)
                    if config.train.debug.check_layer_parameter_range:
                        debug.log_debug_message(check_layer_parameter_range_saved_dir, all_layers, 'check_layer_parameter_range_msg', epoch, i)
                    if config.train.debug.check_gradient_input_range:
                        debug.log_debug_message(check_gradient_input_range_saved_dir, all_layers, 'check_gradient_input_range_msg', epoch, i)
                    if config.train.debug.check_gradient_output_range:
                        debug.log_debug_message(check_gradient_output_range_saved_dir, all_layers, 'check_gradient_output_range_msg', epoch, i)
                    if config.train.debug.check_gradient_parameter_range:
                        debug.log_debug_message(check_gradient_parameter_range_saved_dir, all_layers, 'check_gradient_parameter_range_msg', epoch, i)
                if config.train.grad_clip.enable:
                    if config.train.grad_clip.mode == 'value':
                        torch.nn.utils.clip_grad_value_(my_model.parameters(), config.train.grad_clip.value)
                    elif config.train.grad_clip.mode == 'norm':
                        torch.nn.utils.clip_grad_norm_(my_model.parameters(), config.train.grad_clip.max_norm)
                    else:
                        raise ValueError('mode should be value or norm!')
                optimizer.step()

            # collect the result from all gpus
            pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.train.ddp.nproc_this_node)]
            cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in range(config.train.ddp.nproc_this_node)]
            torch.distributed.all_gather(pred_gather_list, preds)
            torch.distributed.all_gather(cls_label_gather_list, cls_labels)
            torch.distributed.all_reduce(train_loss)
            if rank == 0:
                preds = torch.concat(pred_gather_list, dim=0)
                pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                cls_labels = torch.concat(cls_label_gather_list, dim=0)
                cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
                train_loss /= config.train.ddp.nproc_this_node
                train_loss_list.append(train_loss.detach().cpu().numpy())
                kbar.update(i)

        # decay lr
        current_lr = optimizer.param_groups[0]['lr']
        if config.train.lr_scheduler.enable:
            if config.train.lr_scheduler.which == 'cosLR' and epoch + 1 > config.train.lr_scheduler.cosLR.T_max:
                pass
            else:
                scheduler.step()

        # calculate metrics
        if rank == 0:
            preds = np.concatenate(pred_list, axis=0)
            cls_labels = np.concatenate(cls_label_list, axis=0)
            train_acc = metrics.calculate_accuracy(preds, cls_labels)
            train_loss = sum(train_loss_list) / len(train_loss_list)

        # log results
        if rank == 0:
            if config.wandb.enable:
                metric_dict = {'modelnet_train': {'lr': current_lr, 'loss': train_loss, 'acc': train_acc}}
                if (epoch+1) % config.train.validation_freq:
                    wandb.log(metric_dict, commit=True)
                else:
                    wandb.log(metric_dict, commit=False)

        # start validation
        if not (epoch+1) % config.train.validation_freq:
            my_model.eval()
            val_loss_list = []
            pred_list = []
            cls_label_list = []
            with torch.no_grad():
                for samples, cls_labels in validation_loader:
                    samples, cls_labels = samples.to(device), cls_labels.to(device)
                    preds = my_model(samples)
                    val_loss = loss_fn(preds, cls_labels)

                    # collect the result among all gpus
                    pred_gather_list = [torch.empty_like(preds).to(device) for _ in range(config.train.ddp.nproc_this_node)]
                    cls_label_gather_list = [torch.empty_like(cls_labels).to(device) for _ in range(config.train.ddp.nproc_this_node)]
                    torch.distributed.all_gather(pred_gather_list, preds)
                    torch.distributed.all_gather(cls_label_gather_list, cls_labels)
                    torch.distributed.all_reduce(val_loss)
                    if rank == 0:
                        preds = torch.concat(pred_gather_list, dim=0)
                        pred_list.append(torch.max(preds, dim=1)[1].detach().cpu().numpy())
                        cls_labels = torch.concat(cls_label_gather_list, dim=0)
                        cls_label_list.append(torch.max(cls_labels, dim=1)[1].detach().cpu().numpy())
                        val_loss /= config.train.ddp.nproc_this_node
                        val_loss_list.append(val_loss.detach().cpu().numpy())

            # calculate metrics
            if rank == 0:
                preds = np.concatenate(pred_list, axis=0)
                cls_labels = np.concatenate(cls_label_list, axis=0)
                val_acc = metrics.calculate_accuracy(preds, cls_labels)
                val_loss = sum(val_loss_list) / len(val_loss_list)

            # log results
            if rank == 0:
                kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_acc', train_acc), ('val_loss', val_loss), ('val_acc', val_acc)])
                if config.wandb.enable:
                    # save model
                    if val_acc >= max(val_acc_list):
                        state_dict = my_model.state_dict()
                        torch.save(state_dict, f'/tmp/{run.id}_checkpoint.pt')
                    val_acc_list.append(val_acc)
                    metric_dict = {'modelnet_val': {'loss': val_loss, 'acc': val_acc}}
                    metric_dict['modelnet_val']['best_acc'] = max(val_acc_list)
                    wandb.log(metric_dict, commit=True)
        else:
            if rank == 0:
                kbar.update(i+1, values=[('lr', current_lr), ('train_loss', train_loss), ('train_acc', train_acc)])

    # save artifacts to wandb server
    if config.wandb.enable and rank == 0:
        artifacts = wandb.Artifact(config.wandb.name, type='runs')
        artifacts.add_file(f'/tmp/{run.id}_usr_config.yaml', name='usr_config.yaml')
        artifacts.add_file(f'/tmp/{run.id}_modelnet_model.py', name='modelnet_model.py')
        artifacts.add_file(f'/tmp/{run.id}_dataloader.py', name='dataloader.py')
        artifacts.add_file(f'/tmp/{run.id}_metrics.py', name='metrics.py')
        artifacts.add_file(f'/tmp/{run.id}_ops.py', name='ops.py')
        artifacts.add_file(f'/tmp/{run.id}_data_augmentation.py', name='data_augmentation.py')
        artifacts.add_file(f'/tmp/{run.id}_debug.py', name='debug.py')
        artifacts.add_file(f'/tmp/{run.id}_visualization.py', name='visualization.py')
        artifacts.add_file(f'/tmp/{run.id}_lr_scheduler.py', name='lr_scheduler.py')
        artifacts.add_file(f'/tmp/{run.id}_train_modelnet.py', name='train_modelnet.py')
        artifacts.add_file(f'/tmp/{run.id}_test_modelnet.py', name='test_modelnet.py')
        artifacts.add_file(f'/tmp/{run.id}_checkpoint.pt', name='checkpoint.pt')
        run.log_artifact(artifacts)
        wandb.finish(quiet=True)


if __name__ == '__main__':
    main()
