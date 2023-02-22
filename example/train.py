import os, sys
import pathlib
from argparse import ArgumentParser
import torch
from loguru import logger
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def load_args_from_config(args):
    config_file = args.config_file
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
            for k,v in d.items():
                setattr(args, k, v)
    else:
        print('Config file does not exist.')
    return args

def save_all_hparams(trainer, args):
    """
    save ckpt
    TODO
    """
    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir)
    save_dict = args.__dict__
    save_dict.pop('checkpoint_callback')
    with open(trainer.logger.log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)
    
    
def build_args():
    """
    build ArgumentParser
    Returns:
        ArgumentParser: return ArgumentParser.parse_args()
    """
    parser = ArgumentParser()

    # arguments
    parser.add_argument(
        '--config_file', 
        default=None,   
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--data_path', 
        default=None,   
        type=pathlib.Path,          
        help='If given, experiment configuration will be loaded from this yaml file.',
    )
    parser.add_argument(
        '--experiment_name', 
        default='ct-vit-test',   
        type=str,          
        help='Define the project name.',
    )
    parser.add_argument(
        '--gpus', 
        default=2,   
        type=int,          
        help='num_gpus',
    )
    parser.add_argument("--local_rank", default=-1)

    args = parser.parse_args()
    
    # Load args if config file is given
    if args.config_file is not None:
        args = load_args_from_config(args)
        
    return args


def run():
    args = build_args()
    logger.debug("Build args success!")
    torch.torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.debug("Set seed: " + str(args.seed))
    # ------------
    # dataloader
    # ------------
    
    
    # ------------
    # model
    # ------------
    # model = HUMUSNetModule(
    #     num_cascades=args.num_cascades,
    #     sens_pools=args.sens_pools,
    #     sens_chans=args.sens_chans,
    #     img_size=args.uniform_train_resolution,
    #     patch_size=args.patch_size,
    #     window_size=args.window_size,
    #     embed_dim=args.embed_dim, 
    #     depths=args.depths,
    #     num_heads=args.num_heads,
    #     mlp_ratio=args.mlp_ratio, 
    #     bottleneck_depth=args.bottleneck_depth,
    #     bottleneck_heads=args.bottleneck_heads,
    #     resi_connection=args.resi_connection,
    #     conv_downsample_first=args.conv_downsample_first,
    #     num_adj_slices=args.num_adj_slices,
    #     mask_center=(not args.no_center_masking),
    #     use_checkpoint=args.use_checkpointing,
    #     lr=args.lr,
    #     lr_step_size=args.lr_step_size,
    #     lr_gamma=args.lr_gamma,
    #     weight_decay=args.weight_decay,
    #     no_residual_learning=args.no_residual_learning
    # )
    
    
    
    # # ptl data module - this handles data loaders
    # data_module = FastMriDataModule(
    #     data_path=args.data_path,
    #     challenge=args.challenge,
    #     train_transform=train_transform,
    #     val_transform=val_transform,
    #     test_transform=test_transform,
    #     test_split=args.test_split,
    #     test_path=args.test_path,
    #     sample_rate=args.sample_rate,
    #     volume_sample_rate=args.volume_sample_rate,
    #     use_dataset_cache_file=args.use_dataset_cache_file,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    #     combine_train_val=args.combine_train_val,
    #     train_scanners=args.train_scanners,
    #     val_scanners=args.val_scanners,
    #     combined_scanner_val=args.combined_scanner_val,
    #     num_adj_slices=args.num_adj_slices,
    # )

    
    # # Save all hyperparameters to .yaml file in the current log dir
    # if torch.distributed.is_available():
    #     if torch.distributed.is_initialized():
    #         if torch.distributed.get_rank() == 0:
    #             save_all_hparams(trainer, args)
    # else: 
    #      save_all_hparams(trainer, args)
            
            
    # local_rank = args.local_rank
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')
    
    # device = torch.device("cuda", local_rank)
    # model = model.to(device)
    # if args.resume_from_checkpoint:
    #     model = torch.load(args.resume_from_checkpoint)
        
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # my_trainset = data_module
    # train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)

    # for epoch in range(args.max_epochs):
    #     trainloader.sampler.set_epoch(epoch)
    #     for data, label in trainloader:
    #         prediction = model(data)
    #         loss = loss_fn(prediction, label)
    #         loss.backward()
    #         optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    #         optimizer.step()
            
    # if dist.get_rank() == 0:
    #     torch.save(model.module, "saved_model.ckpt")        

if __name__ == "__main__":
    run()
    