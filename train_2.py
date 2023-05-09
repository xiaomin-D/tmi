import os, sys
import pathlib
from argparse import ArgumentParser
import torch
from loguru import logger
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.append("../")
from data.dataloader import build_dataloader
from data.dataloader import build_dataloader_h5
from models.unet3D import UNet3D
from torchsummary import summary
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.loss import MSELoss
from torchmetrics import Dice
from utils.loss import SoftDiceLoss
from utils.loss import MixLoss
from utils.metric import DiceScore
def save_all_hparams(trainer, cfg):
    """
    save ckpt
    TODO
    """
    if not os.path.exists(trainer.logger.log_dir):
        os.makedirs(trainer.logger.log_dir)
    save_dict = cfg.__dict__
    save_dict.pop('checkpoint_callback')
    with open(trainer.logger.log_dir + '/hparams.yaml', 'w') as f:
        yaml.dump(save_dict, f)


def build_cfg():
    """
    build ArgumentParser
    Returns:
        ArgumentParser: return ArgumentParser.parse_cfg()
    """
    parser = ArgumentParser()

    # arguments
    parser.add_argument(
        '--config_file',
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
    parser.add_argument(
        '--seed', 
        default=3047,   
        type=int,          
        help='seed',
    )

    cfg = parser.parse_args()
    
    # Load cfg if config file is given
    if cfg.config_file:
        config_file = cfg.config_file
        if config_file.exists():
            with config_file.open('r') as f:
                d = yaml.safe_load(f)
                for k,v in d.items():
                    setattr(cfg, k, v)
        else:
            print('Config file does not exist.')
        
    return cfg


def run():
    cfg = build_cfg()
    logger.debug("Build cfg success!")
    logger.debug(cfg)
    torch.torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    logger.debug("Set seed: " + str(cfg.seed))
    # ------------
    # dataloader
    # ------------
    #  /nfs/public/dxm/Task300_ischemic
    # /home/dxm/dxm/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task100_mycardium
    save_dir = "/home/dxm/dxm/Datasets/Myocardial+Ischemic_h5py/Myocardial+Ischemic.h5"
    ct_dataloader = build_dataloader_h5(save_dir, batch_size = 2)
    # for img, label in ct_dataloader:
    #     logger.debug("img shape is " + str(img.shape))
    #     logger.debug("label shape is: " +  str(label.shape))
        
    # ------------
    # model
    # ------------
    
    net = UNet3D(in_channels=1, out_channels=1, init_features=2).cuda()
    summary(net, input_size = (1,512,512,512), batch_size = 4)
    optimizer = torch.optim.Adam(net.parameters(),lr=cfg.lr)
    criterion_ce = CrossEntropyLoss().cuda()
    mse = MSELoss().cuda()
    dice = Dice(average='micro').cuda()
    softdice = SoftDiceLoss().cuda()
    mixloss = MixLoss().cuda()
    dicescore = DiceScore().cuda()
    if cfg.resume_from_checkpoint:
        net.load_state_dict(torch.load('ckpt/epoch_40_.pt'))
        logger.debug("load ckpt from: ckpt/epoch_40_.pt")
        
    for e in range(cfg.max_epochs):
        for img, label1, label2 in  ct_dataloader:
            # breakpoint()
            img = img.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()
            optimizer.step()
            # logger.debug(img)
            # logger.debug(label)
            outputs, ffn1 = net(img)
            logger.debug(ffn1)
            # breakpoint()
            loss = mixloss(outputs, label1, ffn1, label2)
            metric = dicescore(outputs, label1)
            logger.debug("loss: " + str(loss.item()))
            logger.debug("dice score: " + str(metric.item()))
            loss.backward()
            optimizer.step()
        from datetime import datetime
        timestamp_str = datetime.now().strftime('%Y-%m-%d')
        
        if not e % cfg.save_ckpt_every_n_epoch:
            if os.path.exists(os.path.join(cfg.ckpt_path, timestamp_str)):
                torch.save(net.state_dict(), os.path.join(cfg.ckpt_path, timestamp_str,"epoch_" + str(e) + "_.pt"))
            else:
                os.mkdir(os.path.join(cfg.ckpt_path, timestamp_str))
                torch.save(net.state_dict(), os.path.join(cfg.ckpt_path, timestamp_str,"epoch_" + str(e) + "_.pt"))


    
    # # Save all hyperparameters to .yaml file in the current log dir
    # if torch.distributed.is_available():
    #     if torch.distributed.is_initialized():
    #         if torch.distributed.get_rank() == 0:
    #             save_all_hparams(trainer, cfg)
    # else: 
    #      save_all_hparams(trainer, cfg)
            
            
    # local_rank = cfg.local_rank
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')
    
    # device = torch.device("cuda", local_rank)
    # model = model.to(device)
    # if cfg.resume_from_checkpoint:
    #     model = torch.load(cfg.resume_from_checkpoint)
        
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # my_trainset = data_module
    # train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)

    # if dist.get_rank() == 0:
    #     torch.save(model.module, "saved_model.ckpt")        

if __name__ == "__main__":
    run()
    