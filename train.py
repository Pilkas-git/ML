import os

import torch
import pandas as pd
from tensorboardX import SummaryWriter

from torchvision import datasets
from torchvision.transforms import transforms
from config.train_config import cfg
from utils.evaluate_utils import evaluate
from utils.im_utils import Compose, ToTensor, Resize, RandomHorizontalFlip
from utils.plot_utils import plot_loss_and_lr, plot_map
from utils.train_utils import train_one_epoch, write_tb, create_model
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataloader.OpenImagesDatasetCC import OpenImagesDataset

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', torch.cuda.get_device_name(0))

    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    # tensorboard writer
    writer = SummaryWriter(os.path.join(cfg.model_save_dir, 'epoch_log'))

    if not os.path.exists(cfg.train_root_dir):
        raise FileNotFoundError("dataset root dir not exist!")

    data_transform = {
       "train": Compose([ToTensor()]),
       "valid": Compose([ToTensor()])
    }

    print('Loading train dataset')
    train_data_set = OpenImagesDataset(cfg.valid_root_dir, transform=data_transform["train"], 
                 dataset_type="validation")
    batch_size = cfg.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    shuffle=True,
                                                    collate_fn=train_data_set.collate_fn)
    print('train dataset loaded')                                                

    print('Loading val dataset')
    val_data_set = OpenImagesDataset(cfg.valid_root_dir,
                 dataset_type="validation", transform=data_transform["valid"])
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                        batch_size=1,
                                                        num_workers=1,
                                                      shuffle=False,
                                                      collate_fn=val_data_set.collate_fn)
    print('val dataset loaded')

    model = create_model(num_classes=cfg.num_class)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay) 

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.lr_dec_step_size,
                                                   gamma=cfg.lr_gamma)

    # train from pretrained weights
    if cfg.resume != "":
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(cfg.start_epoch))

    train_loss = []
    learning_rate = []
    validation_loss = []
    train_acc = []
    validation_acc = []
    evaluateAfter = 0
    valmAp = 0
    valAp = 0
    bestMap = 0
    print("Epoch start")
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        loss_dict, total_loss = train_one_epoch(model, optimizer, train_data_loader,
                                                device, epoch, train_loss=train_loss, train_lr=learning_rate,
                                                print_freq=105, warmup=False)

        lr_scheduler.step()

        if evaluateAfter == cfg.eval_after:
            evaluateAfter = 0
            valAp, valmAp = evaluate(model, val_data_set_loader,device=device)
            print('valid APs {}'.format(valAp))
            print('valid mAP is {}'.format(valmAp))
        
        evaluateAfter += 1
        board_info = {'lr': optimizer.param_groups[0]['lr'],
                      'losses_reduced': total_loss,
                      'val_mAP': valmAp
                      }

        for k, v in loss_dict.items():
            board_info[k] = v.item()
        board_info['total loss'] = total_loss.item()
        write_tb(writer, epoch, board_info)

        if valmAp > bestMap:
            bestMap = valmAp
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            model_save_dir = cfg.model_save_dir
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(save_files,
                       os.path.join(model_save_dir, "{}-modelBASELINE-{}-mAp-{}.pth".format("res50fasterrcnn", epoch, valmAp)))
    writer.close()
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, cfg.model_save_dir)

if __name__ == "__main__":
    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))
    main()