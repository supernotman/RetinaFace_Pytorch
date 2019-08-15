import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from dataloader import TrainDataset, ValDataset, collater, RandomCroper, RandomFlip, Resizer, PadToSquare
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable, DoubleTable, SingleTable
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import torch.distributed as dist
import eval_widerface
import torchvision
import model
import os
from torch.utils.data.distributed import DistributedSampler
import torchvision_model


def get_args():
    parser = argparse.ArgumentParser(description="Train program for retinaface.")
    parser.add_argument('--data_path', type=str, help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Max training epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle dataset or not')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size')
    parser.add_argument('--verbose', type=int, default=10, help='Log verbose')
    parser.add_argument('--save_step', type=int, default=10, help='Save every save_step epochs')
    parser.add_argument('--eval_step', type=int, default=3, help='Evaluate every eval_step epochs')
    parser.add_argument('--save_path', type=str, default='./out', help='Model save path')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    args = parser.parse_args()
    print(args)

    return args


def main():
    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_path = os.path.join(args.save_path,'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    writer = SummaryWriter(log_dir=log_path)

    data_path = args.data_path
    train_path = os.path.join(data_path,'train/label.txt')
    val_path = os.path.join(data_path,'val/label.txt')
    # dataset_train = TrainDataset(train_path,transform=transforms.Compose([RandomCroper(),RandomFlip()]))
    dataset_train = TrainDataset(train_path,transform=transforms.Compose([Resizer(),PadToSquare()]))
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=args.batch, collate_fn=collater,shuffle=True)
    # dataset_val = ValDataset(val_path,transform=transforms.Compose([RandomCroper()]))
    dataset_val = ValDataset(val_path,transform=transforms.Compose([Resizer(),PadToSquare()]))
    dataloader_val = DataLoader(dataset_val, num_workers=8, batch_size=args.batch, collate_fn=collater)
    
    total_batch = len(dataloader_train)

	# Create the model
    # if args.depth == 18:
    #     retinaface = model.resnet18(num_classes=2, pretrained=True)
    # elif args.depth == 34:
    #     retinaface = model.resnet34(num_classes=2, pretrained=True)
    # elif args.depth == 50:
    #     retinaface = model.resnet50(num_classes=2, pretrained=True)
    # elif args.depth == 101:
    #     retinaface = model.resnet101(num_classes=2, pretrained=True)
    # elif args.depth == 152:
    #     retinaface = model.resnet152(num_classes=2, pretrained=True)
    # else:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Create torchvision model
    return_layers = {'layer2':1,'layer3':2,'layer4':3}
    retinaface = torchvision_model.create_retinaface(return_layers)

    retinaface = retinaface.cuda()
    retinaface = torch.nn.DataParallel(retinaface).cuda()
    retinaface.training = True

    optimizer = optim.Adam(retinaface.parameters(), lr=1e-3)
    # optimizer = optim.SGD(retinaface.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,60], gamma=0.1)

    print('Start to train.')

    epoch_loss = []
    iteration = 0

    for epoch in range(args.epochs):
        retinaface.train()
        #print('Current learning rate:',scheduler.get_lr()[0])
        # retinaface.module.freeze_bn()
        # retinaface.module.freeze_first_layer()

        # Training
        for iter_num,data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, bbox_regression_loss,ldm_regression_loss = retinaface([data['img'].cuda().float(), data['annot']])
            classification_loss = classification_loss.mean()
            bbox_regression_loss = bbox_regression_loss.mean()
            ldm_regression_loss = ldm_regression_loss.mean()

            # loss = classification_loss + 1.0 * bbox_regression_loss + 0.5 * ldm_regression_loss
            loss = classification_loss + bbox_regression_loss + ldm_regression_loss

            loss.backward()
            optimizer.step()
            #epoch_loss.append(loss.item())
            
            if iter_num % args.verbose == 0:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, args.epochs, iter_num, total_batch)
                table_data = [
                    ['loss name','value'],
                    ['total_loss',str(loss.item())],
                    ['classification',str(classification_loss.item())],
                    ['bbox',str(bbox_regression_loss.item())],
                    ['landmarks',str(ldm_regression_loss.item())]
                    ]
                table = AsciiTable(table_data)
                #table = SingleTable(table_data)
                #table = DoubleTable(table_data)
                log_str +=table.table
                print(log_str)
                # write the log to tensorboard
                writer.add_scalar('losses:',loss.item(),iteration*args.verbose)
                writer.add_scalar('class losses:',classification_loss.item(),iteration*args.verbose)
                writer.add_scalar('box losses:',bbox_regression_loss.item(),iteration*args.verbose)
                writer.add_scalar('landmark losses:',ldm_regression_loss.item(),iteration*args.verbose)
                iteration +=1
        
        #scheduler.step()
        #scheduler.step(np.mean(epoch_loss))	

        # Eval
        if epoch % args.eval_step == 0:
            print('-------- RetinaFace Pytorch --------')
            print ('Evaluating epoch {}'.format(epoch))
            recall, precision = eval_widerface.evaluate(dataloader_val,retinaface)
            print('Recall:',recall)
            print('Precision:',precision)

            writer.add_scalar('Recall:', recall, epoch*args.eval_step)
            writer.add_scalar('Precision:', precision, epoch*args.eval_step)

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(retinaface.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))

    writer.close()


if __name__=='__main__':
    main()