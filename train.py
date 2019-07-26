import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from dataloader import TrainDataset, ValDataset, collater, Augmenter, RandomCroper
from torch.utils.data import Dataset, DataLoader
from terminaltables import AsciiTable, DoubleTable, SingleTable
from torch.optim import lr_scheduler
import torch.distributed as dist
import eval_widerface
import torchvision
import model
import os


def get_args():
    parser = argparse.ArgumentParser(description="Train progress for retinaface.")
    parser.add_argument('--data_path', type=str, help='Path for dataset,default WIDERFACE')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
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

    data_path = args.data_path
    train_path = os.path.join(data_path,'train/label.txt')
    val_path = os.path.join(data_path,'val/label.txt')
    dataset_train = TrainDataset(train_path,transform=transforms.Compose([RandomCroper()]))
    dataloader_train = DataLoader(dataset_train, num_workers=4, batch_size=args.batch, collate_fn=collater,shuffle=True )
    dataset_val = ValDataset(val_path,transform=transforms.Compose([RandomCroper()]))
    dataloader_val = DataLoader(dataset_val, num_workers=4, batch_size=16, collate_fn=collater)
    
    total_batch = len(dataloader_train)

	# Create the model
    if args.depth == 18:
        retinaface = model.resnet18(num_classes=2, pretrained=True)
    elif args.depth == 34:
        retinaface = model.resnet34(num_classes=2, pretrained=True)
    elif args.depth == 50:
        retinaface = model.resnet50(num_classes=2, pretrained=True)
    elif args.depth == 101:
        retinaface = model.resnet101(num_classes=2, pretrained=True)
    elif args.depth == 152:
        retinaface = model.resnet152(num_classes=2, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinaface = retinaface.cuda()
    retinaface = torch.nn.DataParallel(retinaface).cuda()
    retinaface.training = True

    #optimizer = optim.Adam(retinaface.parameters(), lr=1e-5)
    optimizer = optim.Adam(retinaface.parameters(), lr=1e-3,weight_decay=0.0005)

    print('Start to train.')

    for epoch in range(args.epochs):
        retinaface.train()
        # retinaface.module.freeze_bn()
        retinaface.module.freeze_first_layer()

        # Training
        for iter_num,data in enumerate(dataloader_train):
            optimizer.zero_grad()
            classification_loss, bbox_regression_loss,ldm_regression_loss = retinaface([data['img'].cuda().float(), data['annot']])
            classification_loss = classification_loss.mean()
            bbox_regression_loss = bbox_regression_loss.mean()
            ldm_regression_loss = ldm_regression_loss.mean()

            loss = classification_loss + 1.0 * bbox_regression_loss + 0.5 * ldm_regression_loss

            loss.backward()
            optimizer.step()
            
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

        # Eval
        if epoch % 3 == 0:
            print('-------- RetinaFace Pytorch --------')
            print ('Evaluating epoch {}'.format(epoch))
            recall, precision = eval_widerface.evaluate(dataloader_val,retinaface)
            print('Recall:',recall)
            print('Precision:',precision)

        # Save model
        if (epoch + 1) % args.save_step == 0:
            torch.save(retinaface.state_dict(), args.save_path + '/model_epoch_{}.pt'.format(epoch + 1))


if __name__=='__main__':
    main()