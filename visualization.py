import argparse
import os
import numpy as np
from tqdm import tqdm

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator

from dataloaders.datasets import pascal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataloaders.utils import decode_seg_map_sequence


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        val_set = pascal.VOCSegmentation(args, split='val')
        self.nclass = val_set.NUM_CLASSES
        self.val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)
        self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            print('device_ids', self.args.gpu_ids)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def visualization(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        num_img_val = len(self.val_loader)
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

            # Save images, predictions, targets into disk
            if i % (num_img_val // 10) == 0:
                self.save_batch_images(image, output, target, i)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            # if i == 0:
            #     break

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        # mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        mIoU = self.evaluator.All_Mean_Intersection_over_Union()

        print('Validation:')
        print("Acc:{}, Acc_class:{}, fwIoU: {}".format(Acc, Acc_class, FWIoU))
        print("mIoU:{:.4f} {:.4f} {:.4f} {:.4f}".format(mIoU[0], mIoU[1], mIoU[2], mIoU[3]))
        print('Loss: %.3f' % test_loss)

    def save_batch_images(self, imgs, preds, targets, batch_index):
        (filepath, _) = os.path.split(self.args.resume)
        save_path = os.path.join(filepath, 'visualization')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        grid_image = make_grid(imgs.clone().detach().cpu(), 8, normalize=True)
        save_image(grid_image, os.path.join(save_path, 'batch_{:0>4}-img.jpg'.format(batch_index)))
        grid_image = make_grid(decode_seg_map_sequence(torch.max(preds, 1)[1].detach().cpu().numpy(),
                                                       dataset=self.args.dataset), 8, normalize=False, range=(0, 255))
        save_image(grid_image, os.path.join(save_path, 'batch_{:0>4}-pred.png'.format(batch_index)))
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(targets, 1).detach().cpu().numpy(),
                                                       dataset=self.args.dataset), 8, normalize=False, range=(0, 255))
        save_image(grid_image, os.path.join(save_path, 'batch_{:0>4}-target.png'.format(batch_index)))

# Usage: python visualization --resume run/pascal/mobilenet-deeplab-128-test/model_best.pth.tar
def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=128,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=128,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='path to test model')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.batch_size is None:
        args.batch_size = 16 * len(args.gpu_ids)

    print(args)
    torch.manual_seed(args.seed)
    tester = Tester(args)
    print('Testing model path: {} , epoch: {}'.format(tester.args.resume, tester.args.start_epoch))
    tester.visualization()


if __name__ == "__main__":
    main()
