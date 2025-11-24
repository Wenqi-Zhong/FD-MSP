import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import logging
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from model.feature_extractor import resnet_feature_extractor
from model.classifier import ASPP_Classifier_Gen
from model.discriminator import FCDiscriminator

# 导入ScaleProtoSeg相关模块
from model.prototype_network import PrototypeNetwork
from model.prototype_loss import PrototypeLoss

from utils.util import *
from data import create_dataset
import cv2

IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMG_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 16
IGNORE_LABEL = 250
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 2
NUM_STEPS = 20000
NUM_STEPS_STOP = 20000
POWER = 0.9
RANDOM_SEED = 1234
RESUME = './snapshots/model_phase2.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

# 原型网络相关参数
PROTOTYPE_SHAPE = (20, 128, 1, 1)  # (num_prototypes, channels, height, width)
NUM_SCALES = 4
PROTOTYPE_ACTIVATION_FUNCTION = 'log'
LAMBDA_PROTO = 0.1
LAMBDA_DIVERSITY = 0.01
LAMBDA_SEPARATION = 0.01

SET = 'train'

def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network Phase 3")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gpus", type=str, default="0", help="selected gpus")
    parser.add_argument("--dist", action="store_true", help="DDP")
    parser.add_argument("--ngpus_per_node", type=int, default=1, help='number of gpus in each node')
    parser.add_argument("--print-every", type=int, default=20, help='output message every n iterations')
    
    parser.add_argument("--layer", type=int, default=1, help='feature extractor layer setting (0, 1 or 2)')
    parser.add_argument("--hidden_dim", type=int, default=128, help='hidden dimension for ASPP classifier')

    parser.add_argument("--src_dataset", type=str, default="endotect", help='training source dataset')
    parser.add_argument("--tgt_dataset", type=str, default="cvc_clinicdb", help='training target dataset')
    parser.add_argument("--tgt_val_dataset", type=str, default="cvc_clinicdb_val", help='training target dataset')
    parser.add_argument("--noaug", action="store_true", help="augmentation")
    parser.add_argument('--resize', type=int, default=384, help='resize image size')
    parser.add_argument("--clrjit_params", type=str, default="0.5,0.5,0.5,0.2", help='brightness,contrast,saturation,hue')
    parser.add_argument('--rcrop', type=str, default='384,384', help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')
    parser.add_argument('--src_rootpath', type=str, default='datasets/EndoTect')
    parser.add_argument('--tgt_rootpath', type=str, default='datasets/CVC-ClinicDB')
    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--no_droplast', action='store_true')
    parser.add_argument('--pseudo_labels_folder', type=str, default='')
    parser.add_argument('--soft_labels_folder', type=str, default='')
    parser.add_argument('--thresholds_path', type=str, default='avg')
    
    parser.add_argument("--batch_size_val", type=int, default=4, help='batch_size for validation')
    parser.add_argument("--resume", type=str, default=RESUME, help='resume weight')
    parser.add_argument("--freeze_bn", action="store_true", help="augmentation")
    parser.add_argument("--src_loss_weight", type=float, default=1.0, help='weight for src_loss')
    
    # 原型网络相关参数
    parser.add_argument("--prototype_shape", type=str, default="20,128,1,1", help='prototype shape')
    parser.add_argument("--num_scales", type=int, default=NUM_SCALES, help='number of scales for prototype')
    parser.add_argument("--lambda_proto", type=float, default=LAMBDA_PROTO, help='weight for prototype loss')
    parser.add_argument("--lambda_diversity", type=float, default=LAMBDA_DIVERSITY, help='weight for diversity loss')
    parser.add_argument("--lambda_separation", type=float, default=LAMBDA_SEPARATION, help='weight for separation loss')
    
    # 添加分布式训练必需的参数
    parser.add_argument("--local-rank", type=int, default=0, help='local rank for distributed training')
    
    return parser.parse_args()


args = get_arguments()


def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main_worker(gpu, world_size, dist_url):
    """Create the model and start the training."""
    if gpu == 0:
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
        logFilename = os.path.join(args.snapshot_dir, str(time.time()))
        logging.basicConfig(
                        level = logging.INFO,
                        format ='%(asctime)s-%(levelname)s-%(message)s',
                        datefmt = '%y-%m-%d %H:%M',
                        filename = logFilename,
                        filemode = 'w+')
        filehandler = logging.FileHandler(logFilename, encoding='utf-8')
        logger = logging.getLogger()
        logger.addHandler(filehandler)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.info(args)

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

    print("gpu: {}, world_size: {}".format(gpu, world_size))
    print("dist_url: ", dist_url)

    # 只有在分布式训练时才初始化分布式
    if dist_url is not None:
        torch.cuda.set_device(gpu)
        args.batch_size = args.batch_size // world_size
        args.batch_size_val = args.batch_size_val // world_size
        args.num_workers = args.num_workers // world_size
        dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=gpu)
    else:
        # 非分布式训练 - 使用第一个可用GPU
        torch.cuda.set_device(0)  # 确保使用GPU 0

    if gpu == 0:
        logger.info("args.batch_size: {}, args.batch_size_val: {}".format(args.batch_size, args.batch_size_val))

    device = torch.device("cuda" if not args.cpu else "cpu")
    args.world_size = world_size

    if gpu == 0:
        logger.info("args: {}".format(args))

    # Create network
    if args.model == 'DeepLab':
        if args.resume:
            resume_weight = torch.load(args.resume, map_location='cpu')
            print("args.resume: ", args.resume)
            model_B2_weights = resume_weight['model_B2_state_dict']
            model_B_weights = resume_weight['model_B_state_dict']
            head_weights = resume_weight['head_state_dict']
            classifier_weights = resume_weight['classifier_state_dict']
            model_B2_weights = {k.replace("module.", ""):v for k,v in model_B2_weights.items()}
            model_B_weights = {k.replace("module.", ""):v for k,v in model_B_weights.items()}
            head_weights = {k.replace("module.", ""):v for k,v in head_weights.items()}
            classifier_weights = {k.replace("module.", ""):v for k,v in classifier_weights.items()}

        if gpu == 0:
            logger.info("freeze_bn: {}".format(args.freeze_bn))

        model = resnet_feature_extractor('resnet101', 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', freeze_bn=args.freeze_bn)

        if args.layer == 0:
            ndf = 64
            model_B2 = nn.Sequential(model[0], model[1], model[2], model[3])
            model_B = nn.Sequential(model[4], model[5], model[6], model[7])
        elif args.layer == 1:
            ndf = 256
            model_B2 = nn.Sequential(model[0], model[1], model[2], model[3], model[4])
            model_B = nn.Sequential(model[5], model[6], model[7])
        elif args.layer == 2:
            ndf = 512
            model_B2 = nn.Sequential(model[0], model[1], model[2], model[3], model[4], model[5])
            model_B = nn.Sequential(model[6], model[7])
        
        if args.resume:
            model_B2.load_state_dict(model_B2_weights)
            model_B.load_state_dict(model_B_weights)

        classifier = ASPP_Classifier_Gen(2048, [3, 6, 9, 12], [3, 6, 9, 12], args.num_classes, hidden_dim=args.hidden_dim)
        head, classifier = classifier.head, classifier.classifier
        if args.resume:
            head.load_state_dict(head_weights)
            classifier.load_state_dict(classifier_weights)

        # 添加原型网络
        args.prototype_shape = [int(x.strip()) for x in args.prototype_shape.split(",")]
        prototype_network = PrototypeNetwork(
            feature_dim=args.hidden_dim,
            prototype_shape=tuple(args.prototype_shape),
            num_classes=args.num_classes,
            num_scales=args.num_scales,
            activation_function=PROTOTYPE_ACTIVATION_FUNCTION
        )

    model_B2.train()
    model_B.train()
    head.train()
    classifier.train()
    prototype_network.train()

    if gpu == 0:
        logger.info(model_B2)
        logger.info(model_B)
        logger.info(head)
        logger.info(classifier)
        logger.info(prototype_network)
    else:
        logger = None

    if gpu == 0:
        logger.info("args.noaug: {}, args.resize: {}, args.rcrop: {}, args.hflip: {}, args.noshuffle: {}, args.no_droplast: {}".format(args.noaug, args.resize, args.rcrop, args.hflip, args.noshuffle, args.no_droplast))
    args.rcrop = [int(x.strip()) for x in args.rcrop.split(",")]
    args.clrjit_params = [float(x) for x in args.clrjit_params.split(',')]

    datasets = create_dataset(args, logger)
    sourceloader_iter = enumerate(datasets.source_train_loader)
    targetloader_iter = enumerate(datasets.target_train_loader)

    # define optimizer
    model_params = [
        {'params': list(model_B2.parameters()) + list(model_B.parameters())},
        {'params': list(head.parameters()) + list(classifier.parameters()), 'lr': args.learning_rate * 10},
        {'params': list(prototype_network.parameters()), 'lr': args.learning_rate * 5}
    ]
    optimizer = optim.SGD(model_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    assert len(optimizer.param_groups) == 3
    optimizer.zero_grad()

    # 根据是否分布式训练来决定是否使用DDP
    if dist_url is not None:
        model_B2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B2)
        model_B2 = torch.nn.parallel.DistributedDataParallel(model_B2.cuda(), device_ids=[gpu], find_unused_parameters=True)

        model_B = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_B)
        model_B = torch.nn.parallel.DistributedDataParallel(model_B.cuda(), device_ids=[gpu], find_unused_parameters=True)

        head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head)
        head = torch.nn.parallel.DistributedDataParallel(head.cuda(), device_ids=[gpu], find_unused_parameters=True)

        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier.cuda(), device_ids=[gpu], find_unused_parameters=True)
        
        prototype_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prototype_network)
        prototype_network = torch.nn.parallel.DistributedDataParallel(prototype_network.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        # 非分布式训练，直接移动到GPU
        model_B2 = model_B2.cuda()
        model_B = model_B.cuda()
        head = head.cuda()
        classifier = classifier.cuda()
        prototype_network = prototype_network.cuda()

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    prototype_loss = PrototypeLoss(
        num_classes=args.num_classes,
        num_prototypes=args.prototype_shape[0],
        lambda_diversity=args.lambda_diversity,
        lambda_separation=args.lambda_separation
    )

    interp = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)

    # set up tensor board
    if args.tensorboard and gpu == 0:
        writer = SummaryWriter(args.snapshot_dir)

    thresholds = np.load(args.thresholds_path)
    class_list = ["background", "polyp"]
    if gpu == 0:
        logger.info('successfully load class-wise thresholds from {}'.format(args.thresholds_path))
        for c in range(len(class_list)):
            logger.info("class {}: {}, threshold: {}".format(c, class_list[c], thresholds[c]))
    thresholds = torch.from_numpy(thresholds).cuda()

    scaler = torch.cuda.amp.GradScaler()
    best_miou = 0.0
    filename = None
    epoch_s, epoch_t = 0, 0
    
    for i_iter in range(args.num_steps):

        model_B2.train()
        model_B.train()
        head.train()
        classifier.train()
        prototype_network.train()

        loss_seg_value = 0
        loss_src_seg_value = 0
        loss_proto_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):

            # train with source
            try:
                _, batch = sourceloader_iter.__next__()
            except StopIteration:
                epoch_s += 1
                if hasattr(datasets, 'source_train_sampler') and datasets.source_train_sampler is not None:
                    datasets.source_train_sampler.set_epoch(epoch_s)
                sourceloader_iter = enumerate(datasets.source_train_loader)
                _, batch = sourceloader_iter.__next__()
                
            images = batch['img'].cuda()
            labels = batch['label'].cuda()
            src_size = images.shape[-2:]

            with torch.cuda.amp.autocast():

                feat_src = model_B2(images)
                feat_B_src = model_B(feat_src)
                feat_head_src = head(feat_B_src)
                pred = classifier(feat_head_src)
                pred = interp(pred)

                # 原型网络处理
                proto_distances, proto_activations = prototype_network(feat_head_src)
                proto_pred = prototype_network.classify(proto_activations)
                proto_pred = interp(proto_pred)

                # 对标签进行下采样以匹配原型特征的尺寸
                # feat_head_src的尺寸通常是原图的1/8，所以需要下采样标签
                labels_downsampled = F.interpolate(labels.unsqueeze(1).float(), 
                                                 size=proto_distances.shape[-2:], 
                                                 mode='nearest').squeeze(1).long()

                loss_seg = seg_loss(pred, labels)
                loss_proto_seg = seg_loss(proto_pred, labels)
                loss_proto_reg = prototype_loss(proto_distances, proto_activations, labels_downsampled, 
                                              prototype_vectors=prototype_network.prototype_layer.prototype_vectors)
                
                loss = loss_seg + args.lambda_proto * (loss_proto_seg + loss_proto_reg)

                # proper normalization
                loss = args.src_loss_weight * loss / args.iter_size
                loss_src_seg_value += loss_seg / args.iter_size
                loss_proto_value += (loss_proto_seg + loss_proto_reg) / args.iter_size
                
            scaler.scale(loss).backward()

            # train with target
            try:
                _, batch = targetloader_iter.__next__()
            except StopIteration:
                epoch_t += 1
                if hasattr(datasets, 'target_train_sampler') and datasets.target_train_sampler is not None:
                    datasets.target_train_sampler.set_epoch(epoch_t)
                targetloader_iter = enumerate(datasets.target_train_loader)
                _, batch = targetloader_iter.__next__()
                
            images = batch['img'].cuda()
            soft_labels = batch['lpsoft'].cuda()
            tgt_size = images.shape[-2:]

            with torch.no_grad():
                soft_labels = F.softmax(soft_labels, 1)
                
                images_full = batch['img_full'].cuda()
                weak_params = batch['weak_params']
                resize_params = weak_params['RandomSized']
                crop_params = weak_params['RandomCrop']
                flip_params = weak_params['RandomHorizontallyFlip']
                
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        feat_full = model_B2(images_full)
                        feat_B_full = model_B(feat_full)
                        feat_head_full = head(feat_B_full)
                        pred_full = F.softmax(interp(classifier(feat_head_full)), 1)
                        
                        # 原型网络预测
                        proto_distances_full, proto_activations_full = prototype_network(feat_head_full)
                        proto_pred_full = F.softmax(interp(prototype_network.classify(proto_activations_full)), 1)
                        
                        # 融合预测结果
                        pred_full = (pred_full + proto_pred_full) / 2.0

                        pred_labels = []
                        for b in range(pred_full.shape[0]):
                            # restore pred_full to crop
                            h, w = resize_params[0][b], resize_params[1][b]
                            pred_resize_b = F.interpolate(pred_full[b].unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)[0]
                            ys, ye, xs, xe = crop_params[0][b], crop_params[1][b], crop_params[2][b], crop_params[3][b]
                            pred_crop_b = pred_resize_b[:, ys:ye, xs:xe]
                            if flip_params[b]:
                                pred_crop_b = torch.flip(pred_crop_b, dims=(2,))
                            pred_labels.append(pred_crop_b)
                        pred_labels = torch.stack(pred_labels, 0)
                        assert pred_labels.shape[-2:] == tgt_size
                pseudo_labels = (pred_labels + soft_labels) / 2.0

            with torch.cuda.amp.autocast():
                feat_tgt = model_B2(images)
                feat_B_tgt = model_B(feat_tgt)
                feat_head_tgt = head(feat_B_tgt)
                pred = classifier(feat_head_tgt)
                pred = interp(pred)

                # 原型网络处理
                proto_distances, proto_activations = prototype_network(feat_head_tgt)
                proto_pred = prototype_network.classify(proto_activations)
                proto_pred = interp(proto_pred)

                conf, pseudo_labels = pseudo_labels.max(1)
                pseudo_labels[conf < thresholds[pseudo_labels]] = args.ignore_label
                pseudo_labels = pseudo_labels.detach()
                
                loss_seg = seg_loss(pred, pseudo_labels)
                loss_proto_seg = seg_loss(proto_pred, pseudo_labels)
                
                loss = loss_seg + args.lambda_proto * loss_proto_seg

                # proper normalization
                loss = loss / args.iter_size
                loss_seg_value += loss_seg / args.iter_size
                loss_proto_value += loss_proto_seg / args.iter_size
                
            scaler.scale(loss).backward()

        # 根据是否分布式来决定是否进行all_reduce
        if dist_url is not None:
            n = torch.tensor(1.0).cuda()
            dist.all_reduce(n), dist.all_reduce(loss_seg_value), dist.all_reduce(loss_src_seg_value), dist.all_reduce(loss_proto_value)
            
            loss_seg_value = loss_seg_value.item() / n.item()
            loss_src_seg_value = loss_src_seg_value.item() / n.item()
            loss_proto_value = loss_proto_value.item() / n.item()
        else:
            # 非分布式训练，直接获取值
            loss_seg_value = loss_seg_value.item()
            loss_src_seg_value = loss_src_seg_value.item()
            loss_proto_value = loss_proto_value.item()
        
        scaler.step(optimizer)
        scaler.update()
        
        if args.tensorboard and gpu == 0:
            scalar_info = {
                'loss_seg': loss_seg_value,
                'loss_src_seg': loss_src_seg_value,
                'loss_proto': loss_proto_value,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        if gpu == 0 and i_iter % args.print_every == 0:
            logger.info('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}, loss_src_seg = {3:.3f}, loss_proto = {4:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value, loss_src_seg_value, loss_proto_value))
        
        if gpu == 0 and i_iter >= args.num_steps_stop - 1:
            logger.info('save model ...')
            filename = osp.join(args.snapshot_dir, 'Phase3_' + str(args.num_steps_stop) + '.pth')
            save_file = {
                'model_B2_state_dict': model_B2.state_dict(), 
                'model_B_state_dict': model_B.state_dict(),
                'head_state_dict': head.state_dict(), 
                'classifier_state_dict': classifier.state_dict(),
                'prototype_network_state_dict': prototype_network.state_dict()
            }
            torch.save(save_file, filename)
            logger.info("saving checkpoint model to {}".format(filename))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            miou, mdice, class_ious, class_dices, loss_val = validate(model_B2, model_B, head, classifier, prototype_network, seg_loss, gpu, logger if gpu == 0 else None, datasets.target_valid_loader, i_iter)
            if args.tensorboard and gpu == 0:
                scalar_info = {
                    'miou_val': miou,
                    'mdice_val': mdice,
                    'loss_val': loss_val,
                    'iou_background': class_ious[0],
                    'iou_polyp': class_ious[1],
                    'dice_background': class_dices[0],
                    'dice_polyp': class_dices[1]
                }
                for k, v in scalar_info.items():
                    writer.add_scalar(k, v, i_iter)

            if gpu == 0 and miou > best_miou:
                best_miou = miou
                logger.info('taking snapshot ...')
                if filename is not None and os.path.exists(filename):
                    os.remove(filename)
                filename = osp.join(args.snapshot_dir, 'Phase3_best.pth')
                save_file = {
                    'model_B2_state_dict': model_B2.state_dict(), 
                    'model_B_state_dict': model_B.state_dict(),
                    'head_state_dict': head.state_dict(), 
                    'classifier_state_dict': classifier.state_dict(),
                    'prototype_network_state_dict': prototype_network.state_dict()
                }
                torch.save(save_file, filename)
                logger.info("saving best model to {} with mIoU = {:.3f}, mDice = {:.3f}".format(filename, miou, mdice))

    if args.tensorboard and gpu == 0:
        writer.close()


def validate(model_B2, model_B, head, classifier, prototype_network, seg_loss, gpu, logger, testloader, i_iter):
    """详细的验证函数，计算每个类别的IoU和Dice指标"""
    model_B2.eval()
    model_B.eval()
    head.eval()
    classifier.eval()
    prototype_network.eval()
    
    interp = nn.Upsample(size=(args.rcrop[1], args.rcrop[0]), mode='bilinear', align_corners=True)
    
    total_loss = 0.0
    # 为每个类别分别计算指标
    class_intersections = np.zeros(args.num_classes)
    class_unions = np.zeros(args.num_classes)
    class_pred_areas = np.zeros(args.num_classes)
    class_label_areas = np.zeros(args.num_classes)
    
    with torch.no_grad():
        for i_iter, batch in enumerate(testloader):
            images = batch['img'].cuda()
            labels = batch['label'].cuda()
            
            feat = model_B2(images)
            feat_B = model_B(feat)
            feat_head = head(feat_B)
            pred = classifier(feat_head)
            pred = interp(pred)
            
            # 原型网络预测
            proto_distances, proto_activations = prototype_network(feat_head)
            proto_pred = prototype_network.classify(proto_activations)
            proto_pred = interp(proto_pred)
            
            # 融合预测结果
            pred = (pred + proto_pred) / 2.0
            
            loss = seg_loss(pred, labels)
            total_loss += loss.item()
            
            pred = pred.data.max(1)[1].cpu().numpy()
            labels = labels.cpu().numpy()
            
            # 计算每个类别的IoU和Dice
            for cls in range(args.num_classes):
                pred_cls = (pred == cls)
                label_cls = (labels == cls)
                
                intersection = np.logical_and(pred_cls, label_cls).sum()
                union = np.logical_or(pred_cls, label_cls).sum()
                pred_area = pred_cls.sum()
                label_area = label_cls.sum()
                
                class_intersections[cls] += intersection
                class_unions[cls] += union
                class_pred_areas[cls] += pred_area
                class_label_areas[cls] += label_area
    
    model_B2.train()
    model_B.train()
    head.train()
    classifier.train()
    prototype_network.train()
    
    # 计算每个类别的IoU和Dice
    class_ious = []
    class_dices = []
    class_names = ["background", "polyp"]
    
    for cls in range(args.num_classes):
        iou = class_intersections[cls] / (class_unions[cls] + 1e-10)
        dice = 2 * class_intersections[cls] / (class_pred_areas[cls] + class_label_areas[cls] + 1e-10)
        class_ious.append(iou)
        class_dices.append(dice)
    
    # 计算平均指标
    miou = np.mean(class_ious)
    mdice = np.mean(class_dices)
    avg_loss = total_loss / len(testloader)
    
    if logger:
        logger.info('=== 验证结果 (第{}步) ==='.format(i_iter))
        logger.info('Val result: mIoU = {:.3f}, mDice = {:.3f}, loss = {:.3f}'.format(miou, mdice, avg_loss))
        for cls in range(args.num_classes):
            logger.info('Class_{} Result: iou = {:.3f}, dice = {:.3f}'.format(
                class_names[cls], class_ious[cls], class_dices[cls]))
        logger.info('=' * 50)
    
    return miou, mdice, class_ious, class_dices, avg_loss


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def main():
    args.gpus = [int(i) for i in args.gpus.split(',')]
    args.ngpus_per_node = len(args.gpus)
    
    # 如果只有一个GPU或者没有启用分布式训练，则使用单GPU模式
    if args.ngpus_per_node == 1 or not args.dist:
        # 非分布式训练，直接在单GPU上运行
        main_worker(0, 1, None)
    else:
        # 分布式训练
        port = find_free_port()
        dist_url = f"tcp://127.0.0.1:{port}"
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, dist_url))


if __name__ == '__main__':
    main() 