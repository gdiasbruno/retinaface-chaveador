from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import MALF, detection_collate_finetune, preproc_finetune, cfg_mnet, cfg_re50, cfg_re50_finetune_large
from layers.modules import MultiBoxLoss_finetune
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Fine-tune')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--validation_dataset', default='', help='Validation dataset directory')
parser.add_argument('--validation_internal', default=1, type=int, help='Interval of iteractions in which a new validation will run')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='/weights/', help='Location to save checkpoint models')
parser.add_argument('--pretrained_weights', default=None, help='Path to pretrained weights')
parser.add_argument('--unfreeze_layers', default=None, nargs='+', help='List of layer names to unfreeze. Layers name: body, fpn, ssh, ClassHead, BboxHead, LandmarkHead')
parser.add_argument('--output_weight_path', default=None, type=str, help='Path of the file where the weights will be saved')
parser.add_argument('--logs_path', default=None, type=str, help='Path of the file where the logs will be saved')

log_lines = []
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50_finetune_large

print('cfg', cfg) # added by me
log_lines.append('cfg: ' + str(cfg)) # added by me

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
validation_dataset = args.validation_dataset
save_folder = args.save_folder
validation_interval = args.validation_internal
net = RetinaFace(cfg=cfg)
# print("Printing net...")
# print(net)

### START - changed code from train

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('ckpt_keys: ', len(ckpt_keys))
    # print('model_keys: ', len(model_keys))
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    log_lines.append('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    log_lines.append('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=True)
    return model


if args.pretrained_weights is not None:
  net = load_model(net, args.pretrained_weights, False)

# freeze all weights
for param in net.parameters():
  param.requires_grad = False

if args.pretrained_weights is not None:
  print('args.unfreeze_layers', args.unfreeze_layers)
  log_lines.append('args.unfreeze_layers: ' + str(args.unfreeze_layers))
  for name, param in net.named_parameters():
      if any(layer_name in name for layer_name in args.unfreeze_layers):
        param.requires_grad = True
        print("Unfreezing ", name)
        log_lines.append("Unfreezing " + str(name))

### END - changed code from train

if args.resume_net is not None:
    print('Loading resume network...')
    log_lines.append('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss_finetune(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def validate(validation_dataset):
    net.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_batch_iterator = iter(data.DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_finetune))

    with torch.no_grad():
        for val_iteration in range(len(validation_dataset) // batch_size):
            val_images, val_targets = next(val_batch_iterator)
            val_images = val_images.cuda()
            val_targets = [anno.cuda() for anno in val_targets]

            val_out = net(val_images)
            val_loss_l, val_loss_c = criterion(val_out, priors, val_targets)
            val_loss += cfg['loc_weight'] * val_loss_l + val_loss_c

    val_loss /= len(validation_dataset)
    return val_loss.item()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')
    log_lines.append('Loading Dataset...')

    dataset = MALF( training_dataset,preproc_finetune(img_dim, rgb_mean))
    dataset_validation = MALF(validation_dataset,preproc_finetune(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % validation_interval == 0:  # Run validation every 'validation_interval' iterations
          val_loss = validate(dataset_validation)
          print(f'Validation Loss: {val_loss}')
          log_lines.append(f'Validation Loss: {val_loss}')
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_finetune))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        log_lines.append('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))


    place_save = args.output_weight_path
    print('place_save', place_save)
    log_lines.append('place_save: '+ str(place_save))

    torch.save(net.state_dict(), place_save)
    if args.logs_path is not None:
      with open(args.logs_path, 'w') as f:
        for line in log_lines:
          f.write(line + '\n')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
