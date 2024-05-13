import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import numpy as np
from models.model import FPCN as Model
from config import cfg
from utils import data_loaders
from models.utils import fps_subsample as fps_operator
from ChamferDistance import ChamferDistanceMean
from ChamferDistance import ChamferDistanceSingle

'''set args'''
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='FPCN', help='model name')
parser.add_argument('--dataset_name',  default='shapenet2048', help='dataset name')
# parser.add_argument('--pretrain', type=str, default='./ckpt.pth', help='pretrain pkl path')
parser.add_argument('--pretrain', type=str, help='pretrain pkl path')
parser.add_argument('--num_input_points', type=int, default=2048)
parser.add_argument('--num_gt_points', type=int, default=2048)
parser.add_argument('--workers', type=int,default=8, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--epochs', type=int, default=300, help='training epochs')

parser.add_argument('--optimizer', type=str, default='ADAM', help='optimizer for training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--epsilon', type=float, default=1e-08, help='epsilon for adam. default=1e-08')

parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--learning_rate_decay_steps', default=50000, type=int)
parser.add_argument('--learning_rate_decay_rate', default=0.7, type=float)
parser.add_argument('--learning_rate_clip', default=1e-6, type=float)

parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')

args = parser.parse_args()
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
USE_CUDA = args.cuda
device = torch.device("cuda:0" if torch.cuda.is_available() and USE_CUDA else "cpu")
cudnn.benchmark = True
resume_epoch = 0

'''create dirs'''
experiment_dir = Path('./experiment/')
experiment_dir.mkdir(exist_ok=True)

'''create logs'''
file_dir = Path(str(experiment_dir) + ('/%s-%s-Train-'%(args.model_name, args.dataset_name)) + str(datetime.now().strftime('%Y-%m-%d_%H-%M')))
file_dir.mkdir(exist_ok=True)
checkpoints_dir = file_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = file_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)


logger = logging.getLogger(args.model_name)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(str(log_dir) + '/train_%s_completion.txt'%args.model_name)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('---------------------------------------------------TRANING---------------------------------------------------')
logger.info('PARAMETER ...')
logger.info(args)


def save_checkpoint(epoch, step, model, optimizer, path, loss, modelnet='checkpoint'):
    savepath  = path + '/%s-%s-%f-%04d.pth' % (modelnet, str(datetime.now().strftime('%Y-%m-%d_%H:%M')), loss, epoch)
    state = {
        'global_epoch': epoch,
        'global_step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
logger.info("Random Seed: %d", args.manualSeed)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    
"data loading"
train_dataset_loader = data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)

train_dataset = train_dataset_loader.get_dataset(data_loaders.DatasetSubset.TRAIN)

train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batchSize,
                                                num_workers=args.workers,
                                                collate_fn=data_loaders.collate_fn,
                                                pin_memory=True,
                                                shuffle=True,
                                                drop_last=False)
test_dataset_loader = data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)

test_dataset = train_dataset_loader.get_dataset(data_loaders.DatasetSubset.TEST)
test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              num_workers=cfg.CONST.NUM_WORKERS,
                                              collate_fn=data_loaders.collate_fn,
                                              pin_memory=True,
                                              shuffle=False)

logger.info("The number of training data is: %d", len(train_dataset))
logger.info("The number of test data is: %d", len(test_dataset))

''''load model'''
model = Model().to(device)
opt = optim.Adam(model.parameters(), betas=(args.beta1, args.beta2), eps=args.epsilon, lr=args.learning_rate)

if args.pretrain is not None:
    print('Use pretrain model...')
    logger.info('Use pretrain model')
    checkpoint = torch.load(args.pretrain)
    start_epoch = checkpoint['global_epoch']
    start_step = checkpoint['global_step']
    best_metrics = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
else:
    print('\n No existing model, starting training from scratch...')
    start_epoch = 0
    start_step = 0
    best_metrics = 0

init_epoch = start_epoch
total_epoch = args.epochs
global_step = start_step
        
length = len(train_data_loader)
lambda1 = lambda epoch: 1 if global_step <= length*100 else args.learning_rate_decay_rate**((global_step-length*100)/args.learning_rate_decay_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda1)

cdloss = ChamferDistanceMean()
pmloss = ChamferDistanceSingle()
logger.info('Start training...')
for epoch_idx in range(init_epoch + 1, total_epoch + 1):
    print('Epoch %d/%s:' % (epoch_idx, args.epochs))
    logger.info('Epoch %d/%s:' % (epoch_idx, args.epochs))
    # train
    model.train()
    stage1_loss_save = []
    # stage2_loss_save = []
    fine_loss_save = []
    pm_loss_save = []
    loss_save = []
    n_batches = len(train_data_loader)
    with tqdm(train_data_loader) as t:
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
            # if batch_idx == 3:
            #     break
            partial = data['partial_cloud'].to(device)
            gt = data['gtcloud'].to(device)
            out = model(partial)
            stage1 = fps_operator(gt.contiguous(), 512)
            
            stage1_loss = cdloss(stage1, out[0])
            fine_loss = cdloss(gt, out[1])
            pm_loss = pmloss(partial, out[1])

            stage1_loss = torch.mean(stage1_loss, 0)
            fine_loss = torch.mean(fine_loss, 0)
            pm_loss = torch.mean(pm_loss, 0)
            
            loss = stage1_loss + fine_loss + pm_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            global_step += 1
            
            stage1_loss_save.append(stage1_loss.data.cpu())
            fine_loss_save.append(fine_loss.data.cpu())
            loss_save.append(loss.data.cpu())
            
            scheduler.step()
            temp = opt.state_dict()['param_groups'][0]['lr']
            lr = np.max((temp, args.learning_rate_clip))
            opt.param_groups[0]['lr'] = lr
            
            t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, args.epochs, batch_idx + 1, n_batches))
            t.set_postfix(loss = '%s' % ['%.4f' % l for l in [1e4 * stage1_loss.data.cpu(),
                                                              1e4 * fine_loss.data.cpu(),
                                                              1e2 * pm_loss.data.cpu()
                                                              ]])
        
    mean_stage1 = np.mean(stage1_loss_save)
    mean_fine = np.mean(fine_loss_save)
    mean_loss = np.mean(loss_save)
    
    # eval
    model.eval()
    all_loss = 0
    txt_dir = '/home/doldolouo/completion/data/ShapeNet/shapenet2048'
    catfile = os.path.join(txt_dir, 'synsetoffset2category.txt')
    cat = {}
    cate_loss = {}
    num = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[1]] = ls[0]
            cate_loss[ls[0]] = 0
            num[ls[0]] = 0
    with torch.no_grad():
        for batch_idx, (taxonomy_ids, model_ids, data) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), smoothing=0.9):
            # if batch_idx == 3:
                # break
            out = model(data['partial_cloud'].to(device))
            gt = data['gtcloud']
            loss = cdloss(gt.cuda(), out[-1].cuda())
            all_loss += loss.mean().data.cpu().numpy()
            for i in range(len(taxonomy_ids)):
                cate_name = cat[taxonomy_ids[i]]
                cate_loss[cate_name] += loss.data.cpu().numpy()[i]
                num[cate_name] += 1
        
        # Print testing results
        print()
        print('============================ TEST RESULTS ============================')
        print('Taxonomy', end='\t\t')
        print(' Average-Loss')
        for k in cate_loss.keys():
            print(' %-010s'% k, end='\t\t') 
            print('%.3f' %(cate_loss[k]/num[k]*1e4))
            
        print('Epoch %d:' % epoch_idx, end='\t')
        print('Average-CD: %.3f' % (1e4*all_loss/len(test_dataset)))
        average_loss_val = all_loss/len(test_dataset)
        
    if epoch_idx == 1:
        best_metrics = average_loss_val
        
        
    if best_metrics > average_loss_val and epoch_idx >= 5:
        best_metrics = average_loss_val
        logger.info('Save model...  best Test_loss:%.4f', 1e4*best_metrics)
        save_checkpoint(
                epoch_idx,
                global_step,
                model,
                opt,
                str(checkpoints_dir),
                best_metrics,
                args.model_name)
        print('Saving model....')

    logger.info('Training Stage1 Loss: %.4f, Training Fine Loss: %.4f, Test CD Value: %.4f, lr: %.8f',
                1e4 * mean_stage1,
                1e4 * mean_fine,
                1e4 * average_loss_val,
                opt.state_dict()['param_groups'][0]['lr'])
