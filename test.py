import os
import random
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
from models.model import FPCN as Model
from config import cfg
from utils import data_loaders
from ChamferDistance import ChamferDistanceMean, ChamferDistanceFscore, ChamferDistanceSqrt
from EMDistance import emd_module

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='FPCN', help='model name')
parser.add_argument('--dataset_name',  default='shapenet2048', help='dataset name')
parser.add_argument('--pretrain', type=str, default='./ckpt.pth', help='pretrain pkl path')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('--batchsize', type=int, default=150, help='input batch size')
parser.add_argument('--cuda', type = bool, default = True, help='enables cuda')

args = parser.parse_args()

'''create dirs'''
experiment_dir = Path('./experiment/')
experiment_dir.mkdir(exist_ok=True)
file_dir = Path(str(experiment_dir) + ('/%s-%s-Test-'%(args.model_name, args.dataset_name)) + str(datetime.now().strftime('%Y-%m-%d_%H-%M')))
file_dir.mkdir(exist_ok=True)

'''create logs'''
log_dir = file_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)
logger = logging.getLogger(args.model_name)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(str(log_dir) + '/Test-value_%s_completion.txt'%args.model_name)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.info('---------------------------------------------------Result---------------------------------------------------')
logger.info('PARAMETER ...')
logger.info(args)

'''create random Seed'''
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
logger.info("Random Seed: %d", args.manualSeed)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)
if args.cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

'''create value metrics'''
CD_T = ChamferDistanceMean()
CD_P = ChamferDistanceSqrt()
EMDistance = emd_module.emdModule()
Fscore = ChamferDistanceFscore()

''''load model'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
if args.pretrain is not None:
    print('Use pretrain model...')
    checkpoint = torch.load(args.pretrain)
    model.load_state_dict(checkpoint['model_state_dict'])

'''create test dataset'''
test_dataset_loader = data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
test_dataset = test_dataset_loader.get_dataset(data_loaders.DatasetSubset.TEST)
test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batchsize,
                                          num_workers=cfg.CONST.NUM_WORKERS,
                                          collate_fn=data_loaders.collate_fn,
                                          pin_memory=True,
                                          shuffle=False)

'''start test'''
txt_dir = '/home/doldolouo/completion/data/ShapeNet/shapenet2048'
catfile = os.path.join(txt_dir, 'synsetoffset2category.txt')
cat = {}
cate_cd_t = {}
cate_cd_p = {}
cate_emd = {}
cate_f = {}
num = {}
model.eval()
cd_t_all = 0
cd_p_all = 0
emd_all = 0
f_score_all = 0
th = 0.01
with open(catfile, 'r') as f:
    for line in f:
        ls = line.strip().split()
        cat[ls[1]] = ls[0]
        cate_cd_t[ls[0]] = 0
        cate_cd_p[ls[0]] = 0
        cate_emd[ls[0]] = 0
        cate_f[ls[0]] = 0
        num[ls[0]] = 0
with torch.no_grad():
    for batch_idx, (taxonomy_ids, model_ids, data) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), smoothing=0.9):
        '''
        plane	    02691156
        cabinet	    02933112
        car	        02958343
        chair	    03001627
        lamp	    03636649
        couch	    04256520
        table	    04379243
        watercraft	04530566
        '''
        out = model(data['partial_cloud'].cuda())
        gt = data['gtcloud']
        cd_t = CD_T(gt.cuda(), out[-1].cuda())
        cd_p = CD_P(gt.cuda(), out[-1].cuda())
        emd, _ = EMDistance(out[-1], gt.cuda(),  0.005, 50)
        d1, d2 = Fscore(out[-1], gt.cuda())
        batch_emd = torch.mean(torch.sqrt(emd.data.cpu()), 1)
        cd_t_all += cd_t.mean().data.cpu().numpy()
        cd_p_all += cd_p.mean().data.cpu().numpy()
        emd_all += batch_emd.mean().data.cpu()

        for i in range(len(taxonomy_ids)):
            dist1 = d1[i, :]
            dist2 = d2[i, :]
            recall = float(sum((d < th).int() for d in dist2)) / float(len(dist2))
            precision = float(sum((d < th).int() for d in dist1)) / float(len(dist1))
                       
            cate_name = cat[taxonomy_ids[i]]
            cate_cd_t[cate_name] += cd_t.data.cpu().numpy()[i]
            cate_cd_p[cate_name] += cd_p.data.cpu().numpy()[i]
            cate_emd[cate_name] += batch_emd[i]
            cate_f[cate_name] += 2 * recall * precision / (recall + precision) if recall + precision else 0
            num[cate_name] += 1
            
    for k in cate_cd_t.keys():
        f_score_all += cate_f[k]/num[k]
        logger.info('%s : CD-T-%.3f, CD-P-%.3f, EMD-%.3f, F_score-%.3f' %(k, cate_cd_t[k]/num[k]*1e4, cate_cd_p[k]/num[k]*1e3, cate_emd[k]/num[k]*1e2, cate_f[k]/num[k]))
        
    logger.info('Average-CD-T : %.3f' %(1e4*cd_t_all/len(test_dataset)*args.batchsize))
    logger.info('Average-CD-P : %.3f' %(1e3*cd_p_all/len(test_dataset)*args.batchsize))
    logger.info('Average-EMD : %.3f' %(1e2*emd_all/len(test_dataset)*args.batchsize))
    logger.info('Average-F-Score : %.3f' %(f_score_all / 8))
        
