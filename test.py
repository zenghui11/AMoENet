"""测试融合网络"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Networks.network import MODEL as net
# from Networks.network5 import MODEL as net
from msrs_data import MSRS_data
from commons import YCrCb2RGB, RGB2YCrCb, clamp
from fvcore.nn import FlopCountAnalysis

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='/scratch/test',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='/home/100')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_pretrained', default='/home/model_20.pth',
                        help='use fusion pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')
    parser.add_argument('--mode', default='low', help='low full')
    parser.add_argument('--act_variant', default='stage1', help='vi iv')

    args = parser.parse_args()
    print(args)

    init_seeds(args.seed)
    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
     
        model = net()
        x = torch.randn(1, 1, 40, 80).cuda()
        y = torch.randn(1, 1, 40, 80).cuda()
        model = model.cuda()
        
        total_params = sum(p.numel() for p in model.parameters())
        print("Params(M): %.3f" % (params_count(model) / (1000 ** 2)))
        flops = FlopCountAnalysis(model, (x, y))
        print("FLOPs(G): %.3f" % (flops.total()/1e9))

        state_dict = torch.load(args.fusion_pretrained)
        model.load_state_dict(torch.load(args.fusion_pretrained))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        model.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            # for vis_y_image, inf_image, name in test_tqdm:
                vis_y_image = vis_y_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()
                inf_image = inf_image.cuda()

                fused_image = model(vis_y_image,inf_image)
                fused_image = clamp(fused_image)
                
                rgb_tensor = YCrCb2RGB(fused_image[0], cb[0], cr[0])
                rgb_fused_image = transforms.ToPILImage()(rgb_tensor)
                rgb_fused_image.save(f'{args.save_path}/{name[0]}')