import os
import argparse


DATA_DIR = '../data'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", nargs='+', type=int,
                    default=[1, 1, 1, 1, 1], help='Epoch number for each task')
parser.add_argument("--batch_size", type=int, default=8,
                    help='training batch size')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=1,
                    help='Number of labeled data')
parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'], help='Task Sequence')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from copy import deepcopy
from utils import create_log_dir
import time
from transformers import BertModel
import torch.nn as nn

import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW

from model import BaseModel
from read_data import prepare_dataloaders
from task_vectors import TaskVector

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
n_gpu = torch.cuda.device_count()

dataset_classes = {
    'amazon'  : 5,
    'yelp'    : 5,
    'yahoo'   : 10,
    'ag'      : 4,
    'dbpedia' : 14,
}

class BaseModelMH(nn.Module):
    def __init__(self, encoder, classifiers):
        super().__init__()

        self.bert = encoder.to(args.device)
        self.classifier = classifiers

    def forward(self, x):
        x = self.bert(x)
        x = x.last_hidden_state
        x = torch.mean(x, 1)

        logits = []
        for t in range(len(self.classifier)):
            logits.append(self.classifier[t](x))

        return logits

def validation(model, t, valid_loader):
    model.eval()
    acc_ = 0
    with torch.no_grad():
        total = 0
        correct = 0
        for x, mask, y in valid_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            logits = model(x)[t]
            _, pred_cls = logits.max(1)
            correct += pred_cls.eq(y.view_as(pred_cls)).sum().item()
            total += batch_size
        log.info("acc on task {} : {}".format(t, correct * 100.0 / total))
        acc_ = correct * 100.0 / total

    return acc_


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    log = create_log_dir('../logs/surgery/', 'log_{}_{}.txt'.format(str(__file__.split("/")[-1].split(".")[0]),
                                                      time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))))

    task_num = len(args.tasks)
    task_classes = [dataset_classes[task] for task in args.tasks]
    offsets = [0 for task in args.tasks]  # multi-head setting

    pretrained_model = BertModel.from_pretrained('../checkpoints/bert-base-uncased', local_files_only=True)

    # Sum the task vectors
    task_vector_sum = None
    classifiers = []
    for task_id in range(task_num):
        # Task Vector
        trained_model = BaseModel(task_classes[task_id]).to(args.device)
        trained_model.load_state_dict(torch.load('../checkpoints/' + args.tasks[task_id] + '.pth'),  strict=False)
        trained_model.to('cpu')
        classifiers.append(trained_model.classifier.to(args.device))

        task_vector_i = TaskVector(pretrained_model.state_dict(), trained_model.state_dict())

        if task_id == 0:
            task_vector_sum = task_vector_i
        else:
            task_vector_sum += task_vector_i

    # Evaluate
    _, _, test_loaders = prepare_dataloaders(DATA_DIR, args.tasks, offsets, 1, 1, args.batch_size, 128, 128)

    scaling_coef = 0.7
    log.info('--' * 20)
    log.info('scaling_coef: ' + str(scaling_coef))
    # Apply the resulting task vector
    merged_encoder = task_vector_sum.apply_to(pretrained_model, scaling_coef=scaling_coef).to(args.device)
    merged_model = BaseModelMH(merged_encoder, classifiers)
    merged_model.to(args.device)

    acc_sum = 0.
    for task_id in range(task_num):
        data_loader = test_loaders[task_id]
        length = len(data_loader)

        acc_i = validation(merged_model, task_id, data_loader)
        acc_sum += acc_i

    log.info('acc_sum / task_num: '+ str(acc_sum / task_num))
