from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import sys
import copy
import random
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain

from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from gcn.models import LAGCN
from tqdm import trange
import random

exc_path = sys.path[0]

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--concat", type=int, default=4)
parser.add_argument('--runs', type=int, default=10, help='The number of experiments.')

parser.add_argument('--dataset', default='cora', help='Dataset string.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

# Adapted from GRAND: https://github.com/THUDM/GRAND
def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss


# Load data
adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

# Normalize adj and features
features = features.toarray()
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) 
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

cvae_model = torch.load("{}/model/{}.pkl".format(exc_path, args.dataset))

def augment_features_with_triangle_check(features, random_prob=0.1):
    augmented_features = [feature.clone() for feature in features]
    adj_matrix = adj_normalized.to_dense().cpu().numpy()  # 将邻接矩阵转换为NumPy数组
    num_nodes = adj_matrix.shape[0]
    for i in range(num_nodes):
        neighbors = adj_matrix[i].nonzero()[0]
        triangle = False
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                if adj_matrix[neighbors[j]][neighbors[k]] != 0:
                    triangle = True
                    break
            if triangle:
                break

        if not triangle and random.random() < random_prob:
            augmented_features[i] = 0

    return augmented_features

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()

        adj_matrix = adj_normalized.to_dense().cpu().numpy()  # 将邻接矩阵转换为NumPy数组
        num_nodes = adj_matrix.shape[0]

        for i in range(num_nodes):
            neighbors = np.nonzero(adj_matrix[i])[0]  # 获取第i个节点的邻居节点
            triangle = False
            for j in range(len(neighbors)):

                for k in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j]][neighbors[k]] != 0:
                        triangle = True
                        break
                if triangle:
                    break

            if not triangle:
                augmented_features[i] = 0  # 将不属于三角形结构的节点对应的增强特征置为0
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list


if args.cuda:
    adj_normalized = adj_normalized.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

all_val = []
all_test = []
for i in trange(args.runs, desc='Run Train'):

    # Model and optimizer
    model = LAGCN(concat=args.concat+1,
                  nfeat=features.shape[1],
                  nhid=args.hidden,
                  nclass=labels.max().item() + 1,
                  dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    # Train model
    best = 999999999
    best_model = None
    best_X_list = None
    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            augmented_features = augment_features_with_triangle_check(X_list+[features_normalized])
            output_list.append(torch.log_softmax(model(augmented_features, adj_normalized), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(output_list)

        loss_consis = consis_loss(output_list)
        loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        model.eval()
        val_X_list = get_augmented_features(args.concat)
        augmented_features = augment_features_with_triangle_check(val_X_list + [features_normalized])
        output = model(augmented_features, adj_normalized)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()))
                
        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            best_X_list = copy.deepcopy(val_X_list)

    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], adj_normalized)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    all_val.append(acc_val.item())
    all_test.append(acc_test.item())

print(np.mean(all_val), np.std(all_val), np.mean(all_test), np.std(all_test))
