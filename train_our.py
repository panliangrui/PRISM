import enum
import re
# from symbol import testlist_star_expr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import os
import pandas as pd
from sklearn.model_selection import KFold

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset
# import redis
import pickle
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    roc_auc_score, roc_curve
import random
import torch.backends.cudnn as cudnn
import json
import joblib

torch.multiprocessing.set_sharing_strategy('file_system')
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
from torch.cuda.amp import GradScaler, autocast
# from torch_geometric.data import Data, Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import h5py
import os
import glob
import h5py
import torch

class BagDataset(Dataset):
    def __init__(self, train_path, args) -> None:
        super(BagDataset).__init__()
        self.train_path = train_path
        self.args = args


    def get_bag_feats(self, csv_file_df, args):
        feats_csv_path = csv_file_df.iloc[4]
        # print(feats_csv_path)

        directory = r'X:\projects\mianyi\all_features\xiangya2\graph_edge'  # gigapath'

        feats_csv_path = os.path.join(directory, feats_csv_path)



        with h5py.File(feats_csv_path, "r") as f:
            # x_img_256 = f['x_img_256'][:]  # 图像 256 特征
            # coord_256 = f['x_img_256_coord'][:]  # 坐标
            image_path_256 = f['image_path_256_fea'][:]  # 图像路径特征
            edge_index_256 = f['edge_index_image_256'][:]  # 图像 256 边

            # x_img_512 = f['x_img_512'][:]  # 图像 256 特征
            # coord_512 = f['x_img_512_coord'][:]  # 坐标
            image_path_512 = f['image_path_512_fea'][:]  # 图像路径特征
            edge_index_512 = f['edge_index_image_512'][:]  # 图像 256 边

            # 读取节点特征 [N, C]
            x = torch.from_numpy(f['x'][:]).float()
            # 读取边索引 [2, E]
            edge_index = torch.from_numpy(f['edge_index'][:]).long()
            # 读取边类型 [E]
            edge_type = torch.from_numpy(f['edge_type'][:]).long()
            # 读取标签 (假设是图级别的分类标签，或者是节点级别的)
            # 如果是节点分类：
            if 'y' in f:
                y = torch.from_numpy(f['y'][:]).long()
            else:
                # 如果没有标签，创建一个假的或者根据你的业务逻辑处理
                y = torch.tensor([0])

                # 读取位置信息 (可选)
            pos = None
            if 'pos' in f:
                pos = torch.from_numpy(f['pos'][:]).float()
        # data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y, pos=pos)

        label = np.zeros(args.num_classes)
        if args.num_classes == 1:
            label[0] = csv_file_df.iloc[3]
        else:
            if int(csv_file_df.iloc[3]) <= (len(label) - 1):
                label[int(csv_file_df.iloc[3])] = 1
        label = torch.tensor(np.array(label))
        #####features
        image_path_256 = torch.tensor(np.array(image_path_256)).float()
        image_path_512 = torch.tensor(np.array(image_path_512)).float()
        return label, image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type

    def dropout_patches(self, feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0] * (1 - p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0] * p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats

    def __getitem__(self, idx):
        row = self.train_path.iloc[idx]
        feats_csv_path = row.iloc[3]
        # 2. 如果是坏文件，就读取下一个数据 (idx + 1)
        if feats_csv_path == '1576525-1.h5':
            print(f"Skipping bad file: {feats_csv_path}, replacing with next sample.")
            # 使用取模运算 % 避免索引越界
            new_idx = (idx + 1) % len(self.train_path)
            return self.__getitem__(new_idx)
        # label, feats, feats_TME = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        label, image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        # return label, feats, feats_TME
        return label, image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type

    def __len__(self):
        return len(self.train_path)


# class BagDataset(InMemoryDataset):
#     def __init__(self, root, train_path, args, transform=None, pre_transform=None):
#         """
#         root: 存储处理后数据（processed）的根目录
#         train_path: csv 文件的路径，包含所有样本的索引信息
#         args: 参数配置
#         """
#         self.train_path = train_path
#         self.args = args
#         self.df = pd.read_csv(train_path)
#
#         # 基础目录，从原始代码中提取
#         self.data_dir = r'X:\projects\mianyi\all_features\xiangya2\graph_edge'
#
#         super().__init__(root, transform, pre_transform)
#         # 加载处理好的数据
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         # 由于我们直接读取 train_path (csv) 和 固定的 data_dir，这里可以留空
#         return []
#
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
#
#     def download(self):
#         # 不需要下载
#         pass
#
#     def process(self):
#         data_list = []
#
#         print(f"开始处理数据，共 {len(self.df)} 个样本...")
#
#         # 遍历 CSV 中的每一行
#         for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
#             # --- 原始 get_bag_feats 的逻辑开始 ---
#
#             # 获取文件名 (假设在第 5 列，索引 4)
#             feats_csv_path = row.iloc[4]
#             full_path = os.path.join(self.data_dir, feats_csv_path)
#
#             if not os.path.exists(full_path):
#                 print(f"Warning: File not found {full_path}, skipping.")
#                 continue
#
#             with h5py.File(full_path, "r") as f:
#                 # 1. 读取 256 尺度特征
#                 image_path_256 = torch.tensor(f['image_path_256_fea'][:]).float()
#                 edge_index_256 = torch.tensor(f['edge_index_image_256'][:]).long()
#
#                 # 2. 读取 512 尺度特征
#                 image_path_512 = torch.tensor(f['image_path_512_fea'][:]).float()
#                 edge_index_512 = torch.tensor(f['edge_index_image_512'][:]).long()
#
#                 # 3. 读取 TME 图特征 (Graph Low)
#                 x = torch.from_numpy(f['x'][:]).float()
#                 edge_index = torch.from_numpy(f['edge_index'][:]).long()
#                 edge_type = torch.from_numpy(f['edge_type'][:]).long()
#
#                 # 读取位置
#                 pos = None
#                 if 'pos' in f:
#                     pos = torch.from_numpy(f['pos'][:]).float()
#
#                 # 读取原始 H5 中的 y (可能是节点标签)
#                 if 'y' in f:
#                     node_y = torch.from_numpy(f['y'][:]).long()
#                 else:
#                     node_y = torch.tensor([0])
#
#             # 4. 处理 Slide 级别标签 (Label)
#             label = np.zeros(self.args.num_classes)
#             # 假设 row.iloc[3] 是 label 的索引
#             label_idx = int(row.iloc[3])
#
#             if self.args.num_classes == 1:
#                 label[0] = label_idx
#             else:
#                 if label_idx < self.args.num_classes:
#                     label[label_idx] = 1
#
#             y_slide = torch.tensor(label).float()  # Slide 级标签通常是 float 用于 BCE 或 Long 用于 CE
#             # 如果是 CrossEntropyLoss，通常需要 LongTensor 且不是 one-hot，这里保持原逻辑的一致性
#             # 如果原模型需要 one-hot，保持 float；如果需要 class index，建议改存 label_idx
#
#             # --- 构造 Data 对象 ---
#             # 将所有分散的数据打包到一个 Data 对象中
#             # PyG 的 Data 对象支持任意属性
#             data = Data(
#                 x=x,
#                 edge_index=edge_index,
#                 edge_type=edge_type,
#                 pos=pos,
#
#                 # 将 Slide Label 存为 y (标准做法)
#                 y=y_slide.unsqueeze(0),  # 增加一个维度 [1, num_classes]
#
#                 # 存储节点级标签（如果有用）
#                 node_y=node_y,
#
#                 # 存储额外特征
#                 x_256=image_path_256,
#                 edge_index_256=edge_index_256,
#
#                 x_512=image_path_512,
#                 edge_index_512=edge_index_512
#             )
#
#             data_list.append(data)
#
#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]
#
#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#
#         # 内存化关键步骤：Collate
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])
#         print(f"数据处理完成，保存至 {self.processed_paths[0]}")



def train(train_df, milnet, criterion, optimizer, args, log_path):
    milnet.train()
    total_loss = 0
    atten_max = 0
    atten_min = 0
    atten_mean = 0
    scaler = GradScaler()
    # test_labels = []
    test_predictions = []

    for i, (label, image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type)  in enumerate(train_df):
        torch.cuda.empty_cache()
        bag_label = label.to(device)  # .cuda()
        image_path_256 = image_path_256.to(device)
        image_path_512 = image_path_512.to(device)
        edge_index_256 = edge_index_256.to(device)
        edge_index_512 = edge_index_512.to(device)
        # x = x.to(device)
        # hyperedge_index = hyperedge_index.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)

        torch.cuda.empty_cache()

        optimizer.zero_grad()

        # with autocast():
        results, att20, att10 = milnet(image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type)  # bag_feats_TME)

        ###层级损失(多尺度深度监督loss)
        loss = criterion(results.view(1, -1), bag_label.view(1, -1))

        # torch.cuda.empty_cache()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), max_norm=1.0)
        optimizer.step()

        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ' % (i, len(train_df), loss.item()))

    if args.c_path:
        atten_max = atten_max / len(train_df)
        atten_min = atten_min / len(train_df)
        atten_mean = atten_mean / len(train_df)
        with open(log_path, 'a+') as log_txt:
            log_txt.write('\n atten_max' + str(atten_max))
            log_txt.write('\n atten_min' + str(atten_min))
            log_txt.write('\n atten_mean' + str(atten_mean))
    return total_loss / len(train_df)


def test(test_df, milnet, criterion, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, (label, image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type) in enumerate(test_df):
            torch.cuda.empty_cache()
            bag_label = label.to(device)  # .cuda()
            label = bag_label.cpu().numpy()
            image_path_256 = image_path_256.to(device)
            image_path_512 = image_path_512.to(device)
            edge_index_256 = edge_index_256.to(device)
            edge_index_512 = edge_index_512.to(device)
            # x = x.to(device)
            # hyperedge_index = hyperedge_index.to(device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_type = edge_type.to(device)

            torch.cuda.empty_cache()

            optimizer.zero_grad()

            # if args.model == 'mid_fusion':
            results, att20, att10 = milnet(image_path_256, image_path_512, edge_index_256, edge_index_512, x, edge_index, edge_type)  # bag_feats_TME)

            ###层级损失(多尺度深度监督loss)
            loss = criterion(results.view(1, -1), bag_label.view(1, -1))
            total_loss = total_loss + loss.item()
            # torch.cuda.empty_cache()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend(label)
            max_prediction = results
            bag_prediction = results
            if args.average:  # notice args.average here
                test_predictions.extend([(0.5 * torch.sigmoid(max_prediction) + 0.5 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])

            else:
                test_predictions.extend([(0.0 * torch.sigmoid(max_prediction) + 1.0 * torch.sigmoid(
                    bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)

    # nan_mask = ~np.isnan(test_predictions).any(axis=1)

    # 只保留不包含 NaN 的行
    # test_predictions = test_predictions[nan_mask]
    # test_labels = test_labels[nan_mask]

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n *****************Threshold by optimal*****************')
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
        print(confusion_matrix(test_labels, test_predictions))
        info = confusion_matrix(test_labels, test_predictions)
        with open(log_path, 'a+') as log_txt:
            log_txt.write('\n' + str(info))

    else:
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            print(confusion_matrix(test_labels[:, i], test_predictions[:, i]))
            info = confusion_matrix(test_labels[:, i], test_predictions[:, i])
            with open(log_path, 'a+') as log_txt:
                log_txt.write('\n' + str(info))
    bag_score = 0
    # average acc of all labels
    for i in range(0, len(test_predictions)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
    avg_score = bag_score / len(test_predictions)  # ACC
    cls_report = classification_report(test_labels, test_predictions, digits=4)

    # print(confusion_matrix(test_labels,test_predictions))
    print('\n multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100, sum(auc_value) / len(auc_value) * 100))
    print('\n', cls_report)
    with open(log_path, 'a+') as log_txt:
        log_txt.write('\n  multi-label Accuracy:{:.2f}, AUC:{:.2f}'.format(avg_score * 100,
                                                                           sum(auc_value) / len(auc_value) * 100))
        log_txt.write('\n' + cls_report)

    return total_loss / len(test_predictions), avg_score, auc_value, thresholds_optimal


def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        if sum(label) == 0:
            continue
        prediction = predictions[:, c]
        # print(label, prediction,label.shape, prediction.shape, labels.shape, predictions.shape)
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def main():
    parser = argparse.ArgumentParser(description='Train our model')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1536, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=5e-5, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='SXYSCU_T', type=str, help='Dataset folder name')
    # parser.add_argument('--datasets', default='xiangya2', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='our', type=str, help='model our')
    parser.add_argument('--hidden_channels', type=int, default=300)
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True,
                        help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--input_dim_ct', default=1316, type=str, help='ct feature size')
    parser.add_argument('--agg', type=str, help='which agg')
    parser.add_argument('--c_path', nargs='+',
                        default=None, type=str,
                        help='directory to confounders')  # './datasets_deconf/STAS/train_bag_cls_agnostic_feats_proto_8_transmil.npy'
    # parser.add_argument('--dir', type=str,help='directory to save logs')

    args = parser.parse_args()
    # assert args.model == 'transmil'

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    if args.c_path:
        save_path = os.path.join('deconf', datetime.date.today().strftime("%m%d%Y"),
                                 str(args.dataset) + '_' + str(args.model) + '_' + str(args.agg) + '_c_path')
    else:
        save_path = os.path.join('baseline_EGFR_2', datetime.date.today().strftime("%m%d%Y"),
                                 str(args.dataset) + '_' + str(args.model) + '_' + str(args.agg) + '_fulltune')
    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file, 'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    bags_excel = os.path.join('datasets', args.dataset + '.xlsx')
    bags_path = pd.read_excel(bags_excel, sheet_name="use_label1")


    # 遍历每个折数进行训练
    for fold in range(5):
        with open(log_path, 'a+') as log_txt:
            info = '\n' + 'Fold at: ' + str(fold) + '\n'
            log_txt.write(info)
        print('Fold at: '+ str(fold))

        train_path = (bags_path[bags_path['Fold'] != fold]).drop(columns=['Fold'])
        test_path = (bags_path[bags_path['Fold'] == fold]).drop(columns=['Fold'])
    # from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    #
    # skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,
    #                               random_state=42)  # StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # X = bags_path['path']  # .index
    # y = bags_path['label']  # 假设你有 label 列

    # for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        with open(log_path, 'a+') as log_txt:
            info = '\n' + 'Fold at: ' + str(fold) + '\n'
            log_txt.write(info)
        print('Fold at: ' + str(fold))
        # train_path = (bags_path[bags_path['Fold'] != fold]).drop(columns=['Fold'])
        # test_path = (bags_path[bags_path['Fold'] == fold]).drop(columns=['Fold'])
        # train_path = bags_path.loc[tr_idx].reset_index(drop=True)
        # test_path = bags_path.loc[te_idx].reset_index(drop=True)
        #
        # file_name = f'fold{fold}_train.csv'
        # full_path = os.path.join(save_path, file_name)
        # train_path.to_csv(full_path, index=False)
        #
        # file_name1 = f'fold{fold}_val.csv'
        # full_path1 = os.path.join(save_path, file_name1)
        # test_path.to_csv(full_path1, index=False)
        # print(f"Fold {fold} — train: {len(train_path)}, val: {len(test_path)}")
        # fold += 1
        #
        trainset = BagDataset(train_path, args)
        train_loader = DataLoader(trainset, 1, shuffle=True, num_workers=4)
        testset = BagDataset(test_path, args)
        test_loader = DataLoader(testset, 1, shuffle=False, num_workers=4)

        '''
            model 
            1. set require_grad    
            2. choose model and set the trainable params 
            3. load init
        '''
        import Models.our as mil
        if args.model == 'our':
            # import Models.our as mil
            milnet = mil.fusion_model_graph(num_classes=args.num_classes).to(
                device)  # input_size=args.feats_size, n_classes=args.num_classes,

        elif args.model == 'mid_fusion':
            milnet = mil.Intermediate_fusionmodel(input_size_wsi=args.feats_size, input_dim_ct=args.input_dim_ct,
                                                  num_classes=args.num_classes).to(device)

        for name, _ in milnet.named_parameters():
            print('Training {}'.format(name))
            with open(log_path, 'a+') as log_txt:
                log_txt.write('\n Training {}'.format(name))

        # sanity check begins here
        print('*******sanity check *********')
        for k, v in milnet.named_parameters():
            if v.requires_grad == True:
                print(k)

        # loss, optim, schduler
        if args.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        original_params = []
        confounder_parms = []
        for pname, p in milnet.named_parameters():
            if ('confounder' in pname):
                confounder_parms += [p]
                print('confounders:', pname)
            else:
                original_params += [p]

        print('lood ahead optimizer in our model....')

        from torch.optim import RAdam
        from torch_optimizer import Lookahead  # 需 pip install torch-optimizer
        base_optimizer = RAdam([
            {'params': original_params},
            {'params': confounder_parms, ' weight_decay': 0.0001},
        ],
            lr=0.00001,
            weight_decay=0.00001)
        optimizer = Lookahead(base_optimizer)

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, milnet.parameters()),
        #                              lr=args.lr, betas=(0.5, 0.8),
        #                              weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        best_score = 0

        for epoch in range(1, args.num_epochs):
            start_time = time.time()
            train_loss_bag = train(train_loader, milnet, criterion, optimizer, args, log_path)  # iterate all bags
            print('epoch time:{}'.format(time.time() - start_time))
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_loader, milnet, criterion, optimizer, args,
                                                                      log_path, epoch)

            info = 'Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % (
                epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
                'class-{}>>{}'.format(*k) for k in enumerate(aucs)) + '\n'
            with open(log_path, 'a+') as log_txt:
                log_txt.write(info)
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' %
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join(
                'class-{}>>{}'.format(*k) for k in enumerate(aucs)))
            if args.model != 'transmil':
                scheduler.step()
            current_score = (sum(aucs) + avg_score) / 2
            if current_score >= best_score:
                best_score = current_score
                save_name = os.path.join(save_path, str(run + 1) + f'_{fold}.pth')
                torch.save(milnet.state_dict(), save_name)
                with open(log_path, 'a+') as log_txt:
                    info = 'Best model saved at: ' + save_name + '\n'
                    log_txt.write(info)
                    info = 'Best thresholds ===>>> ' + '|'.join(
                        'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n'
                    log_txt.write(info)
                print('Best model saved at: ' + save_name)
                print(
                    'Best thresholds ===>>> ' + '|'.join(
                        'class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
            if epoch == args.num_epochs - 1:
                save_name = os.path.join(save_path, f'_{fold}_last.pth')
                torch.save(milnet.state_dict(), save_name)
        log_txt.close()


if __name__ == '__main__':
    main()