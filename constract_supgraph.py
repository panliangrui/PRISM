from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict
from itertools import chain
import dgl
from scipy.stats import pearsonr
import h5py
import torch


# ========= 1) 配置路径 =========
# SVS_DIR = Path(r"X:\mianyi_prediction\xiangya_svs")

GIGAPATH_DIR = Path(r"X:\projects\mianyi\all_features\xiangya2\gigapath")
CONCH_DIR    = Path(r"X:\projects\mianyi\all_features\xiangya2\conch_v15")
TME_DIR      = Path(r"X:\projects\mianyi\all_features\xiangya2\tme")

PREFIX_LEN = 7


# ========= 2) 工具函数：列文件、取前缀、建立 prefix->文件路径映射 =========
def build_prefix_map(folder: Path, exts=None, prefix_len: int = 7):
    """
    返回:
      prefixes: set[str]
      pmap: dict[str, list[Path]]  # prefix -> 所有可能文件（可能同前缀多个文件）
    """
    if exts is None:
        # 不过滤
        files = [p for p in folder.iterdir() if p.is_file()]
    else:
        exts = {e.lower() for e in exts}
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

    pmap = defaultdict(list)
    for p in files:
        prefix = p.stem#[:prefix_len]  # 用 stem 避免扩展名影响
        pmap[prefix].append(p)

    prefixes = set(pmap.keys())
    return prefixes, pmap


def pick_one_file(pmap: dict, prefix: str, prefer_ext: str = None) -> Path:
    """
    如果同一个 prefix 对应多个文件，按 prefer_ext 优先，否则取第一个。
    """
    candidates = pmap.get(prefix, [])
    if not candidates:
        raise FileNotFoundError(f"No file for prefix={prefix}")

    if prefer_ext is not None:
        prefer_ext = prefer_ext.lower()
        for c in candidates:
            if c.suffix.lower() == prefer_ext:
                return c

    return candidates[0]


# ========= 3) 读取 H5 / PT 内容 =========
def read_h5_all_datasets(h5_path: Path, max_elems: int = 5):
    """
    读取 H5 中所有 dataset（不递归 group 的话也可以扩展）。
    返回 dict: key -> numpy array / 或 shape 信息
    为避免超大数组直接塞内存，这里默认读取全量；如果你很大，可以改成只读shape或抽样。
    """
    out = {}
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 这里读全量；如太大可改成 obj[0:...] 或只存shape
                data = obj[()]
                out[name] = {
                    "shape": getattr(data, "shape", None),
                    "dtype": str(getattr(data, "dtype", type(data))),
                    "data": data  # 如果你不想把数据放内存，可删掉这一行
                }
        f.visititems(visit)
    return out


def read_pt(pt_path: Path, map_location="cpu"):
    """
    读取 torch 保存的 .pt
    """
    obj = torch.load(pt_path, weights_only=False)
    return obj





from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nmslib
import h5py
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv


# -----------------------------
# 0) Label constants (from your meta_info.labels_text)
# -----------------------------
# meta_info.labels_text: {-100:'unlabeled', 1:'tumor', 2:'stromal', 3:'sTILs', 4:'blood', 5:'macrophage', 6:'dead', 7:'other'}
IGNORE_LABEL = -100

TUMOR = 1
STROMA = 2
STILS = 3
BLOOD = 4
MACRO = 5
DEAD = 6
OTHER = 7

NUM_CLASSES = 8  # keep slot 0 for bg (even if unused)


# -----------------------------
# 1) HNSW KNN wrapper (nmslib)
# -----------------------------
class Hnsw:
    """
    KNN model (HNSW) cloned from Patch-GCN style.
    """
    def __init__(self, space='l2', index_params=None, query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X: np.ndarray):
        if self.index_params is None:
            self.index_params = {'M': 16, 'post': 0, 'efConstruction': 400}
        if self.query_params is None:
            self.query_params = {'ef': 90}

        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(self.index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(self.query_params)

        self.index_ = index
        return self

    def query(self, vector: np.ndarray, topn: int):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

    def query_batch(self, Xq: np.ndarray, topn: int) -> List[np.ndarray]:
        """
        Much faster for many queries.
        Return: list of indices arrays
        """
        res = self.index_.knnQueryBatch(Xq, k=topn, num_threads=0)
        return [r[0] for r in res]  # only indices


# -----------------------------
# 2) Parsing utilities
# -----------------------------
def cell_centers_from_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [N,4] = (x1, y1, x2, y2) slide-level coords
    return centers: [N,2] = (cx, cy)
    """
    boxes = boxes.float()
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return torch.stack([cx, cy], dim=1)


def filter_cells(cell_stats: dict, score_thr: float = 0.3, ignore_label: int = IGNORE_LABEL):
    """
    Remove unlabeled (-100) and low-confidence cells.
    Returns centers, labels, scores.
    """
    boxes = cell_stats["boxes"]
    labels = cell_stats["labels"]
    scores = cell_stats["scores"].float()  # float16 -> float32

    keep = (labels != ignore_label) & (scores >= score_thr)
    boxes = boxes[keep]
    labels = labels[keep].long()
    scores = scores[keep]

    centers = cell_centers_from_boxes(boxes)
    return centers, labels, scores


def build_subgraph_masks(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Define TME semantic groups (subgraphs).
    """
    masks = {
        "tumor": (labels == TUMOR),
        "immune": (labels == STILS) | (labels == MACRO),
        "stroma": (labels == STROMA),
        "vascular": (labels == BLOOD),
        "necrosis": (labels == DEAD),
        "other": (labels == OTHER),
    }
    masks["tme_all"] = torch.ones_like(labels, dtype=torch.bool)
    return masks


# -----------------------------
# 3) Node features (baseline)
# -----------------------------
def labels_to_onehot(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    One-hot (size=8). Your labels are in {1..7}, so index 0 will be mostly 0.
    """
    out = torch.zeros((labels.numel(), num_classes), dtype=torch.float32)
    valid = (labels >= 0) & (labels < num_classes)
    idx = torch.arange(labels.numel())[valid]
    out[idx, labels[valid]] = 1.0
    return out


def build_node_features(centers: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """
    Feature = [onehot(8), score, normalized_xy] => dim = 8 + 1 + 2 = 11
    """
    onehot = labels_to_onehot(labels, NUM_CLASSES)
    score_col = scores.float().unsqueeze(1)

    xy = centers.float()
    if xy.numel() > 0:
        mean = xy.mean(dim=0, keepdim=True)
        std = xy.std(dim=0, keepdim=True).clamp_min(1.0)
        xy_norm = (xy - mean) / std
    else:
        xy_norm = xy

    return torch.cat([onehot, score_col, xy_norm], dim=1)


# -----------------------------
# 4) HNSW-based hyperedge builder
# -----------------------------
def hyperedges_by_hnsw_knn(
    centers: torch.Tensor,
    centers_ids: torch.Tensor,   # A: hyperedge centers (global ids)
    member_ids: torch.Tensor,    # B: candidate members (global ids)
    k: int = 16,
    include_center: bool = True,
    min_size: int = 2,
    max_size: Optional[int] = None,
    hnsw_space: str = "l2",
    index_params: Optional[dict] = None,
    query_params: Optional[dict] = None,
) -> torch.Tensor:
    """
    For each a in A, build hyperedge with kNN_B(a) (and optionally include a).
    Return hyperedge_index: [2, num_incidence] with GLOBAL node ids.
    """
    if centers_ids.numel() == 0 or member_ids.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    pts = centers.detach().cpu().numpy().astype(np.float32)
    pts_A = pts[centers_ids.detach().cpu().numpy()]
    pts_B = pts[member_ids.detach().cpu().numpy()]

    k_eff = min(int(k), int(pts_B.shape[0]))
    if k_eff <= 0:
        return torch.empty((2, 0), dtype=torch.long)

    knn_model = Hnsw(space=hnsw_space, index_params=index_params, query_params=query_params, print_progress=False)
    knn_model.fit(pts_B)

    neigh_list = knn_model.query_batch(pts_A, topn=k_eff)  # indices in B

    node_ids: List[torch.Tensor] = []
    hedge_ids: List[torch.Tensor] = []
    e = 0

    for i, inds_in_B in enumerate(neigh_list):
        if len(inds_in_B) == 0:
            continue

        members = member_ids[torch.from_numpy(np.asarray(inds_in_B)).long()]  # global ids

        if include_center:
            members = torch.unique(torch.cat([members, centers_ids[i:i+1]], dim=0))
        else:
            members = torch.unique(members)

        if max_size is not None and members.numel() > max_size:
            members = members[:max_size]
        if members.numel() < min_size:
            continue

        node_ids.append(members)
        hedge_ids.append(torch.full((members.numel(),), e, dtype=torch.long))
        e += 1

    if e == 0:
        return torch.empty((2, 0), dtype=torch.long)

    node_idx = torch.cat(node_ids, dim=0)
    hedge_idx = torch.cat(hedge_ids, dim=0)
    return torch.stack([node_idx, hedge_idx], dim=0)


def concat_hyperedges_with_offset(hyperedges: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Merge multiple hyperedge_index into one, offsetting hyperedge ids.
    Also return hyperedge_type vector and type_names list.
    """
    keys = list(hyperedges.keys())
    merged_list = []
    type_list = []
    type_names = []
    offset = 0

    for t, k in enumerate(keys):
        he = hyperedges[k]
        if he.numel() == 0:
            continue

        node_idx = he[0]
        hedge_idx = he[1] + offset
        merged_list.append(torch.stack([node_idx, hedge_idx], dim=0))

        num_hedges = int(he[1].max().item()) + 1 if he[1].numel() else 0
        type_list.append(torch.full((num_hedges,), t, dtype=torch.long))
        type_names.append(k)

        offset += num_hedges

    if len(merged_list) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long), []

    merged_hyperedge_index = torch.cat(merged_list, dim=1)
    hyperedge_type = torch.cat(type_list, dim=0)
    return merged_hyperedge_index, hyperedge_type, type_names


# -----------------------------
# 5) Config and Data builder
# -----------------------------
@dataclass
class HNSWHyperConfig:
    score_thr: float = 0.3

    # kNN sizes
    k_small: int = 16
    k_large: int = 32

    # caps
    max_size_small: int = 32
    max_size_large: int = 64

    include_center: bool = True

    # HNSW tuning (optional)
    index_params: Optional[dict] = None  # e.g. {'M': 16, 'post': 0, 'efConstruction': 400}
    query_params: Optional[dict] = None  # e.g. {'ef': 90}
    hnsw_space: str = "l2"


def build_tme_hyperedges_hnsw(
    centers: torch.Tensor,
    labels: torch.Tensor,
    cfg: HNSWHyperConfig,
) -> Dict[str, torch.Tensor]:
    """
    Multi-relation hyperedges:
      - immune niche
      - tumor-immune interaction
      - stroma niche
      - perivascular
      - necrosis environment
    """
    masks = build_subgraph_masks(labels)
    all_ids = torch.arange(labels.numel(), dtype=torch.long)

    tumor_ids = all_ids[masks["tumor"]]
    immune_ids = all_ids[masks["immune"]]
    stroma_ids = all_ids[masks["stroma"]]
    blood_ids = all_ids[masks["vascular"]]
    dead_ids = all_ids[masks["necrosis"]]

    common = dict(
        include_center=cfg.include_center,
        hnsw_space=cfg.hnsw_space,
        index_params=cfg.index_params,
        query_params=cfg.query_params,
    )

    hyperedges: Dict[str, torch.Tensor] = {}

    hyperedges["immune_niche_hnsw"] = hyperedges_by_hnsw_knn(
        centers,
        centers_ids=immune_ids,
        member_ids=immune_ids,
        k=cfg.k_small,
        min_size=2,
        max_size=cfg.max_size_small,
        **common
    )

    hyperedges["tumor_immune_hnsw"] = hyperedges_by_hnsw_knn(
        centers,
        centers_ids=tumor_ids,
        member_ids=immune_ids,
        k=cfg.k_small,
        min_size=2,
        max_size=cfg.max_size_small,
        **common
    )

    hyperedges["stroma_niche_hnsw"] = hyperedges_by_hnsw_knn(
        centers,
        centers_ids=stroma_ids,
        member_ids=stroma_ids,
        k=cfg.k_small,
        min_size=2,
        max_size=cfg.max_size_small,
        **common
    )

    hyperedges["perivascular_hnsw"] = hyperedges_by_hnsw_knn(
        centers,
        centers_ids=blood_ids,
        member_ids=all_ids,
        k=cfg.k_large,
        min_size=3,
        max_size=cfg.max_size_large,
        **common
    )

    hyperedges["necrosis_env_hnsw"] = hyperedges_by_hnsw_knn(
        centers,
        centers_ids=dead_ids,
        member_ids=all_ids,
        k=cfg.k_large,
        min_size=3,
        max_size=cfg.max_size_large,
        **common
    )

    return hyperedges


def build_pyg_data_from_pred_hnsw(pred_obj: dict, cfg: HNSWHyperConfig = HNSWHyperConfig()) -> Data:
    centers, labels, scores = filter_cells(pred_obj["cell_stats"], score_thr=cfg.score_thr)
    x = build_node_features(centers, labels, scores)

    hyperedges = build_tme_hyperedges_hnsw(centers, labels, cfg)
    merged_hyperedge_index, hyperedge_type, type_names = concat_hyperedges_with_offset(hyperedges)

    data = Data(
        x=x,
        pos=centers.float(),
        y=labels.long(),
        hyperedge_index=merged_hyperedge_index.long(),
        hyperedge_type=hyperedge_type.long(),
    )
    data.hyperedge_type_names = type_names
    return data



# -----------------------------
# 7) Load pred_obj from .pt and run example
# -----------------------------
def load_pred_obj(pt_path: str) -> dict:
    obj = torch.load(pt_path,weights_only=False)
    if not isinstance(obj, dict) or "cell_stats" not in obj:
        raise ValueError("Loaded .pt is not a pred_obj dict with key 'cell_stats'.")
    return obj

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()  # 确保张量在 CPU 上
    return tensor

def save_data_to_h5(data, image_path_512_fea, edge_index_image_512, image_path_256_fea, edge_index_image_256, data_256, data_512, filename):
    """
    Save PyTorch Geometric Data object to an HDF5 file.

    Args:
    - data (Data): PyTorch Geometric Data object.
    - filename (str): Path to the HDF5 file where data will be saved.
    """
    with h5py.File(filename, 'w') as f:
        # Save node features (x), labels (y), and positions (pos)
        # f.create_dataset('x_img_256', data=to_numpy(data_256['features']["data"]), compression='gzip', compression_opts=9)
        f.create_dataset('x_img_256_coord', data=to_numpy(data_256['coords']["data"]), compression='gzip', compression_opts=9)
        f.create_dataset('image_path_256_fea', data=to_numpy(image_path_256_fea), compression='gzip',
                         compression_opts=9)
        f.create_dataset('edge_index_image_256', data=to_numpy(edge_index_image_256), compression='gzip',
                         compression_opts=9)

        # f.create_dataset('x_img_512', data=to_numpy(data_512['features']["data"]), compression='gzip', compression_opts=9)
        f.create_dataset('x_img_512_coord', data=to_numpy(data_512['coords']["data"]), compression='gzip', compression_opts=9)
        f.create_dataset('image_path_512_fea', data=to_numpy(image_path_512_fea), compression='gzip',
                         compression_opts=9)
        f.create_dataset('edge_index_image_512', data=to_numpy(edge_index_image_512), compression='gzip',
                         compression_opts=9)


        f.create_dataset('x', data=data.x.cpu().numpy())  # Node features (N x F)
        # f.create_dataset('y', data=data.y.cpu().numpy())  # Node labels (N,)
        # f.create_dataset('pos', data=data.pos.cpu().numpy())  # Node positions (N x 2)

        # Save hyperedges
        f.create_dataset('hyperedge_index', data=data.hyperedge_index.cpu().numpy())  # Hyperedge index (2 x M)
        # f.create_dataset('hyperedge_type', data=data.hyperedge_type.cpu().numpy())  # Hyperedge types (E,)

        # Save hyperedge type names (this will be a simple string list)
        # f.create_dataset('hyperedge_type_names',
        #                  data=np.array(data.hyperedge_type_names, dtype='S'))  # Hyperedge type names (list)

        print(f"Data successfully saved to {filename}")

def example_forward_from_pt(pt_path: str):
    pred_obj = load_pred_obj(pt_path)

    cfg = HNSWHyperConfig(
        score_thr=0.4,
        k_small=10,
        k_large=20,
        max_size_small=16,
        max_size_large=32,
        include_center=True,
        # Optional tuning:
        index_params={'M': 12, 'post': 0, 'efConstruction': 200},
        query_params={'ef': 50},
        hnsw_space="l2",
    )

    data = build_pyg_data_from_pred_hnsw(pred_obj, cfg)
    # save_data_to_h5(data, 'X:\\mianyi_prediction\\features\\1495822-8-HE.h5')
    # model = TMEHyperNet(in_dim=data.x.size(1), hidden=128, out_dim=NUM_CLASSES, dropout=0.2)
    # model.eval()

    # with torch.no_grad():
    #     logits = model(data)

    print("Data:", data)
    print("Relation names:", getattr(data, "hyperedge_type_names", None))
    print("hyperedge_index:", tuple(data.hyperedge_index.shape))
    print("hyperedge_type:", tuple(data.hyperedge_type.shape))
    # print("logits:", tuple(logits.shape))
    return data


knn_model = Hnsw(space='l2')


# ========= 4) 主逻辑 =========
def main():
    # A: svs 文件名(前7)
    # A, Amap = build_prefix_map(SVS_DIR, exts=None, prefix_len=PREFIX_LEN)

    # B: gigapath features 文件名(前7)  —— 你的描述是“features_gigapath中所有文件名”，
    #    但后面读取的是 .h5，所以这里建议过滤 .h5，避免同名干扰
    B, Bmap = build_prefix_map(GIGAPATH_DIR, exts={".h5"}, prefix_len=PREFIX_LEN)

    # C: conch_v15 features 文件名(前7)，读取 .h5
    C, Cmap = build_prefix_map(CONCH_DIR, exts={".h5"}, prefix_len=PREFIX_LEN)

    # D: tme 目录下 .pt 文件名(前7)
    D, Dmap = build_prefix_map(TME_DIR, exts={".pt"}, prefix_len=PREFIX_LEN)

    # 共同匹配（A,B,C,D都包含）
    common = sorted(B & C & D)
    # print(f"SVS(A): {len(A)}  GigaPath(B): {len(B)}  CONCH(C): {len(C)}  TME(D): {len(D)}")
    print(f"Common(A∩B∩C∩D): {len(common)}")

    # 对每个共同前缀读取内容
    results = {}
    base_dir = r"X:\projects\mianyi\all_features\xiangya2\graph_low"
    for sid in common:
        pt_path = os.path.join(base_dir, f"{sid}.h5")
        print(sid)
        # 如果已经存在，就跳过
        if os.path.exists(pt_path):
            print(f"已存在 {pt_path}，跳过")
            continue
        if sid=='1643015-1':
            continue


        gigapath_h5 = pick_one_file(Bmap, sid, prefer_ext=".h5")
        conch_h5    = pick_one_file(Cmap, sid, prefer_ext=".h5")
        tme_pt      = pick_one_file(Dmap, sid, prefer_ext=".pt")

        # 读取内容（注意：H5可能很大）
        giga_data = read_h5_all_datasets(gigapath_h5)
        conch_data = read_h5_all_datasets(conch_h5)
        ###构图
        image_path_256_fea = giga_data['features']["data"]
        image_path_512_fea = conch_data['features']["data"]
        node_image_path_256_fea, node_image_path_512_fea = giga_data['features']["data"], conch_data['coords']["data"]
        # node_image_path_256_fea, node_image_path_512_fea, node_image_path_1024_fea = image_path_256_fea, image_path_512_fea, image_path_1024_fea#get_node(image_path_256_fea, image_path_512_fea, image_path_1024_fea)
        # node_image_path_256_fea = torch.Tensor(np.stack(node_image_path_256_fea))
        # node_image_path_512_fea = torch.Tensor(np.stack(node_image_path_512_fea))
        # node_image_path_1024_fea = torch.Tensor(np.stack(node_image_path_1024_fea))
        # 构建256的边
        n_patches = len(image_path_256_fea)  # .shape[0]
        # 使用列表推导式提取所有值
        all_values = [value for value in image_path_256_fea]

        # 使用vstack函数将所有值堆叠在一起
        image_path_256_fea = np.vstack(all_values)
        radius = 9
        # Construct graph using spatial coordinates
        knn_model.fit(image_path_256_fea)

        a = np.repeat(range(n_patches), radius - 1)
        b = np.fromiter(
            chain(
                *[knn_model.query(image_path_256_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
            ), dtype=int
        )
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Create edge types
        edge_type = []
        edge_sim = []
        for (idx_a, idx_b) in zip(a, b):
            metric = pearsonr
            corr = metric(image_path_256_fea[idx_a], image_path_256_fea[idx_b])[0]
            edge_type.append(1 if corr > 0 else 0)
            edge_sim.append(corr)

        # Construct dgl heterogeneous graph
        graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        image_path_256_fea = torch.tensor(image_path_256_fea, device='cpu').float()
        # 获取图中的边
        edges1 = graph.edges()
        edge_index_image_256 = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)

        # 构建512的边
        n_patches = len(image_path_512_fea)  # .shape[0]
        # 使用列表推导式提取所有值
        all_values = [value for value in image_path_512_fea]

        # 使用vstack函数将所有值堆叠在一起
        image_path_512_fea = np.vstack(all_values)
        radius = 9
        # Construct graph using spatial coordinates
        knn_model.fit(image_path_512_fea)

        a = np.repeat(range(n_patches), radius - 1)
        b = np.fromiter(
            chain(
                *[knn_model.query(image_path_512_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
            ), dtype=int
        )
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Create edge types
        edge_type = []
        edge_sim = []
        for (idx_a, idx_b) in zip(a, b):
            metric = pearsonr
            corr = metric(image_path_512_fea[idx_a], image_path_512_fea[idx_b])[0]
            edge_type.append(1 if corr > 0 else 0)
            edge_sim.append(corr)

        # Construct dgl heterogeneous graph
        graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
        image_path_512_fea = torch.tensor(image_path_512_fea, device='cpu').float()
        # 获取图中的边
        edges1 = graph.edges()
        edge_index_image_512 = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)

        # tme_data = read_pt(tme_pt)
        tme_graph = example_forward_from_pt(tme_pt)
        GRAPH_DIR = Path(r"X:\projects\mianyi\all_features\xiangya2\graph_low")
        file_path = GRAPH_DIR / f"{sid}.h5"
        save_data_to_h5(tme_graph, image_path_512_fea, edge_index_image_512, image_path_256_fea, edge_index_image_256, giga_data, conch_data,filename=file_path)


        print(f"[OK] {sid} | giga={gigapath_h5.name} conch={conch_h5.name} tme={tme_pt.name}")

    # 你可以在这里把 results 保存成 pickle / 或仅保存路径清单
    # 示例：只保存共同样本ID列表
    # import json
    # with open("matched_ids.json", "w", encoding="utf-8") as f:
    #     json.dump(common, f, ensure_ascii=False, indent=2)

    return results


if __name__ == "__main__":
    results = main()
