# from h5py import File
# import numpy as np
# import os.path as osp
# import torch
# from tqdm import tqdm

# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
# from torch.utils.data import Dataset

# from utils.graph import construct_coordinate
# from utils.metrics import StandardDeviationRecord

# import os
# import glob
# import torch
# from torch.utils.data import Dataset
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy
# import numpy as np


# # ----------------------------
# # VTK 読み込みユーティリティ
# # ----------------------------
# def read_vtk(file_path):
#     """VTKファイルを読み込み、点群・速度・圧力を返す"""
#     reader = vtk.vtkUnstructuredGridReader()
#     reader.SetFileName(file_path)
#     reader.Update()
#     data = reader.GetOutput()

#     points = vtk_to_numpy(data.GetPoints().GetData())           # (N, 3)
#     velocity = vtk_to_numpy(data.GetPointData().GetArray("U"))  # (N, 3)
#     pressure = vtk_to_numpy(data.GetPointData().GetArray("p"))  # (N, )
#     return points, velocity, pressure


# class VTKTimeSeriesDataset(Dataset):
#     def __init__(self, root_dir, cases=[0,1,2,3,4], delta_t=10):
#         """
#         root_dir: /home/openfoam/cases/
#         cases: どのケースを使うか (例: [0,1,2,3,4])
#         delta_t: ファイル名の刻み (例: case0_200.vtk → case0_210.vtk → ...)
#         """
#         self.samples = []
#         for case_id in cases:
#             case_path = os.path.join(root_dir, f"case{case_id}", "VTK")
#             vtk_files = sorted(glob.glob(os.path.join(case_path, "case*.[vV][tT][kK]")))

#             for i in range(len(vtk_files)-1):
#                 self.samples.append((vtk_files[i], vtk_files[i+1]))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         file_t, file_t1 = self.samples[idx]
#         points, u_t, p_t = read_vtk(file_t)
#         _, u_t1, p_t1 = read_vtk(file_t1)

#         # 特徴量とラベルを結合 or 個別出力
#         x = np.concatenate([u_t, p_t.reshape(-1, 1)], axis=1)  # [Ux, Uy, p]
#         y = np.concatenate([u_t1, p_t1.reshape(-1, 1)], axis=1)

#         return torch.tensor(points, dtype=torch.float32), torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# dataset = VTKTimeSeriesDataset("/workspace/openfoam/cases/", cases=[0,1])
# print("Total samples:", len(dataset))

# points, x, y = dataset[0]
# print(points.shape, x.shape, y.shape)


# class CylinderFlowDataset:
#     def __init__(self, data_path, sample_factor=1, 
#                  in_t=1, out_t=1, duration_t=10, 
#                  train_batchsize=10, eval_batchsize=10, 
#                  train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, 
#                  normalize=True, **kwargs):        
#         self.load_data(data_path=data_path, sample_factor=sample_factor,
#                        train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio, 
#                        in_t=in_t, out_t=out_t, duration_t=duration_t, 
#                        normalize=normalize)

#         self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
#         self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
#         self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)

#     def load_data(self, data_path, 
#                   train_ratio, valid_ratio, test_ratio, 
#                   sample_factor,
#                   in_t, out_t, duration_t, 
#                   normalize):
#         process_path = data_path.split('.')[0] + '_processed.pt'

#         if osp.exists(process_path):
#             print('Loading processed data from ', process_path)
#             train_data, valid_data, test_data = torch.load(process_path)
#         else:
#             print('Processing data...')
#             raw_data = load_vtk_series(data_path, cases=[0,1,2,3,4])
#             # raw_data = File(data_path, 'r')
#             data_size = len(raw_data.keys())
#             train_idx = int(data_size * train_ratio)
#             valid_idx = int(data_size * (train_ratio + valid_ratio))
#             test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))

#             raw_list = [raw_data['dict_' + str(i)] for i in range(data_size)]
#             del raw_data
            
#             train_data, normalizer = self.pre_process(raw_list[:train_idx], mode='train', sample_factor=sample_factor,
#                                                       in_t=in_t, out_t=out_t, duration_t=duration_t, 
#                                                       normalize=normalize)
#             valid_data = self.pre_process(raw_list[train_idx:valid_idx], mode='valid', sample_factor=sample_factor,
#                                         in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
#                                         normalizer=normalizer)
#             test_data = self.pre_process(raw_list[valid_idx:test_idx], mode='test', sample_factor=sample_factor,
#                                          in_t=in_t, out_t=out_t, duration_t=duration_t, normalize=normalize,
#                                          normalizer=normalizer)
#             print('Saving data...')
#             torch.save((train_data, valid_data, test_data), process_path)
#             print('Data processed and saved to', process_path)
        
#         self.train_dataset = CylinderFlowBase(train_data, mode='train')
#         self.valid_dataset = CylinderFlowBase(valid_data, mode='valid')
#         self.test_dataset = CylinderFlowBase(test_data, mode='test')

#     def pre_process(self, raw_data, sample_factor, in_t, out_t, duration_t, 
#                     mode='train', normalize=False, normalizer=None, **kwargs):
#         all_data = []
#         x_record = StandardDeviationRecord(num_features=3)
#         y_record = StandardDeviationRecord(num_features=3)
        
#         for idx in tqdm(range(len(raw_data))):
#             data = self.single_process(raw_data[idx], in_t, out_t, duration_t, sample_factor, mode)
#             all_data.extend(data)
#             if normalize and normalizer is None:
#                 for d in data:
#                     x_record.update(d.x.numpy(), n=d.x.size(0))
#                     y_record.update(d.y.numpy(), n=d.y.size(0))
        
#         if normalize and normalizer is None:
#             x_mean, x_std = x_record.avg(), x_record.std()
#             y_mean, y_std = y_record.avg(), y_record.std()
#             for data in all_data:
#                 data.x = (data.x - x_mean) / x_std
#                 data.y = (data.y - y_mean) / y_std
#         elif normalize and normalizer is not None:
#             for data in all_data:
#                 data.x = (data.x - normalizer[0]) / normalizer[1]
#                 data.y = (data.y - normalizer[2]) / normalizer[3]
        
#         if mode == 'train':
#             return all_data, [x_record.avg(), x_record.std(), y_record.avg(), y_record.std()]
#         else:
#             return all_data

#     def single_process(self, data, in_t, out_t, duration_t, sample_factor, mode='train'):
#         pressure = torch.from_numpy(np.array(data['pressure'])).unsqueeze(-1).to(torch.float32)
#         velocity = torch.from_numpy(np.array(data['velocity'])).to(torch.float32)
        
#         attrs = torch.cat([pressure, velocity], dim=-1)
#         pos = torch.from_numpy(np.array(data['point'])).to(torch.float32)
#         cell = torch.from_numpy(np.array(data['cell'])).to(torch.long)
        
#         if mode == 'train':
#             x = attrs[:in_t, ::sample_factor, :]
#             y = attrs[in_t:in_t+1, ::sample_factor, :]
#             for i in range(1, duration_t):
#                 x = torch.cat([x, attrs[i:in_t+i, ::sample_factor, :]], dim=0)
#                 y = torch.cat([y, attrs[in_t+i:in_t+i+1, ::sample_factor, :]], dim=0)
#         else:
#             x = attrs[out_t-in_t:out_t, ::sample_factor, :]
#             y = attrs[out_t:out_t+1, ::sample_factor, :]
#             for i in range(1, duration_t):
#                 x = torch.cat([x, attrs[out_t+i-in_t:out_t+i, ::sample_factor, :]], dim=0)
#                 y = torch.cat([y, attrs[out_t+i:out_t+i+1, ::sample_factor, :]], dim=0)
        
#         all_data = [construct_coordinate(Data(x=x[i], y=y[i], pos=pos, cell=cell)) for i in range(x.shape[0])]
        
#         return all_data


# class CylinderFlowBase(Dataset):
#     def __init__(self, data, mode='train', **kwargs):
#         self.mode = mode
#         self.all_data = data
        
#     def __len__(self):
#         return len(self.all_data)
    
#     def __getitem__(self, idx):
#         return self.all_data[idx]

import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import vtk
import random
from vtk.util.numpy_support import vtk_to_numpy

from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from utils.graph import construct_coordinate
from utils.metrics import StandardDeviationRecord


# ----------------------------
# VTK 読み込みユーティリティ
# ----------------------------
def read_vtk(file_path):
    """VTKファイルを読み込み、点群・速度・圧力を返す"""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    data = reader.GetOutput()

    points = vtk_to_numpy(data.GetPoints().GetData())           # (N, 3)
    velocity = vtk_to_numpy(data.GetPointData().GetArray("U"))  # (N, 3)
    pressure = vtk_to_numpy(data.GetPointData().GetArray("p"))  # (N,)
    return points, velocity, pressure


def load_vtk_series(root_dir, cases=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """VTKフォルダを探索し、(points, velocity_t, pressure_t) の時系列を構築"""
    all_samples = []
    for case_id in cases:
        case_path = os.path.join(root_dir, f"case{case_id}", )
        # vtk_files = sorted(glob.glob(os.path.join(case_path, "case*.[vV][tT][kK]")))
        vtk_files = sorted(glob.glob(os.path.join(case_path, "case*_*.vtk")))

        print(f"[Info] case{case_id}: found {len(vtk_files)} files.")

        for i in range(len(vtk_files) - 1):
            all_samples.append((vtk_files[i], vtk_files[i + 1]))
    return all_samples


# ----------------------------
# メイン Dataset クラス
# ----------------------------
class CylinderFlowDataset:
    def __init__(self, data_path,
                 in_t=1, out_t=1, duration_t=10,
                 train_batchsize=10, eval_batchsize=10,
                 train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,
                 normalize=True, **kwargs):

        # 全サンプルペアを取得
        num_cases = 100
        self.samples = load_vtk_series(data_path, cases=list(range(num_cases)))
        total_size = len(self.samples)
        print(f"[Info] Total time-step pairs: {total_size}")

        # ★ ランダムシャッフルを追加（Reなどに偏らないように）
        random.seed(42)  # 再現性を確保
        random.shuffle(self.samples)
        # データ分割
        train_end = int(total_size * train_ratio)
        valid_end = int(total_size * (train_ratio + valid_ratio))

        train_samples = self.samples[:train_end]
        valid_samples = self.samples[train_end:valid_end]
        test_samples = self.samples[valid_end:]

        # 各 split に対応する Dataset を構築
        self.train_dataset = CylinderFlowBase(train_samples, normalize=normalize, mode='train')
        self.valid_dataset = CylinderFlowBase(valid_samples, normalize=normalize, mode='valid')
        self.test_dataset = CylinderFlowBase(test_samples, normalize=normalize, mode='test')

        # DataLoader 構築
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)


class CylinderFlowBase(Dataset):
    def __init__(self, samples, normalize=True, mode='train', **kwargs):
        self.samples = samples
        self.mode = mode
        self.normalize = normalize

        # 標準化のための統計を計算（訓練時のみ）
        if self.mode == 'train' and self.normalize:
            self.x_mean, self.x_std, self.y_mean, self.y_std = self.compute_stats()
        else:
            self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def compute_stats(self):
        """標準化のための平均・分散を計算"""
        x_record = StandardDeviationRecord(num_features=3)
        y_record = StandardDeviationRecord(num_features=3)
        for file_t, file_t1 in tqdm(self.samples, desc="[Compute stats]"):
            _, u_t, p_t = read_vtk(file_t)
            _, u_t1, p_t1 = read_vtk(file_t1)
            x = np.concatenate([u_t[:, :2], p_t.reshape(-1, 1)], axis=1)  # [Ux, Uy, p]
            y = np.concatenate([u_t1[:, :2], p_t1.reshape(-1, 1)], axis=1)

            x_record.update(x, n=x.shape[0])
            y_record.update(y, n=y.shape[0])
        return x_record.avg(), x_record.std(), y_record.avg(), y_record.std()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_t, file_t1 = self.samples[idx]
        points, u_t, p_t = read_vtk(file_t)
        _, u_t1, p_t1 = read_vtk(file_t1)

        # x = np.concatenate([u_t, p_t.reshape(-1, 1)], axis=1)  # [Ux, Uy, Uz, p]
        # y = np.concatenate([u_t1, p_t1.reshape(-1, 1)], axis=1)

        # Z成分を削除（2D流れ用）
        x = np.concatenate([u_t[:, :2], p_t.reshape(-1, 1)], axis=1)  # [Ux, Uy, p]
        y = np.concatenate([u_t1[:, :2], p_t1.reshape(-1, 1)], axis=1)

        # 標準化（訓練時のみ）
        if self.normalize and self.x_mean is not None:
            x = (x - self.x_mean) / self.x_std
            y = (y - self.y_mean) / self.y_std

        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            pos=torch.tensor(points[:, :2], dtype=torch.float32)  # ← XYのみ
        )

        return construct_coordinate(data)
