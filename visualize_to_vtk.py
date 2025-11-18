import os
import glob
import torch
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from torch_geometric.data import Data
from AMG.utils.graph import construct_coordinate
from AMG.models.grapher.grapher import Grapher
from AMG.datasets.cylinder_flow import read_vtk


# =====================================================
# 設定
# =====================================================
MODEL_PATH = "./logs/CylinderFlow/11_16/Grapher_14_49_11/best_model.pth"  # 毎回変更! モデルの重み
CASE_DIR = "/workspace/data/VTK132/case201"                           # CFD結果のあるフォルダ
OUTPUT_DIR = "/workspace/AMG/test_cons/CASE201_DATA150_batch8"                             # 出力フォルダ
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 対象となるVTKファイル一覧
VTK_FILES = sorted(glob.glob(os.path.join(CASE_DIR, "case201*.vtk")))


# =====================================================
# モデル読み込み
# =====================================================
model = Grapher(
    input_features=3,
    output_features=3,
    pos_dim=2,
    feature_width=128,
    num_layers=3,
    num_heads=8,
    global_ratio=0.1,
    global_k=2,
    local_nodes=1024,
    local_ratio=0.25,
    local_k=3,
)
model = model.to("cuda")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
print(f"[Loaded model] {MODEL_PATH}")


# =====================================================
# VTK保存関数
# =====================================================
def save_vtk_with_mesh(template_vtk_path, vel_pred, p_pred, out_path):
    """
    元のVTKのメッシュ構造を保持したまま、予測結果をPointDataに追加して保存。
    """
    # --- ① VTK読み込み ---
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(template_vtk_path)
    reader.Update()
    grid = reader.GetOutput()

    N = grid.GetNumberOfPoints()
    vel_pred = np.asarray(vel_pred)
    p_pred = np.asarray(p_pred).reshape(-1)

    # --- ② サイズチェック ---
    if vel_pred.shape[0] != N or p_pred.shape[0] != N:
        raise ValueError(f"予測サイズ不一致: grid_points={N}, vel={vel_pred.shape}, p={p_pred.shape}")

    # --- ③ z成分補間 ---
    if vel_pred.shape[1] == 2:
        vel_pred = np.hstack([vel_pred, np.zeros((N, 1))])

    # --- ④ 予測値を追加 ---
    vel_vtk = numpy_to_vtk(vel_pred, deep=True)
    vel_vtk.SetName("U_pred")
    grid.GetPointData().AddArray(vel_vtk)

    p_vtk = numpy_to_vtk(p_pred, deep=True)
    p_vtk.SetName("p_pred")
    grid.GetPointData().AddArray(p_vtk)

    # --- ⑤ メッシュ構造を保持して出力 (.vtu推奨) ---
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(out_path.replace(".vtk", ".vtu"))
    writer.SetInputData(grid)
    writer.SetDataModeToAscii()  # ← 確実に読める形式
    writer.Write()


# =====================================================
# 時系列推論ループ
# =====================================================
print(f"[Info] Found {len(VTK_FILES)} files.")
for i, vtk_file in enumerate(VTK_FILES):
    points, velocity, pressure = read_vtk(vtk_file)
    x = np.concatenate([velocity[:, :2], pressure.reshape(-1, 1)], axis=1)
    pos = points[:, :2]

    graph = construct_coordinate(Data(
        x=torch.tensor(x, dtype=torch.float32).cuda(),
        y=torch.zeros_like(torch.tensor(x, dtype=torch.float32)).cuda(),  # ダミー解
        pos=torch.tensor(pos, dtype=torch.float32).cuda()
    ))
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long).cuda()

    with torch.no_grad():
        y_pred = model(graph).cpu().numpy()

    vel_pred = y_pred[:, :2]
    p_pred = y_pred[:, 2]

    out_name = os.path.basename(vtk_file).replace(".vtk", "_pred.vtk")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    # save_vtk(pos, vel_pred, p_pred, out_path)
    # 元メッシュを引き継いで保存！
    save_vtk_with_mesh(vtk_file, vel_pred, p_pred, out_path)

    print(f"[{i+1}/{len(VTK_FILES)}] Saved → {out_path}")

print(f"\n✅ 全{len(VTK_FILES)}時刻分の予測VTKを {OUTPUT_DIR} に保存しました。")
print("ParaViewで開けば、時系列アニメーションとして再生できます。")



#export PYTHONPATH=/workspace:$PYTHONPATH   これをbashでいれたら通るようになる