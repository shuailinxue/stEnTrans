from stEnTrans import *
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载数据
# (279, 15666)
counts = pd.read_csv('data/mel1_rep1/counts.csv', index_col=0)  # index_col=0:第一列为索引值
coords = pd.read_csv('data/mel1_rep1/coords.csv', index_col=0)
adata = ad.AnnData(X=counts.values, obs=coords, var=pd.DataFrame(index=counts.columns.values))  # var只有行索引

# 未改变原adata,添加要插值的点  C矩阵
integral_coords = adata.obs[['array_row','array_col']]
integral_coords.loc["5x10",:]=[5,10]  # 添加三个在组织中但未测量的点
integral_coords.loc["10x14",:]=[10,14]
integral_coords.loc["15x22",:]=[15,22]

position_info = get_ST_position_info(integral_coords)

# 添加指控指标到原adata中 如var添加n_cells_by_counts(列标签)， 表示这个基因出现在了几个点中  基因名字是行索引，n_cells是列标签
sc.pp.calculate_qc_metrics(adata, inplace=True)
# 以下两个函数除了有筛选功能，还会分别添加一列，用于表示每个点有几个基因、每个基因存在于几个点中
# 改变了原adata
sc.pp.filter_cells(adata, min_genes=20)  # 筛除低于20个基因的点
sc.pp.filter_genes(adata, min_cells=10)  # 筛除低于在10个点中存在的基因

train_adata=adata[:,adata.var["n_cells_by_counts"]>len(adata.obs.index)*0.1]

train_counts=np.array(train_adata.X)  # ndarray对象
train_coords=train_adata.obs[['array_row','array_col']]  # DataFrame对象 经过添加指标，obs已经不再只有行列了

# （基因数，行，列）
train_lr,train_hr,in_tissue_matrix = get_train_data(train_counts, train_coords)
in_tissue_matrix = torch.Tensor(in_tissue_matrix).to(device)

test_counts = np.array(adata.X)
test_coords = adata.obs[['array_row', 'array_col']]
test_3D_data = getSTtestset(test_counts, test_coords)

imputed_adata = stEnTrans(adata, test_3D_data, integral_coords, position_info, train_lr, train_hr, in_tissue_matrix, patch_size=3, num_heads=4,epoch=50)

show_genes=['CSPG4', 'DLL3', 'CD37']
plot_geness(adata, imputed_adata, show_genes, size=(10,8))