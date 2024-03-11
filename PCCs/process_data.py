from stEnTrans import *
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
def get_placenta_data():
    adata = sc.read_h5ad('C:\\Users\DELL\PycharmProjects\stEnTrans\data\placenta_adata.h5ad')
    adata.var_names = adata.var['gene_name']  # 让var的索引值变成基因名字，而不是0，1，2
    coords = pd.DataFrame(adata.obsm['spatial'])
    mask = coords.notnull().iloc[:, 0].values
    coords.fillna(-1, inplace=True)
    adata.obsm['spatial'] = coords.values

    for i in range(len(adata.obsm['spatial'])):
        if adata.obsm['spatial'][i, 0] == -1:
            continue
        adata.obsm['spatial'][i, 0] = round(adata.obsm['spatial'][i, 0] / 218)
        adata.obsm['spatial'][i, 1] = round(adata.obsm['spatial'][i, 1] / 170)

    coords = pd.DataFrame(adata.obsm['spatial']).sort_values([0, 1], ascending=[True, False])
    for i in range(1, len(adata.obsm['spatial'])):
        if coords.iloc[i, 1] == -1:
            continue
        if coords.iloc[i, 1] >= coords.iloc[i - 1, 1] and coords.iloc[i, 0] == coords.iloc[i - 1, 0]:
            if coords.iloc[i - 1, 1] - 1 < 0:
                index = coords.index[i]
                mask[index] = False
                coords.iloc[i, 1] = 0
            else:
                coords.iloc[i, 1] = coords.iloc[i - 1, 1] - 1
    coords = coords.sort_index()
    adata.obsm['spatial'] = coords.values

    truth_counts = adata.raw.X[mask]
    truth_coords = adata.obsm['spatial'][mask]

    #将spot排序，方便最终比较相关性
    con = np.concatenate((truth_coords, truth_counts), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values

    return get_useful_data(truth_counts, truth_coords, adata.var_names)

def get_mouse_data():
    adata = sc.read_h5ad('C:\\Users\DELL\PycharmProjects\stEnTrans\data\Mouse_brain.h5ad')
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.filter_cells(adata, min_genes=100)
    for i in range(len(adata.obsm['spatial'])):
        adata.obsm['spatial'][i, 0] = round((adata.obsm['spatial'][i, 0] + 40.464753887937))  # 0-203
        adata.obsm['spatial'][i, 1] = round((adata.obsm['spatial'][i, 1] + 429.185464140973))  # 0-256 重合点移动后变为258

    coords = pd.DataFrame(adata.obsm['spatial']).sort_values([0, 1], ascending=[True, True])
    for i in range(1, len(adata.obsm['spatial'])):
        if coords.iloc[i, 0] == coords.iloc[i - 1, 0] and coords.iloc[i, 1] <= coords.iloc[i - 1, 1]:
            coords.iloc[i, 1] = coords.iloc[i - 1, 1] + 1

    adata.obsm['spatial'] = coords.sort_index().values  # 至此，坐标的值处理完毕

    # 将spot排序，方便最终比较相关性
    con = np.concatenate((adata.obsm['spatial'], adata.layers['count'].todense()), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values
    return get_useful_data(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_useful_data(truth_counts, truth_coords, adata_var_names):
    # DIST需要的是：simu_3D， not_in_tissue_coords， simu_coords，adata.var_names
    # not_in_tissue_coords: 真实值的M矩阵

    if not max(truth_coords[:, 0]) % 2:
        x_index = (truth_coords[:, 0] < max(truth_coords[:, 0]))
        truth_coords = truth_coords[x_index]
        truth_counts = truth_counts[x_index]
    if not max(truth_coords[:, 1]) % 2:
        y_index = (truth_coords[:, 1] < max(truth_coords[:, 1]))
        truth_coords = truth_coords[y_index]
        truth_counts = truth_counts[y_index]

    simu_3D, simu_counts, simu_coords, not_in_tissue_coords, truth_counts = get_down_ST(truth_counts, truth_coords)
    # imputed_x, imputed_y = np.mgrid[min(truth_coords[:, 0]):max(truth_coords[:, 0]) + 1:1,
    #                        min(truth_coords[:, 1]):max(truth_coords[:, 1]) + 1:1]
    imputed_x, imputed_y = np.mgrid[0:max(truth_coords[:, 0])-min(truth_coords[:, 0]) + 1:1,
                           0:max(truth_coords[:, 1])-min(truth_coords[:, 1]) + 1:1]

    impute_position = [imputed_x, imputed_y, not_in_tissue_coords]
    b, h, w = simu_3D.shape
    simu_3D = torch.Tensor(simu_3D.reshape((b, 1, h, w)))
    return truth_counts, [simu_3D, impute_position, simu_coords, adata_var_names], [simu_counts, impute_position,
                                                                                    simu_coords, adata_var_names]

def get_mel1_rep1():
    counts = pd.read_csv('C:\\Users\DELL\PycharmProjects\stEnTrans\data\mel1_rep1\counts.csv', index_col=0)  # index_col=0:第一列为索引值
    coords = pd.read_csv('C:\\Users\DELL\PycharmProjects\stEnTrans\data\mel1_rep1\coords.csv', index_col=0)
    adata = ad.AnnData(X=counts.values, obs=coords, var=pd.DataFrame(index=counts.columns.values))  # var只有行索引

    # 添加指控指标到原adata中 如var添加n_cells_by_counts(列标签)， 表示这个基因出现在了几个点中  基因名字是行索引，n_cells是列标签
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    # 以下两个函数除了有筛选功能，还会分别添加一列，用于表示每个点有几个基因、每个基因存在于几个点中
    # 改变了原adata
    sc.pp.filter_cells(adata, min_genes=20)  # 筛除低于20个基因的点
    sc.pp.filter_genes(adata, min_cells=10)  # 筛除低于在10个点中存在的基因

    train_adata = adata[:, adata.var["n_cells_by_counts"] > len(adata.obs.index) * 0.1]
    truth_counts = np.array(train_adata.X)  # ndarray对象
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    return get_useful_data(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_10X_Mouse():
    train_adata = sc.read_visium("C:\\Users\DELL\PycharmProjects\stEnTrans\data\Mouse_Brain_Sagittal_Posterior")
    train_adata.var_names_make_unique()
    # train_adata.X:稀疏矩阵
    sc.pp.calculate_qc_metrics(train_adata, inplace=True)
    sc.pp.filter_cells(train_adata, min_genes=200)
    sc.pp.filter_genes(train_adata, min_cells=10)

    train_adata = train_adata[:, train_adata.var["n_cells_by_counts"] > len(train_adata.obs.index) * 0.1]
    truth_counts = np.array(train_adata.X.todense())
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    # 将spot排序，方便最终比较相关性
    con = np.concatenate((truth_coords, truth_counts), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values

    return get_useful_data_10X(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_10X_Human():
    train_adata = sc.read_visium("C:\\Users\DELL\PycharmProjects\stEnTrans\data\Human_Invasive_Ductal_Carcinoma")
    train_adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(train_adata, inplace=True)
    sc.pp.filter_cells(train_adata, min_genes=5)
    sc.pp.filter_genes(train_adata, min_cells=10)

    train_adata = train_adata[:, train_adata.var["n_cells_by_counts"] > len(train_adata.obs.index) * 0.1]
    truth_counts = np.array(train_adata.X.todense())
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    # 将spot排序，方便最终比较相关性
    con = np.concatenate((truth_coords, truth_counts), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values

    return get_useful_data_10X(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_10X_HBC():
    train_adata = sc.read_visium("C:\\Users\DELL\PycharmProjects\stEnTrans\data\Human Breast Cancer")
    train_adata.var_names_make_unique()
    sc.pp.calculate_qc_metrics(train_adata, inplace=True)
    sc.pp.filter_cells(train_adata, min_genes=200)
    sc.pp.filter_genes(train_adata, min_cells=10)

    train_adata = train_adata[:, train_adata.var["n_cells_by_counts"] > len(train_adata.obs.index) * 0.1]
    truth_counts = np.array(train_adata.X.todense())
    truth_coords = train_adata.obs[['array_row', 'array_col']].values

    # 将spot排序，方便最终比较相关性
    con = np.concatenate((truth_coords, truth_counts), axis=1)
    con_df = pd.DataFrame(con).sort_values([0, 1], ascending=[True, True])
    truth_counts = con_df.iloc[:, 2:].values
    truth_coords = con_df.iloc[:, 0:2].values

    return get_useful_data_10X(truth_counts, truth_coords, [i for i in range(truth_counts.shape[1])])

def get_useful_data_10X(truth_counts, truth_coords, adata_var_names):

    if not max(truth_coords[:, 0]) % 2:
        x_index = (truth_coords[:, 0] < max(truth_coords[:, 0]))
        truth_coords = truth_coords[x_index]
        truth_counts = truth_counts[x_index]
    if not max(truth_coords[:, 1]) % 4:
        y_index = (truth_coords[:, 1] < max(truth_coords[:, 1]))
        truth_coords = truth_coords[y_index]
        truth_counts = truth_counts[y_index]

    # DIST需要的是：simu_3D， not_in_tissue_coords， simu_coords，adata.var_names
    # not_in_tissue_coords: 真实值的M矩阵
    simu_3D, simu_counts, simu_coords, not_in_tissue_coords, truth_counts = get_down_10x(truth_counts, truth_coords)
    # imputed_x, imputed_y = np.mgrid[min(truth_coords[:, 0]):max(truth_coords[:, 0]) + 1:1,
    #                        min(truth_coords[:, 1]):max(truth_coords[:, 1]) + 1:1]
    imputed_x, imputed_y = np.mgrid[0:max(truth_coords[:, 0])-min(truth_coords[:, 0]) + 1:1,
                           0:max(truth_coords[:, 1])-min(truth_coords[:, 1]):2]
    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] = imputed_y[i] + 1

    impute_position = [imputed_x, imputed_y, not_in_tissue_coords]
    b, h, w = simu_3D.shape
    simu_3D = torch.Tensor(simu_3D.reshape((b, 1, h, w)))
    return truth_counts, [simu_3D, impute_position, simu_coords, adata_var_names], [simu_counts, impute_position,
                                                                                    simu_coords, adata_var_names]
