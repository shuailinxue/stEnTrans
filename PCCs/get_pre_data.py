from baselines.NEDI import NEDI_run as NEDI
from stEnTrans.network import VisionTransformer as model
from stEnTrans import *
from baselines import DIST
from main import get_R

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data_DIST(process_data, load, truth_counts):
    simu_3D, impute_position, simu_coords, adata_var_names = process_data
    net = DIST.Model().to(device)
    net.load_state_dict(torch.load(load))
    pre_3D = []
    for i in range(0, simu_3D.shape[0], 128):
        with torch.no_grad():
            data = simu_3D[i:min((i + 128), simu_3D.shape[0]), :, :, :]
            data = data.to(device)
            pre_data = net(data)
            pre_data = get_test_data(pre_data)  # （b, 2h, 2w）
            pre_3D.append(pre_data)
    pre_3D = torch.cat(pre_3D, dim=0)  # （all, 2h, 2w）
    # 下采样时，可能原始数据是奇数，这样上采样后，会多出一行/列
    #pre_3D = pre_3D[:, :, 0:-1]
    imputed_counts, _ = img2expr(pre_3D, adata_var_names, simu_coords, impute_position)
    R, _ = get_R(truth_counts, imputed_counts.values)
    return R

def get_data_interpolation(process_data, method, truth_counts):
    simu_counts, impute_position, simu_coords, adata_var_names = process_data
    interpolated_x, interpolated_y, _ = impute_position

    interpolated_set = [None] * simu_counts.shape[1]  # 基因数
    for i in range(simu_counts.shape[1]):
        interpolated_data = griddata(simu_coords, simu_counts[:, i], (interpolated_x, interpolated_y), method=method,
                                     fill_value=0)
        #interpolated_data = interpolated_data[:, 0:-1]
        interpolated_set[i] = interpolated_data
    interpolated_set = torch.Tensor(np.array(interpolated_set))

    imputed_counts, _ = img2expr(interpolated_set, adata_var_names, simu_coords, impute_position)
    R, _ = get_R(truth_counts, imputed_counts.values)
    return R

def get_data_NEDI(process_data, truth_counts):
    simu_3D, impute_position, simu_coords, adata_var_names = process_data
    simu_3D = simu_3D.reshape((simu_3D.shape[0], simu_3D.shape[2], simu_3D.shape[3])).numpy()
    interpolated_set = [None] * simu_3D.shape[0]
    # 这个size的宽高位置反过来
    size = (simu_3D.shape[2] * 2, simu_3D.shape[1] * 2)
    for i in range(simu_3D.shape[0]):
        interpolated_data = NEDI(simu_3D[i], new_size=size)
        ##############
        #interpolated_data = interpolated_data[:, 0:-1]
        interpolated_set[i] = interpolated_data
    interpolated_set = torch.Tensor(interpolated_set)

    imputed_counts, _ = img2expr(interpolated_set, adata_var_names, simu_coords, impute_position)
    R, _ = get_R(truth_counts, imputed_counts.values)
    return R

# stEnTrans
def get_data_stEnTrans(process_data, load, patch_size, truth_counts, num_heads):
    # simu_3D:b,1,h,w 还未pad
    simu_3D, impute_position, simu_coords, adata_var_names = process_data
    simu_3D_h, simu_3D_w = simu_3D.shape[2], simu_3D.shape[3]  # 原始数据的行/列
    simu_3D = torch.Tensor(simu_3D.reshape((simu_3D.shape[0], simu_3D.shape[2], simu_3D.shape[3])))
    simu_3D = data_pad(simu_3D, patch_size=patch_size)
    simu_3D = torch.Tensor(simu_3D.reshape((simu_3D.shape[0], 1, simu_3D.shape[1], simu_3D.shape[2])))
    net = model(patch_size=patch_size, embed_dim=4*patch_size*patch_size, num_heads=num_heads).to(device)
    net.load_state_dict(torch.load(load))
    pre_3D = []
    for i in range(0, simu_3D.shape[0], 128):
        with torch.no_grad():
            data = simu_3D[i:min((i + 128), simu_3D.shape[0]), :, :, :]
            data = data.to(device)
            pre_data = net(data)
            # 把padding的部分mask掉
            pre_data = get_test_data(pre_data, is_pad=True, train_lr_h=simu_3D_h, train_lr_w=simu_3D_w)  # （b, 2h, 2w）
            pre_3D.append(pre_data)
    pre_3D = torch.cat(pre_3D, dim=0)  # （all, 2h, 2w）
    # 如果原始数据的行列都是偶数个，则去掉这一行
    #pre_3D = pre_3D[:, 0:-1, 0:-1]
    imputed_counts, _ = img2expr(pre_3D, adata_var_names, simu_coords, impute_position)
    R, _ = get_R(truth_counts, imputed_counts.values)
    return R