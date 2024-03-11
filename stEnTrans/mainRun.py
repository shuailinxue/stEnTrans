import anndata as ad
from .network import VisionTransformer as Model
from .func import *
def stEnTrans(adata, test_3D_data, integral_coords, position_info, train_lr, train_hr, in_tissue_matrix, patch_size, batch_size=512, num_heads=8, epoch=500, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Model(patch_size=patch_size, embed_dim=patch_size*patch_size*4, num_heads=num_heads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.6), eps=1e-6)

    train_lr = data_pad(train_lr, patch_size)
    # （基因数，1, 行，列）
    train_lr = torch.Tensor(train_lr.reshape((int(train_lr.shape[0]), 1, int(train_lr.shape[1]), int(train_lr.shape[2]))))
    train_hr = torch.Tensor(train_hr.reshape((int(train_hr.shape[0]), 1, int(train_hr.shape[1]), int(train_hr.shape[2]))))
    for epoch in range(epoch):
        loss_running = 0
        idx = 0
        for b_id, data in enumerate(data_iter(train_lr, train_hr, 512), 0):
            idx += 1
            lr, hr = data
            lr, hr = lr.to(device), hr.to(device)
            pre_hr = net(lr)
            loss = criterion(pre_hr, hr, in_tissue_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_running += loss.item()
        print(f'epoch:{epoch + 1}, loss:{round(loss_running / idx, 3)}')

    b, h, w = test_3D_data.shape
    test_3D_data = data_pad(test_3D_data, patch_size=patch_size)
    test_3D_data = torch.Tensor(test_3D_data.reshape((b, 1, test_3D_data.shape[1], test_3D_data.shape[2])))
    pre_3D_data = []
    for i in range(0, test_3D_data.shape[0], 512):
        with torch.no_grad():
            data = test_3D_data[i:min((i + 512), test_3D_data.shape[0]), :, :, :]
            data = data.to(device)
            pre_data = net(data)
            pre_data = get_test_data(pre_data, is_pad=True, train_lr_h=h, train_lr_w=w)  # （b, 2h, 2w）
            pre_3D_data.append(pre_data)
    pre_3D_data = torch.cat(pre_3D_data, dim=0)
    # integral_coords的作用是保留原点的名字（行索引）
    imputed_counts, imputed_coords = img2expr(pre_3D_data, adata.var_names, integral_coords, position_info)
    imputed_adata = ad.AnnData(X=imputed_counts, obs=imputed_coords)
    adata.X[adata.X < 0.5] = 0
    return imputed_adata