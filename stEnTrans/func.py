import random
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
def img2expr(imputed_img, gene_ids, integral_coords, position_info):
    [imputed_x, imputed_y, not_in_tissue_coords] = position_info

    imputed_img = imputed_img.numpy()
    if type(not_in_tissue_coords) == np.ndarray:
        not_in_tissue_coords = [list(val) for val in not_in_tissue_coords]

    integral_barcodes = integral_coords.index
    imputed_counts = pd.DataFrame(np.zeros((imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords),
                                            imputed_img.shape[0])), columns=gene_ids)
    imputed_coords = pd.DataFrame(np.zeros((imputed_img.shape[1] * imputed_img.shape[2] - len(not_in_tissue_coords),
                                            2)), columns=['array_row', 'array_col'])
    imputed_barcodes = [None] * len(imputed_counts)
    integral_coords = [list(i.astype(np.float32)) for i in np.array(integral_coords)]

    flag = 0
    for i in range(imputed_img.shape[1]):
        for j in range(imputed_img.shape[2]):

            spot_coords = [imputed_x[i, j], imputed_y[i, j]]
            if spot_coords in not_in_tissue_coords:
                continue

            if spot_coords in integral_coords:
                imputed_barcodes[flag] = integral_barcodes[integral_coords.index(spot_coords)]
            else:
                if int(imputed_x[i, j]) == imputed_x[i, j]:
                    x_id = str(int(imputed_x[i, j]))
                else:
                    x_id = str(imputed_x[i, j])
                if int(imputed_y[i, j]) == imputed_y[i, j]:
                    y_id = str(int(imputed_y[i, j]))
                else:
                    y_id = str(imputed_y[i, j])

                imputed_barcodes[flag] = x_id + "x" + y_id

            imputed_counts.iloc[flag , :] = imputed_img[:, i, j]

            imputed_coords.iloc[flag , :] = spot_coords
            flag = flag + 1

    imputed_counts.index = imputed_barcodes
    imputed_coords.index = imputed_barcodes

    return imputed_counts, imputed_coords

def get_not_in_tissue_coords(coords, img_xy):
    img_x, img_y = img_xy
    coords = coords.astype(img_x.dtype)
    coords = [list(val) for val in np.array(coords)]
    not_in_tissue_coords = []
    not_in_tissue_index = []
    for i in range(img_x.shape[0]):
        for j in range(img_x.shape[1]):
            ij_coord = [img_x[i, j], img_y[i, j]]
            if ij_coord not in coords:
                not_in_tissue_coords.append(ij_coord)
                not_in_tissue_index.append([int(i), int(j)])
    return not_in_tissue_coords, np.array(not_in_tissue_index)

def get_ST_position_info(integral_coords):
    # 保留原点或者原点之间的点（这个两点之间也可以是斜边）
    integral_coords = np.array(integral_coords)
    delta_x = 1
    delta_y = 1

    # Get coordinates of imputed spots.
    imputed_x, imputed_y = np.mgrid[min(integral_coords[:, 0]):max(integral_coords[:, 0]) + delta_x:delta_x / 2,
                           min(integral_coords[:, 1]):max(integral_coords[:, 1]) + delta_y:delta_y / 2]

    # Count the number of adjacent spots from original data.
    integral_coords = integral_coords.astype(np.float64)
    imputed_barcodes = [str(val[0]) + "x" + str(val[1]) for val in
                        np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).T]
    imputed_coords = pd.DataFrame(np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).astype(np.float64).T,
                                  columns=['row', 'col'], index=imputed_barcodes)
    neighbor_matrix = pd.DataFrame(np.zeros((imputed_coords.shape[0], imputed_coords.shape[0]), dtype=np.int8),
                                   columns=imputed_barcodes, index=imputed_barcodes)

    row1 = imputed_coords[imputed_coords["row"] == min(imputed_coords["row"])].sort_values("col")
    for i in range(len(row1) - 1):
        if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
            neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
            neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
    for row in list(np.array(imputed_x).T[0])[:-1]:
        row0 = imputed_coords[imputed_coords["row"] == row].sort_values("col")
        row1 = imputed_coords[imputed_coords["row"] == row + delta_x / 2].sort_values("col")
        for i in range(len(row1) - 1):
            if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
                neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
                neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
        for i in range(len(row0)):
            for j in range(len(row1)):
                flag = 0
                if abs(imputed_coords.loc[row0.index[i], "col"] - imputed_coords.loc[
                    row1.index[j], "col"]) == delta_y / 2:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = -1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = -1
                    flag += 1
                if imputed_coords.loc[row0.index[i], "col"] == imputed_coords.loc[row1.index[j], "col"]:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = 1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = 1
                    flag += 1
                if flag >= 3:
                    continue

    # Get not-in-tissue coordinates.
    neighbor_matrix = neighbor_matrix.loc[:, [str(val[0]) + "x" + str(val[1]) for val in integral_coords]]
    not_in_tissue_coords = []
    for i in range(len(imputed_coords)):
        if imputed_coords.index[i] in neighbor_matrix.columns:
            continue
        i_row = neighbor_matrix.iloc[i]
        if sum(i_row != 0) < 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))
        if sum(i_row != 0) == 2 and sum(i_row == -1) == 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))

    position_info = [imputed_x, imputed_y, not_in_tissue_coords]

    return position_info

def get_10X_position_info(integral_coords):

    integral_coords = np.array(integral_coords)
    delta_x = 1
    delta_y = 2

    # Get coordinates of imputed spots.
    x_min = min(integral_coords[:, 0]) - min(integral_coords[:, 0]) % 2  # start with even row
    y_min = min(integral_coords[:, 1]) - min(integral_coords[:, 1]) % 2  # start with even col

    y = list(np.arange(y_min, max(integral_coords[:, 1]) + delta_y, delta_y))  # 0，2，...，max+1
    imputed_x, imputed_y = np.mgrid[x_min:max(integral_coords[:, 0]) + delta_x:delta_x / 2,
                           y_min:y[-1] + delta_y:delta_y / 2]

    for i in range(1, imputed_y.shape[0], 2):
        imputed_y[i] -= delta_y / 4
    for i in range(2, imputed_y.shape[0], 4):
        imputed_y[i:i + 2] += delta_y / 2

    # Count number of adjacent original spots
    integral_coords = integral_coords.astype(np.float32)
    imputed_barcodes = [str(val[0]) + "x" + str(val[1]) for val in
                        np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).T]
    imputed_coords = pd.DataFrame(np.vstack((imputed_x.reshape(-1), imputed_y.reshape(-1))).astype(np.float32).T,
                                  columns=['row', 'col'], index=imputed_barcodes)
    neighbor_matrix = pd.DataFrame(np.zeros((imputed_coords.shape[0], imputed_coords.shape[0]), dtype=np.int8),
                                   columns=imputed_barcodes, index=imputed_barcodes)

    row1 = imputed_coords[imputed_coords["row"] == min(imputed_coords["row"])].sort_values("col")
    for i in range(len(row1) - 1):
        if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
            neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
            neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
    for row in list(np.array(imputed_x).T[0])[:-1]:
        row0 = imputed_coords[imputed_coords["row"] == row].sort_values("col")
        row1 = imputed_coords[imputed_coords["row"] == row + delta_x / 2].sort_values("col")
        for i in range(len(row1) - 1):
            if row1["col"][i + 1] - row1["col"][i] == delta_y / 2:
                neighbor_matrix.loc[row1.index[i], row1.index[i + 1]] = 1
                neighbor_matrix.loc[row1.index[i + 1], row1.index[i]] = 1
        for i in range(len(row0)):
            for j in range(len(row1)):
                flag = 0
                if abs(imputed_coords.loc[row0.index[i], "col"] - imputed_coords.loc[
                    row1.index[j], "col"]) == delta_y / 4:
                    neighbor_matrix.loc[row0.index[i], row1.index[j]] = 1
                    neighbor_matrix.loc[row1.index[j], row0.index[i]] = 1
                    flag += 1
                if flag >= 2:
                    continue

    # Get not-in-tissue coordinates.
    neighbor_matrix = neighbor_matrix.loc[:, [str(val[0]) + "x" + str(val[1]) for val in integral_coords]]
    not_in_tissue_coords = []
    for i in range(len(imputed_coords)):
        if imputed_coords.index[i] in neighbor_matrix.columns:
            continue
        if sum(neighbor_matrix.iloc[i]) < 2:
            not_in_tissue_coords.append(list(imputed_coords.iloc[i]))

    position_info = [imputed_x, imputed_y, not_in_tissue_coords]

    return position_info

# down-sample
def get_train_data(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    # 坐标从0开始
    train_coords[:, 0] = train_coords[:, 0] - min(train_coords[:, 0])
    train_coords[:, 1] = train_coords[:, 1] - min(train_coords[:, 1])
    delta_x = 1
    delta_y = 1

    # The number of rows and columns must be a multiple of 2 (note starting from 0). This is where the boundaries of the coordinate system are constructed
    if not max(train_coords[:, 0]) % 2:
        x_index = (train_coords[:, 0] < max(train_coords[:, 0]))  # mask
        train_coords = train_coords[x_index]
        train_counts = train_counts[x_index]
    if not max(train_coords[:, 1]) % 2:
        y_index = (train_coords[:, 1] < max(train_coords[:, 1]))
        train_coords = train_coords[y_index]
        train_counts = train_counts[y_index]

    lr_spot_index = []  # Record which spots are retained
    lr_x, lr_y = np.mgrid[0:max(train_coords[:, 0]):2 * delta_x,
                 0:max(train_coords[:, 1]):2 * delta_y]
    # The subsampled coordinate system is typed into the list
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]  # 列表 [[x,y], ***, [x,y]]

    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)

    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]  # How many genes there are, how many profiles there are
    for i in range(train_counts.shape[1]):
        # Fill the gene value into the profile
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")

        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0  # Not spot is set to 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]) + delta_y:delta_y]

    hr_not_in_tissue_coords, hr_not_in_tissue_xy = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))
    hr_not_in_tissue_x = hr_not_in_tissue_xy.T[0]
    hr_not_in_tissue_y = hr_not_in_tissue_xy.T[1]

    train_hr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        hr = griddata(train_coords, train_counts[:, i], (hr_x, hr_y), method="nearest")
        hr[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0
        train_hr[i] = hr
    train_hr = np.array(train_hr)

    in_tissue_matrix = np.ones_like(hr_x)
    in_tissue_matrix[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0

    return train_lr, train_hr, in_tissue_matrix

def get10Xtrainset(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    delta_x = 1
    delta_y = 2

    # Get low-resolution images by down-sample.
    x_min = min(train_coords[:, 0]) + min(train_coords[:, 0]) % 2  # start with even row
    y_min = min(train_coords[:, 1]) + min(train_coords[:, 1]) % 2  # start with even col
    lr_x, lr_y = np.mgrid[x_min:max(train_coords[:, 0]) + delta_x:2 * delta_x,
                 y_min:max(train_coords[:, 1]):2 * delta_y]

    # Determine which of the sampled coordinates are spots
    lr_spot_index = []
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[x_min:max(lr_coords[:, 0]) + delta_x * 1.5:delta_x,
                 y_min:max(lr_coords[:, 1]) + delta_y * 1.5:delta_y]
    # Let the adjacent row peak value, so that all spots can be taken
    for i in range(1, hr_y.shape[0], 2):
        hr_y[i] = hr_y[i] - delta_y / 2

    # Get the coordinates of missing values on HR images.
    hr_not_in_tissue_coords, hr_not_in_tissue_xy = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))
    hr_not_in_tissue_x = hr_not_in_tissue_xy.T[0]
    hr_not_in_tissue_y = hr_not_in_tissue_xy.T[1]

    train_hr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        hr = griddata(train_coords, train_counts[:, i], (hr_x, hr_y), method="nearest")
        hr[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0
        train_hr[i] = hr
    train_hr = np.array(train_hr)

    in_tissue_matrix = np.ones_like(hr_x)
    in_tissue_matrix[hr_not_in_tissue_x, hr_not_in_tissue_y] = 0

    return train_lr, train_hr, in_tissue_matrix

def getSTtestset(test_counts, test_coords):
    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)
    delta_x = 1
    delta_y = 1

    # 真实坐标值，可以与索引不同，这里是相同的，在下采样时是可以不同的
    # 是整个平面的所有坐标值，不是spot的
    test_input_x, test_input_y = np.mgrid[min(test_coords[:, 0]):max(test_coords[:, 0]) + delta_x:delta_x,
                                 min(test_coords[:, 1]):max(test_coords[:, 1]) + delta_y:delta_y]

    # Get the coordinates of missing values on input images.
    not_in_tissue_coords, not_in_tissue_xy = get_not_in_tissue_coords(test_coords, (test_input_x, test_input_y))
    not_in_tissue_x = not_in_tissue_xy.T[0]  # 索引 0.1.2...
    not_in_tissue_y = not_in_tissue_xy.T[1]

    # Get testing set.
    test_set = [None] * test_counts.shape[1]
    for i in range(test_counts.shape[1]):
        test_data = griddata(test_coords, test_counts[:, i], (test_input_x, test_input_y), method="nearest")
        test_data[not_in_tissue_x, not_in_tissue_y] = 0
        test_set[i] = test_data
    test_set = np.array(test_set)

    return test_set

def get10Xtestset(test_counts, test_coords):

    test_counts = np.array(test_counts)
    test_coords = np.array(test_coords)
    delta_x = 1
    delta_y = 2

    # Get coordinates of spots on testing images.
    x_min = min(test_coords[:, 0]) - min(test_coords[:, 0]) % 2  # start with even row
    y_min = min(test_coords[:, 1]) - min(test_coords[:, 1]) % 2  # start with even col

    test_input_x, test_input_y = np.mgrid[x_min:max(test_coords[:, 0]) + delta_x:delta_x,
                                 y_min:max(test_coords[:, 1]) + delta_y:delta_y]
    # 相邻的列错峰取值
    for i in range(1, test_input_y.shape[0], 2):
        test_input_y[i] = test_input_y[i] + delta_y / 2

    not_in_tissue_coords, not_in_tissue_xy = get_not_in_tissue_coords(test_coords, (test_input_x, test_input_y))
    not_in_tissue_x = not_in_tissue_xy.T[0]
    not_in_tissue_y = not_in_tissue_xy.T[1]

    test_set = [None] * test_counts.shape[1]
    for i in range(test_counts.shape[1]):
        test_data = griddata(test_coords, test_counts[:, i], (test_input_x, test_input_y), method="nearest")
        test_data[not_in_tissue_x, not_in_tissue_y] = 0
        test_set[i] = test_data
    test_set = np.array(test_set)

    return test_set

# This function is called when the box diagram of ST is drawn.
def get_down_ST(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)

    train_coords[:, 0] = train_coords[:, 0] - min(train_coords[:, 0])
    train_coords[:, 1] = train_coords[:, 1] - min(train_coords[:, 1])
    delta_x = 1
    delta_y = 1

    lr_spot_index = []
    lr_x, lr_y = np.mgrid[0:max(train_coords[:, 0])+1:2 * delta_x,
                 0:max(train_coords[:, 1])+1:2 * delta_y]

    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]  # 列表 [[x,y], ***, [x,y]]

    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)

    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]) + delta_y:delta_y]

    hr_not_in_tissue_coords, _ = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))

    coords_index = [None] * len(lr_coords)
    for i in range(len(lr_coords)):
        coords_index[i] = str(lr_coords[i, 0]) + 'x' + str(lr_coords[i, 1])
    lr_coords = pd.DataFrame(lr_coords, index=coords_index)

    return train_lr, lr_counts, lr_coords, hr_not_in_tissue_coords, train_counts

# This function is called when the box diagram of 10X is drawn.
def get_down_10x(train_counts, train_coords):
    train_counts = np.array(train_counts)
    train_coords = np.array(train_coords)
    train_coords[:, 0] = train_coords[:, 0] - min(train_coords[:, 0])
    train_coords[:, 1] = train_coords[:, 1] - min(train_coords[:, 1])
    delta_x = 1
    delta_y = 2

    lr_x, lr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:2 * delta_x,
                 0:max(train_coords[:, 1]):2 * delta_y]

    # 判断下采样后的坐标哪些是spot
    lr_spot_index = []
    lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
    for i in range(train_coords.shape[0]):
        if list(train_coords[i]) in lr_xy:
            lr_spot_index.append(i)
    lr_counts = train_counts[lr_spot_index]
    lr_coords = train_coords[lr_spot_index]

    # 索引 0.1.2...
    lr_not_in_tissue_coords, lr_not_in_tissue_xy = get_not_in_tissue_coords(lr_coords, (lr_x, lr_y))
    lr_not_in_tissue_x = lr_not_in_tissue_xy.T[0]
    lr_not_in_tissue_y = lr_not_in_tissue_xy.T[1]

    train_lr = [None] * train_counts.shape[1]
    for i in range(train_counts.shape[1]):
        lr = griddata(lr_coords, lr_counts[:, i], (lr_x, lr_y), method="nearest")
        lr[lr_not_in_tissue_x, lr_not_in_tissue_y] = 0
        train_lr[i] = lr
    train_lr = np.array(train_lr)

    hr_x, hr_y = np.mgrid[0:max(train_coords[:, 0]) + delta_x:delta_x,
                 0:max(train_coords[:, 1]):delta_y]

    # 让相邻的行错峰取值，可以取到全部点
    for i in range(1, hr_y.shape[0], 2):
        hr_y[i] = hr_y[i] + delta_y / 2

    hr_not_in_tissue_coords, _ = get_not_in_tissue_coords(train_coords, (hr_x, hr_y))

    coords_index = [None] * len(lr_coords)
    for i in range(len(lr_coords)):
        coords_index[i] = str(lr_coords[i, 0]) + 'x' + str(lr_coords[i, 1])
    lr_coords = pd.DataFrame(lr_coords, index=coords_index)

    return train_lr, lr_counts, lr_coords, hr_not_in_tissue_coords, train_counts

# mini-batch
def data_iter(data_lr, data_hr, batch_size):
    num_gene = data_lr.shape[0]
    indices = list(range(num_gene))
    random.shuffle(indices)
    for i in range(0, num_gene, batch_size):
        yield data_lr[indices[i:min(i+batch_size, num_gene)]], data_hr[indices[i:min(i+batch_size, num_gene)]]

# Loss
def criterion(pre_hr, hr, in_tissue_matrix):
    # pre: (b, 4, h_lr, w_lr)
    # hr: (b, 1, h_hr, w_hr)
    # li：# (batch, h_hr, w_hr)
    pre_hr = pre_hr[:,:,0:int(hr.shape[2]/2),0:int(hr.shape[3]/2)]
    l1 = torch.reshape(torch.square((pre_hr[:, 0, :, :] - hr[:, 0, 0::2, 0::2]) * in_tissue_matrix[0::2, 0::2]), [-1])
    l2 = torch.reshape(torch.square((pre_hr[:, 1, :, :] - hr[:, 0, 0::2, 1::2]) * in_tissue_matrix[0::2, 1::2]), [-1])
    l3 = torch.reshape(torch.square((pre_hr[:, 2, :, :] - hr[:, 0, 1::2, 0::2]) * in_tissue_matrix[1::2, 0::2]), [-1])
    l4 = torch.reshape(torch.square((pre_hr[:, 3, :, :] - hr[:, 0, 1::2, 1::2]) * in_tissue_matrix[1::2, 1::2]), [-1])
    l = torch.cat([l1, l2, l3, l4], dim=0)

    return torch.mean(l)

# Merging the 4-channel profile to get the HR data.
def get_test_data(pre_hr, is_pad=False, train_lr_h=None, train_lr_w=None):
    if is_pad:
        pre_hr = pre_hr[:,:,0:train_lr_h,0:train_lr_w]
    b, _, h, w = pre_hr.shape
    hr = torch.zeros(size=(b, h*2, w*2))
    hr[:, 0::2, 0::2] = pre_hr[:, 0, :, :]
    hr[:, 0::2, 1::2] = pre_hr[:, 1, :, :]
    hr[:, 1::2, 0::2] = pre_hr[:, 2, :, :]
    hr[:, 1::2, 1::2] = pre_hr[:, 3, :, :]
    return hr

def data_pad(train_lr, patch_size):
    train_lr = torch.Tensor(train_lr)
    # gene_num, h, w
    pad_b = patch_size - train_lr.shape[1] % patch_size
    pad_r = patch_size - train_lr.shape[2] % patch_size
    train_lr = torch.nn.functional.pad(train_lr, (0, pad_r, 0, pad_b))
    return train_lr.detach().numpy()

def plot_geness(adata1, adata2, show_genes, size=(10, 8), cmap=None):
    titles = show_genes
    genes_index1 = [list(adata1.var_names).index(gene) for gene in show_genes]
    genes_index2 = [list(adata2.var_names).index(gene) for gene in show_genes]
    plt.figure(figsize=size)
    flag = 0
    for j in range(2):  # 两层
        if j == 0:
            adata = adata1
            genes_index = genes_index1
        else:
            k = flag - 3
            adata = adata2
            genes_index = genes_index2
        for i in range(3):

            plt.subplot(2, 3, flag + 1)  # 2行3列的子图
            if j == 0:
                # x, y, 值， marker：点的形状， s:点的大小
                plt.scatter(adata.obs['array_row'], adata.obs['array_col'], c=np.array(adata.X)[:, genes_index[flag]],
                            marker='s', s=42, vmin=0, cmap=cmap)
                plt.title(titles[flag])
            else:
                # x, y, 值， marker：点的形状， s:点的大小
                plt.scatter(adata.obs['array_row'], adata.obs['array_col'], c=np.array(adata.X)[:, genes_index[k]],
                            marker='s', s=8, vmin=0, cmap=cmap)
                plt.title(titles[k])
                k += 1
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')  # 关闭坐标轴
            flag += 1
    plt.show()

def plot_gene(adata, show_gene, size, point_size):
    plt.figure(figsize=size)
    plt.scatter(adata.obs['array_col'], adata.obs['array_row'],
               c=np.array(adata.X)[:,list(adata.var_names).index(show_gene)],
               marker='.', s=point_size)

    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    # plt.axis('off')
    plt.show()