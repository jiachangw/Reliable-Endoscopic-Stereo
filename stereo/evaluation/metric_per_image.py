import torch
import torch.nn.functional as F

def d1_metric(disp_pred, disp_gt, mask):
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)

    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    d1_per_image = num_errors.float() / num_valid_pixels.float() * 100
    d1_per_image = torch.where(num_valid_pixels > 0, d1_per_image, torch.zeros_like(d1_per_image))

    return d1_per_image


def threshold_metric(disp_pred, disp_gt, mask, threshold):
    E = torch.abs(disp_gt - disp_pred)
    err_mask = E > threshold

    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    bad_per_image = num_errors.float() / num_valid_pixels.float() * 100
    bad_per_image = torch.where(num_valid_pixels > 0, bad_per_image, torch.zeros_like(bad_per_image))

    return bad_per_image


def epe_metric(disp_pred, disp_gt, mask):
    E = torch.abs(disp_gt - disp_pred)
    E_masked = torch.where(mask, E, torch.zeros_like(E))

    E_sum = E_masked.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])
    epe_per_image = E_sum / num_valid_pixels
    epe_per_image = torch.where(num_valid_pixels > 0, epe_per_image, torch.zeros_like(epe_per_image))

    return epe_per_image

def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        vgrid = grid.cuda()
    else:
        vgrid = grid
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    #(input, kernel, stride, padding)
    sigma_x  = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y  = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y , 3, 1, 0) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def gradient(pred):
    D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy

def compute_grad2_smoothness_loss(flo, image, beta):
    """
    Calculate the image-edge-aware second-order smoothness loss
    """

    img_grad_x, img_grad_y = gradient(image)
    weights_x = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_x), 1, keepdim=True))
    weights_y = torch.exp(-10.0 * torch.mean(torch.abs(img_grad_y), 1, keepdim=True))

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    return (torch.mean(beta*weights_x[:,:, :, 1:]*torch.abs(dx2)) + torch.mean(beta*weights_y[:, :, 1:, :]*torch.abs(dy2))) / 2.0

def reconstruction_loss(x, y):
    ssim = torch.mean(SSIM(x, y))
    l1 = torch.mean(torch.abs(x - y))
    return 0.85*ssim + 0.15*l1


def loss_metric(pred, data,smt_weight):
    imgL = data['left']
    imgR = data['right']
    disp_pred = pred['disp_pred']
    B, H, W, _ = imgL.shape
    loss = torch.zeros(B, device=imgL.device)  # 每张图像的总损失，形状为 (B,)
    loss_gamma = 0.9
    disp_preds = pred['disp_preds']
    n_predictions = len(disp_preds)
    assert n_predictions >= 1

    # compute left reconstruction loss
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        recon_imgL = warp(imgR, disp_preds[i])
        i_loss = reconstruction_loss(recon_imgL[:,:,:,48:448],imgL[:,:,:,48:448])

        loss += i_weight * i_loss

    # compute left disparity smoothness loss
    left_smth_loss = compute_grad2_smoothness_loss(disp_pred/20, imgL, 1.0)
    loss += smt_weight * left_smth_loss

    return loss


def correlation_metric(disp_pred, disp_gt, conf_pred, mask):
    """
    计算预测视差与真实视差之间的相关系数

    参数:
        disp_pred: 预测视差图 [B, H, W]
        disp_gt: 真实视差图 [B, H, W]
        mask: 有效像素掩码 [B, H, W]

    返回:
        corr_per_image: 每张图像的相关系数 [B]
    """
    batch_size = disp_pred.shape[0]
    corr_per_image = torch.zeros(batch_size, device=disp_pred.device)

    for i in range(batch_size):
        # 获取当前图像的预测、真实视差和掩码
        pred = disp_pred[i]
        gt = disp_gt[i]
        conf = conf_pred[i]
        img_mask = mask[i]

        # 只考虑有效像素
        valid_pred = pred[img_mask]
        valid_gt = gt[img_mask]
        valid_conf = conf[img_mask]

        # 如果没有有效像素，相关系数为0
        if valid_pred.numel() == 0:
            corr_per_image[i] = 0.0
            continue

        # 计算均值
        mean_pred = valid_conf.mean()
        mean_gt = torch.abs(valid_pred - valid_gt).mean()

        # 计算协方差
        cov = torch.sum((valid_pred - mean_pred) * (valid_gt - mean_gt))

        # 计算标准差
        std_pred = torch.sqrt(torch.sum((valid_pred - mean_pred) ** 2))
        std_gt = torch.sqrt(torch.sum((valid_gt - mean_gt) ** 2))

        # 计算相关系数
        if std_pred > 0 and std_gt > 0:
            corr = cov / (std_pred * std_gt)
            # 确保相关系数在[-1, 1]范围内
            corr = torch.clamp(corr, -1.0, 1.0)
        else:
            corr = 0.0

        corr_per_image[i] = corr

    return corr_per_image
