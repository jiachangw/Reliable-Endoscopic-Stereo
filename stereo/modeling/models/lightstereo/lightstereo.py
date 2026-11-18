import torch.nn as nn
import torch.nn.functional as F
import torch
from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d
from stereo.modeling.cost_volume.cost_volume import correlation_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
from stereo.modeling.disp_refinement.disp_refinement import context_upsample

from .backbone import Backbone, FPNLayer
from .aggregation import Aggregation


class LightStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.args = cfgs
        self.max_disp = cfgs.MAX_DISP
        self.left_att = cfgs.LEFT_ATT

        # backbobe
        self.backbone = Backbone(cfgs.get('BACKCONE', 'MobileNetv2'))

        # aggregation
        self.cost_agg = Aggregation(in_channels=48,
                                    left_att=self.left_att,
                                    blocks=cfgs.AGGREGATION_BLOCKS,
                                    expanse_ratio=cfgs.EXPANSE_RATIO,
                                    backbone_channels=self.backbone.output_channels)

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(self.backbone.output_channels[0], 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']

        features_left = self.backbone(image1)
        features_right = self.backbone(image2)

        gwc_volume = correlation_volume(features_left[0], features_right[0], self.max_disp // 4)
        encoding_volume = self.cost_agg(gwc_volume, features_left)  # [bz, 1, max_disp/4, H/4, W/4]
        squeezed_encoding = encoding_volume[0].reshape(encoding_volume[0].size(0), -1, encoding_volume[0].size(2), encoding_volume[0].size(3))  # [bz, max_disp/4, H/4, W/4]

        prob = F.softmax(squeezed_encoding, dim=1)
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(image1))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)  # [bz, 9, H, W]
        disp_pred = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # # [bz, 1, H, W]

        result = {'disp_pred': disp_pred}

        if self.training:
            disp_4 = F.interpolate(init_disp, image1.shape[2:], mode='bilinear', align_corners=False)
            disp_4 *= 4
            result['disp_4'] = disp_4

        return result

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, 1, h, w]

        disp_pred = model_pred['disp_pred']
        disp_4 = model_pred['disp_4']
        if self.args.ignore_ratio == 0:
            loss = 1.0 * F.smooth_l1_loss(disp_pred[mask], disp_gt[mask], reduction='mean')
            loss += 0.3 * F.smooth_l1_loss(disp_4[mask], disp_gt[mask], reduction='mean')

        else:
            init_per_pixel_loss = F.smooth_l1_loss(disp_pred, disp_gt, reduction='none')
            init_valid = mask.bool() & ~torch.isnan(disp_pred)
            init_masked_loss, init_mask = self.ignore_top_loss_regions(
                init_per_pixel_loss, init_valid, ignore_ratio=self.args.ignore_ratio
            )

            # 计算初始预测的平均损失
            loss = 1.0 * (init_masked_loss.sum() / init_mask.sum())
            per_pixel_loss = F.smooth_l1_loss(disp_4, disp_gt,reduce='none')
            masked_loss_4, mask_4 = self.ignore_top_loss_regions(
                per_pixel_loss, init_valid, ignore_ratio=self.args.ignore_ratio
            )
            loss += 0.3 * (masked_loss_4.sum() / mask_4.sum())

        if self.args.GM_RATIO != 0:
            gm_loss = self.gm_loss(disp_pred, disp_gt, mask) + 0.3 * self.gm_loss(disp_4, disp_gt, mask)
            loss += self.args.GM_RATIO * gm_loss

            loss_info = {'scalar/train/loss_disp': loss.item(),
                     'scalar/train/gm loss': gm_loss.item()}
        else:

            loss_info = {'scalar/train/loss_disp': loss.item()}

        return loss, loss_info

    # def gm_loss(self, disp_pred, disp_gt, mask=None):
    #     def gradient_x(img):
    #         return img[:, :, :, :-1] - img[:, :, :, 1:]

    #     def gradient_y(img):
    #         return img[:, :, :-1, :] - img[:, :, 1:, :]

    #     if mask is None:
    #         mask = torch.ones_like(disp_pred)
    #     mask = mask.float()
    #     total_loss = 0.0
    #     residual = disp_pred - disp_gt

    #     for scale in range(4):
    #         if scale > 0:
    #             scale_factor = 1.0 / (2 ** scale)
    #             current_residual = F.interpolate(residual, scale_factor=scale_factor,
    #                                             mode='bilinear', align_corners=False)
    #             current_mask = F.interpolate(mask, scale_factor=scale_factor,
    #                                         mode='nearest')
    #         else:
    #             current_residual = residual
    #             current_mask = mask

    #         grad_x = torch.abs(gradient_x(current_residual))[current_mask[:, :, :, :-1].bool()]
    #         grad_y = torch.abs(gradient_y(current_residual))[current_mask[:, :, :-1, :].bool()]

    #         total_loss += (grad_x.sum() + grad_y.sum()) / (mask.sum())

    #     return total_loss

    def gm_loss(self, disp_pred, disp_gt, mask=None):
        def gradient_x(img):
            return img[:, :, :, :-1] - img[:, :, :, 1:]

        def gradient_y(img):
            return img[:, :, :-1, :] - img[:, :, 1:, :]

        if mask is None:
            mask = torch.ones_like(disp_pred)
        mask = mask.float()
        total_loss = 0.0

        for scale in range(4):
            if scale > 0:
                scale_factor = 1.0 / (2 ** scale)
                # 分别对预测视差和真实视差进行下采样
                disp_pred_scaled = F.interpolate(disp_pred, scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                disp_gt_scaled = F.interpolate(disp_gt, scale_factor=scale_factor,
                                            mode='bilinear', align_corners=False)
                current_mask = F.interpolate(mask, scale_factor=scale_factor,
                                            mode='nearest')
            else:
                disp_pred_scaled = disp_pred
                disp_gt_scaled = disp_gt
                current_mask = mask

            # 先计算视差的梯度
            grad_pred_x = gradient_x(disp_pred_scaled)
            grad_pred_y = gradient_y(disp_pred_scaled)
            grad_gt_x = gradient_x(disp_gt_scaled)
            grad_gt_y = gradient_y(disp_gt_scaled)

            # 计算梯度之间的差异
            grad_diff_x = torch.abs(grad_pred_x - grad_gt_x)
            grad_diff_y = torch.abs(grad_pred_y - grad_gt_y)

            # 应用mask（注意梯度图的尺寸比原图小1）
            mask_x = current_mask[:, :, :, :-1]
            mask_y = current_mask[:, :, :-1, :]

            total_loss += (grad_diff_x[mask_x.bool()].sum() + grad_diff_y[mask_y.bool()].sum()) / (mask.sum())

        return total_loss


    def ignore_top_loss_regions(self, loss_map, valid_mask, ignore_ratio=0.1):
        # 计算要保留的元素数量
        num_valid = valid_mask.sum(dim=(1,2,3), keepdim=True)
        keep_num = (num_valid * (1 - ignore_ratio)).int()

        # 创建调整后的损失（无效区域设为极大值）
        adjusted_loss = torch.where(valid_mask, loss_map, torch.tensor(float('inf'), device=loss_map.device))

        # 获取每个样本的 topk 阈值（可导操作）
        topk_values = torch.topk(
            adjusted_loss.view(adjusted_loss.size(0), -1),
            k=keep_num.view(-1).max().item(),
            dim=1,
            largest=False
        ).values

        # # 创建阈值张量
        # thresholds = topk_values.gather(
        #     1, (keep_num.view(-1) - 1).clamp(min=0).unsqueeze(1)
        # ).view(-1, 1, 1, 1)

        thresholds = topk_values.gather(
            1, (keep_num.view(-1) - 1).clamp(min=0).to(torch.long).unsqueeze(1)
        ).view(-1, 1, 1, 1)

        # 创建保留掩码（可导）
        keep_mask = (loss_map <= thresholds) & valid_mask

        return loss_map * keep_mask.float(), keep_mask.float()
