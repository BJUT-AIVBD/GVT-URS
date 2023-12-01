# coding: utf-8
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


def draw_CAM(self, features, output, img_path, dim=1, save_path=None, Show=True):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''

    img = cv2.imread(img_path)  # 用cv2加载原始图像

    features_grads = list()
    heatmaps = list()

    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        features_grads.append(g)

    if not isinstance(dim, list):
        dim = [dim]

    features.register_hook(extract)
    Tensor = torch.ones(output[:, 0, :, :].shape).to(0)

    # obtain all feature maps' grident
    for i in range(len(dim)):
        # 预测得分最高的那一类对应的输出score
        pred_class = output[:, dim[i], :, :]
        self.zero_grad()
        pred_class.backward(Tensor, retain_graph=True)  # 计算梯度

    for i in range(len(features_grads)):
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(features_grads[i], (1, 1))

        # 此处batch size默认为1，所以去掉了第0维（batch size维）
        pooled_grad = pooled_grads[0]
        feature = features[0]

        feature = feature * pooled_grad

        # 以下部分同Keras版实现
        heatmap = feature.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)

        heatmap = (heatmap - np.min(heatmap)) / np.max(heatmap)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.4 + img * 0.6  # 这里的0.4是热力图强度因子

        if Show:
            plt.matshow(np.uint8(superimposed_img))
            plt.show()

        if save_path is not None:
            cv2.imwrite(save_path, superimposed_img)

        heatmaps.append(np.uint8(superimposed_img))

    heatmaps = torch.from_numpy(np.array(heatmaps)).permute(0, 3, 1, 2)

    return heatmaps
