import torch
import torch.nn as nn
import numpy as np
import time


class HQS_iteration_model(nn.Module):
    def __init__(self, D, T, W,  D_D_H_inv, alpha, theta, mu):

        super(HQS_iteration_model, self).__init__()
        self.D = D                  ## D
        self.D_H = W                ## D_H
        self.Inv = D_D_H_inv        ## (D * D_H)^(-1)

        self.T = T                  ## Layers
        self.M = D.shape[0]   ## 6400
        self.N = D.shape[1]   ## 6400
        
        self.theta = theta          ## z(k+1)里的软阈值界限
        self.alpha = alpha          ## z(k+1)里的
        self.mu = mu                ## x(k+1)里的
                                                    
        """ 设置网络层。"""
        self.setup_layers()
        
    def setup_layers(self):
        self.alphas = nn.ParameterList([nn.Parameter(torch.Tensor([self.alpha]).cuda()) for t in range(self.T)])
        self.thetas = nn.ParameterList([nn.Parameter(torch.Tensor([self.theta]).cuda()) for t in range(self.T)])
        self.mus = nn.ParameterList([nn.Parameter(torch.Tensor([self.mu]).cuda()) for t in range(self.T)])
    def forward(self, y, x0=None):
        xhs = []
        zhs = []
        
        if x0 is None:
            # batch_size = y.shape[0]
            xh = torch.zeros(y.shape[0], self.N, 1, dtype=torch.complex64).cuda()
            zh = torch.zeros(y.shape[0], self.N, 1, dtype=torch.complex64).cuda()
        else:
            zh = x0
            xh = x0
        xh = torch.zeros(y.shape[0], self.N, 1, dtype=torch.complex64).cuda()
        xhs.append(xh)
        zhs.append(zh)
        # 在PyTorch中，不需要使用`tf.variable_scope`，直接使用nn.Module即可
        for t in range(self.T):
            alpha = self.alphas[t]
            theta = self.thetas[t]
            mu = self.mus[t]
            # print(alpha, theta)
            alpha = alpha.cuda()
            theta = theta.cuda()
            mu = mu.cuda()
            
            #! 求解x(k+1)
            k1 = torch.einsum('...ij,...jk->...ik', [self.D_H,self.Inv])
            k2 = torch.einsum('...ij,...jk->...ik', [self.D,zh])
            k3 = torch.einsum('...ij,...jk->...ik', [k1,k2])
            xh = zh + mu * k3
            
            #! 求解z(k+1)
            res = xh - k2
            mid = zh + alpha * torch.einsum('...ij,...jk->...ik', [self.D_H,res])
            zh = self.complex_shrink(mid, theta)
            
        x_final = xh
        z_final = zh
        sc = torch.einsum('...ij,...jk->...ik', [self.D,z_final])
        # sc = z_final
        return sc
    
    def complex_shrink(self, x, T):
        #这是一个软阈值函数
        #x是输入的张量，T是阈值
        X_abs = torch.abs(x)
        X_abs = torch.clamp(X_abs, min=1e-8)
        def Complex_sign(x):
            return x / X_abs
        def Complex_max(x, b):
            re = torch.where(x > b, x, b)
            return re

        soft = torch.multiply(Complex_sign(x), (Complex_max(torch.Tensor([0]).cuda(), torch.abs(x)-T)))
        return soft
    
# class HQS_iteration_model(nn.Module):
#     def __init__(self, D, T, W,  D_D_H_inv, alpha, theta, mu):

#         super(HQS_iteration_model, self).__init__()
#         self.D = D                  ## D
#         self.D_H = W                ## D_H
#         self.Inv = D_D_H_inv        ## (D * D_H)^(-1)

#         self.T = T                  ## Layers
#         self.M = D.shape[0]   ## 6400
#         self.N = D.shape[1]   ## 6400
        
#         self.theta = theta          ## z(k+1)里的软阈值界限
#         self.alpha = alpha          ## z(k+1)里的
#         self.mu = mu                ## x(k+1)里的
                                                    
#         """ 设置网络层。"""
#         self.setup_layers()
        
#     def setup_layers(self):
#         self.alphas = nn.ParameterList([nn.Parameter(torch.Tensor([self.alpha]).cuda()) for t in range(self.T)])
#         self.thetas = nn.ParameterList([nn.Parameter(torch.Tensor([self.theta]).cuda()) for t in range(self.T)])
#         self.mus = nn.ParameterList([nn.Parameter(torch.Tensor([self.mu]).cuda()) for t in range(self.T)])
#     def forward(self, y, x0=None):
#         xhs = []
#         zhs = []
        
#         if x0 is None:
#             # batch_size = y.shape[0]
#             xh = torch.zeros(y.shape[0], self.N, 1, dtype=torch.complex64).cuda()
#             zh = torch.zeros(y.shape[0], self.N, 1, dtype=torch.complex64).cuda()
#         else:
#             zh = x0
#             xh = x0
#         xh = torch.zeros(y.shape[0], self.N, 1, dtype=torch.complex64).cuda()
#         xhs.append(xh)
#         zhs.append(zh)
#         # 在PyTorch中，不需要使用`tf.variable_scope`，直接使用nn.Module即可
#         for t in range(self.T):
#             alpha = self.alphas[t]
#             theta = self.thetas[t]
#             mu = self.mus[t]
#             # print(alpha, theta)
#             alpha = alpha.cuda()
#             theta = theta.cuda()
#             mu = mu.cuda()
            
#             #! 求解x(k+1)
#             k1 = torch.einsum('...ij,...jk->...ik', [self.D_H,self.Inv])
#             k2 = torch.einsum('...ij,...jk->...ik', [self.D,zh])
#             k3 = torch.einsum('...ij,...jk->...ik', [k1,k2])
#             xh = zh + mu * k3
            
#             #! 求解z(k+1)
#             res = xh - k2
#             mid = zh + alpha * torch.einsum('...ij,...jk->...ik', [self.D_H,res])
#             zh = self.complex_shrink(mid, theta)
            
#             xhs.append(xh)
#             zhs.append(zh)
#         x_final = xh
#         z_final = zh
#         return xhs, x_final,zhs,z_final
    
#     def complex_shrink(self, x, T):
#         #这是一个软阈值函数
#         #x是输入的张量，T是阈值
#         X_abs = torch.abs(x)
#         def Complex_sign(x):
#             return x / X_abs
#         def Complex_max(x, b):
#             re = torch.where(x > b, x, b)
#             return re

#         soft = torch.multiply(Complex_sign(x), (Complex_max(torch.Tensor([0]).cuda(), torch.abs(x)-T)))
#         return soft