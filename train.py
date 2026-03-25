import sys
sys.path.append("..")
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import mstar_dataset
import math
import torch.autograd as autograd
import numpy as np
from net import Discriminator,Generator
import scipy.io as sio
import model_HQS
from tensorboardX import SummaryWriter
writer = SummaryWriter('../tensorboard/'+'phigan')

        
# 随机采样类别和角度生成图像
def show_result(G,num_epoch, show = False, save = False, path = 'result.png'): #
    # fixed noise & label&az

    fixed_z_ = torch.randn(100, 64)
    fixed_y_ = torch.zeros(10, 1)
    for i in range(9):
        temp = torch.ones(10, 1) + i
        fixed_y_ = torch.cat([fixed_y_, temp], 0)
    fixed_y_label_ = torch.zeros(100, 10)
    fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)

    fixed_az = torch.deg2rad(360*torch.rand(100, 1))
    fixed_az_vec = torch.zeros(100, 10)
    for i in range(100):
        fixed_az_vec[i][0] = torch.cos(fixed_az[i])
        fixed_az_vec[i][1] = torch.sin(fixed_az[i])
        fixed_az_vec[i][2] = torch.cos(2*fixed_az[i])
        fixed_az_vec[i][3] = torch.sin(2*fixed_az[i])
        fixed_az_vec[i][4] = torch.cos(3*fixed_az[i])
        fixed_az_vec[i][5] = torch.sin(3*fixed_az[i])
        fixed_az_vec[i][6] = torch.cos(4*fixed_az[i])
        fixed_az_vec[i][7] = torch.sin(4*fixed_az[i])
        fixed_az_vec[i][8] = torch.cos(5*fixed_az[i])
        fixed_az_vec[i][9] = torch.sin(5*fixed_az[i])

    with torch.no_grad():
        fixed_z_ = Variable(fixed_z_.cuda())
        fixed_y_label_ = Variable(fixed_y_label_.cuda())
        fixed_az_vec = Variable(fixed_az_vec.cuda())
    G.eval()
    test_images = G(fixed_z_, fixed_y_label_, fixed_az_vec)

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def crop_center(img_batch, cropx, cropy):
    batch_size = img_batch.shape[0]
    crop_img = torch.zeros([batch_size, 80, 80]).cuda()
    for i in range(batch_size):
        img = img_batch[i, :, :]
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        crop_img[i, :, :] = img[starty:starty + cropy, startx:startx + cropx]
    return crop_img


def NormImage(img_batch):
    '''
    Normalize images in a batch by L2 normalization
    '''
    # 假设 img_batch 是一个形状为 (batch_size, channels, height, width) 的 PyTorch 张量
    batch_size = img_batch.shape[0]
    norm_img = torch.zeros([batch_size, 80, 80]).cuda()
    for i in range(batch_size):
        img = img_batch[i, :, :]
        norm_img[i, :, :] = img / torch.linalg.norm(img)
    return norm_img



def parameter_setting(args):
    config = {}
    config['bs'] = args.bs
    config['lrg'] = args.lrg
    config['lrd'] = args.lrd
    config['num_epochs'] = args.num_epochs
    config['save_dir'] = args.save_dir
    config['train_txt'] = args.train_txt
    config['d_mat'] = args.d_mat
    config['d_h_mat'] = args.d_h_mat
    config['inv_d_mat'] = args.inv_d_mat
    config['f_est'] = args.f_est
    return config

def train(config):
# training parameters
    batch_size = config['bs']
    lr_G = config['lrg']
    lr_D = config['lrd']
    train_epoch = config['num_epochs']
    train_txt = config['train_txt']
    d_mat = config['d_mat']
    d_h_mat = config['d_h_mat']
    inv_d_mat = config['inv_d_mat']
    f_est = config['f_est']
# data_loader
    data_transforms = transforms.Compose([
            transforms.ToTensor(),
    ])

    dataset_train = mstar_dataset.MSTAR_Dataset(txt_file=train_txt,
                                                transform=data_transforms,
                                                )
    train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size, shuffle=True)


    # network
    G = Generator()
    D = Discriminator()
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()



    D_deep_H = sio.loadmat(d_h_mat)['D_H']
    D_deep = sio.loadmat(d_mat)['D']
    Inv_D = sio.loadmat(inv_d_mat)['Inv_D']

    D_deep = torch.from_numpy(D_deep.astype('complex64'))
    D_deep_H = torch.from_numpy(D_deep_H.astype('complex64'))
    Inv_D = torch.from_numpy(Inv_D.astype('complex64'))
    D_deep = D_deep.cuda()
    D_deep_H = D_deep_H.cuda()
    Inv_D = Inv_D.cuda()
    model_du = model_HQS.HQS_iteration_model(D = D_deep, T = 2, W = D_deep_H, D_D_H_inv = Inv_D,alpha=0.0001, theta=0.0001, mu = 0.0001)
    model_du.load_state_dict(torch.load(f_est))
   
    model_du.eval()


    # Binary Cross Entropy loss
    Decision_loss = nn.BCELoss()
    Class_loss = nn.CrossEntropyLoss()
    Az_loss = nn.MSELoss()
    one2onemapping = nn.L1Loss()


    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

    # results save folder
    root = config['save_dir']
    model = 'phigan_'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'generated_results'):
        os.mkdir(root + 'generated_results')

    # label preprocess
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1)

    print('training start.')
    start_time = time.time()

    for epoch in range(train_epoch):
        D_losses = []
        G_losses = []
        C_G_losses = []
        A_G_losses = []
        C_D_losses = []
        A_D_losses = []
        logits_fake = []
        logits_real = []
        logits_g = []
        DU_losses = []

        if (epoch+1) <= 80:
            d_steps = 1
            g_steps = 5
        else:
            d_steps = 1
            g_steps = 1

        epoch_start_time = time.time()

        for idx, data in enumerate(train_loader):
            x_data = data['image']
            x_label = data['label'].cuda()
            x_az = data['az']
            D.zero_grad()

            mini_batch = x_data.size()[0]

            y_real = torch.ones(mini_batch)
            y_fake = torch.zeros(mini_batch)
            y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())


            real_label = onehot[x_label]

            random_z = torch.randn((mini_batch, 64))

            real_angle = torch.deg2rad(x_az)
            real_az_vec = torch.zeros(mini_batch, 10)
            for i in range(mini_batch):
                real_az_vec[i][0] = torch.cos(real_angle[i])
                real_az_vec[i][1] = torch.sin(real_angle[i])
                real_az_vec[i][2] = torch.cos(2*real_angle[i])
                real_az_vec[i][3] = torch.sin(2*real_angle[i])
                real_az_vec[i][4] = torch.cos(3*real_angle[i])
                real_az_vec[i][5] = torch.sin(3*real_angle[i])
                real_az_vec[i][6] = torch.cos(4*real_angle[i])
                real_az_vec[i][7] = torch.sin(4*real_angle[i])
                real_az_vec[i][8] = torch.cos(5*real_angle[i])
                real_az_vec[i][9] = torch.sin(5*real_angle[i])

            x_data = Variable(x_data.cuda())
            real_label = Variable(real_label.cuda())
            real_az_vec = Variable(real_az_vec.cuda())
            random_z = Variable(random_z.cuda())
            x_az = Variable(x_az.cuda())

            for d_index in range(d_steps):
           # train discriminator D
                x_array = x_data.squeeze()        
                real_in = crop_center(x_array,80,80)
                real_in = real_in.type(torch.complex64)
                real_in = NormImage(real_in)
                real_image_tim = abs(real_in.permute(0, 2, 1).reshape(mini_batch,6400,1))
                real_image_tim = real_image_tim.type(torch.complex64).cuda()
                sc_real = model_du(real_image_tim, x0 = real_image_tim)
                sc_real_img = abs(sc_real).reshape(mini_batch,1,80,80).cuda()
                sc_real_in = F.pad(sc_real_img, (24, 24, 24, 24), mode='constant', value=0)

                d_real, class_real, theta_real ,res_f64_real, res_f32_real, res_f16_real, res_f8_real = D(x_data,sc_real_in)
                decision_real = d_real.squeeze()
                class_real = class_real.squeeze()
                theta_real = theta_real.squeeze()

                D_real_loss = Decision_loss(decision_real,y_real)
                C_real_loss = Class_loss(class_real, real_label)
                A_real_loss = Az_loss(theta_real, real_az_vec)

                x_re = G(random_z, real_label, real_az_vec)

                x_re_array = x_re.squeeze()
                fake_in = crop_center(x_re_array,80,80)
                fake_in = fake_in.type(torch.complex64)
                fake_in = NormImage(fake_in)
                fake_image_tim = abs(fake_in.permute(0, 2, 1).reshape(mini_batch,6400,1))       
                fake_image_tim = fake_image_tim.type(torch.complex64).cuda()

                sc_fake = model_du(fake_image_tim, x0 = fake_image_tim)
                sc_fake_img = abs(sc_fake).reshape(mini_batch,1,80,80).cuda()
                sc_fake_in = F.pad(sc_fake_img, (24, 24, 24, 24), mode='constant', value=0)

                d_fake, class_fake, theta_fake, res_f64_fake, res_f32_fake, res_f16_fake, res_f8_fake = D(x_re,sc_fake_in)
                decision_fake = d_fake.squeeze()
                class_fake = class_fake.squeeze()
                theta_fake = theta_fake.squeeze()

                D_fake_loss = Decision_loss(decision_fake,y_fake)
                C_fake_loss = Class_loss(class_fake, real_label)
                A_fake_loss = Az_loss(theta_fake, real_az_vec)

                DF_8_D_loss = 0.5*(torch.norm(res_f8_real)+torch.norm(res_f8_fake))/(res_f8_fake.shape[1]*res_f8_fake.shape[2]*res_f8_fake.shape[3])
                DF_16_D_loss = 0.5*(torch.norm(res_f16_real)+torch.norm(res_f16_fake))/(res_f16_fake.shape[1]*res_f16_fake.shape[2]*res_f16_fake.shape[3])
                DF_32_D_loss = 0.5*(torch.norm(res_f32_real)+torch.norm(res_f32_fake))/(res_f32_fake.shape[1]*res_f32_fake.shape[2]*res_f32_fake.shape[3])
                DF_64_D_loss = 0.5*(torch.norm(res_f64_real)+torch.norm(res_f64_fake))/(res_f64_fake.shape[1]*res_f64_fake.shape[2]*res_f64_fake.shape[3])
                DF_D_loss =  (DF_8_D_loss + DF_16_D_loss+ DF_32_D_loss+ DF_64_D_loss)

                D_train_loss = 0.5*( D_fake_loss + D_real_loss) + 5*(C_real_loss + C_fake_loss) + 20*(A_real_loss + A_fake_loss) + DF_D_loss
                C_Dtrain_losses = C_real_loss 
                A_Dtrain_losses = A_real_loss 
                D_train_loss.backward()
                D_optimizer.step()


            D_losses.append(D_fake_loss + D_real_loss)
            C_D_losses.append(C_Dtrain_losses)
            A_D_losses.append(A_Dtrain_losses)
            logits_real.append(decision_real.mean())
            logits_fake.append(decision_fake.mean())


            for g_index in range(g_steps):
                G.zero_grad()
            
                x_re = G(random_z, real_label, real_az_vec)


                x_array = x_data.squeeze()        
                real_in = crop_center(x_array,80,80)
                real_in = real_in.type(torch.complex64)
                real_in = NormImage(real_in)
                real_image_tim = abs(real_in.permute(0, 2, 1).reshape(mini_batch,6400,1))
                real_image_tim = real_image_tim.type(torch.complex64).cuda()

                x_re_array = x_re.squeeze()
                fake_in = crop_center(x_re_array,80,80)
                fake_in = fake_in.type(torch.complex64)
                fake_in = NormImage(fake_in)
                fake_image_tim = abs(fake_in.permute(0, 2, 1).reshape(mini_batch,6400,1))       
                fake_image_tim = fake_image_tim.type(torch.complex64).cuda()

                sc_real = model_du(real_image_tim, x0 = real_image_tim)
                sc_fake = model_du(fake_image_tim, x0 = fake_image_tim)
                sc_real_img = abs(sc_real).reshape(mini_batch,1,80,80).cuda()
                sc_real_in = F.pad(sc_real_img, (24, 24, 24, 24), mode='constant', value=0)

                du_loss = torch.mean(torch.norm(abs(sc_real-sc_fake)))

                d_gf, class_gf, theta_gf ,res_f64_sc_real, res_f32_sc_real, res_f16_sc_real, res_f8_sc_real = D(x_re,sc_real_in)
                DF_8_G_loss = torch.norm(res_f8_sc_real)/(res_f8_sc_real.shape[1]*res_f8_sc_real.shape[2]*res_f8_sc_real.shape[3])
                DF_16_G_loss = torch.norm(res_f16_sc_real)/(res_f16_sc_real.shape[1]*res_f16_sc_real.shape[2]*res_f16_sc_real.shape[3])
                DF_32_G_loss = torch.norm(res_f32_sc_real)/(res_f32_sc_real.shape[1]*res_f32_sc_real.shape[2]*res_f32_sc_real.shape[3])
                DF_64_G_loss = torch.norm(res_f64_sc_real)/(res_f64_sc_real.shape[1]*res_f64_sc_real.shape[2]*res_f64_sc_real.shape[3])
                DF_G_loss =  (DF_8_G_loss + DF_16_G_loss+ DF_32_G_loss+ DF_64_G_loss)

    
                decision_gf = d_gf.squeeze()
                class_gf = class_gf.squeeze()
                theta_gf = theta_gf.squeeze()

                G_loss = Decision_loss(decision_gf,y_real)
                C_Gtrain_loss = Class_loss(class_gf, real_label)
                A_Gtrain_loss = Az_loss(theta_gf, real_az_vec)

                G_train_loss = (G_loss + 5*C_Gtrain_loss + 20*A_Gtrain_loss+ one2onemapping(x_re,x_data) + du_loss + DF_G_loss)
                G_train_loss.backward()
                G_optimizer.step()

            G_losses.append(D_fake_loss)
            C_G_losses.append(C_Gtrain_loss)
            A_G_losses.append(A_Gtrain_loss)
            DU_losses.append(du_loss)
            logits_g.append(decision_gf.mean())

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time


        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f  ' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                                torch.mean(torch.FloatTensor(G_losses))))
        fixed_p = root + 'generated_results/' + model + str(epoch + 1) + '.png'
        show_result(G,(epoch+1), save=True, path=fixed_p)

        writer.add_scalars('G_losses', {'train': torch.mean(torch.FloatTensor(G_losses))}, epoch + 1)
        writer.add_scalars('D_losses', {'train': torch.mean(torch.FloatTensor(D_losses))},epoch+1)
        writer.add_scalars('C_losses', {'D': torch.mean(torch.FloatTensor(C_D_losses)),'G': torch.mean(torch.FloatTensor(C_G_losses))}, epoch + 1)
        writer.add_scalars('A_losses', {'D': torch.mean(torch.FloatTensor(A_D_losses)),'G': torch.mean(torch.FloatTensor(A_G_losses))}, epoch + 1)
        writer.add_scalars('DU_losses', {'train': torch.mean(torch.FloatTensor(DU_losses))},epoch+1)
        writer.add_scalars('logits_fake', {'train': torch.mean(torch.FloatTensor(logits_fake))},epoch+1)
        writer.add_scalars('logits_real', {'train': torch.mean(torch.FloatTensor(logits_real))},epoch+1)
        writer.add_scalars('logits_g', {'train': torch.mean(torch.FloatTensor(logits_g))},epoch+1)

    print("Training finish... save training results")
    torch.save(G.state_dict(), root + model + 'generator_param.pkl') 
    torch.save(D.state_dict(), root + model + 'discriminator_param.pkl') 



if __name__ == '__main__': 
    parser = argparse.ArgumentParser(prog='GAN_training')

    parser.add_argument('--bs', type=int,default=32)
    parser.add_argument('--lrg', default=0.0001)
    parser.add_argument('--lrd', default=0.0001)
    parser.add_argument('--num_epochs', type=int,default=2000)
    parser.add_argument('--save_dir', default='phigan/')
    # parser.add_argument('--train_txt', default='../data/train.txt')
    parser.add_argument('--train_txt', default='train.txt')
    parser.add_argument('--d_mat', default='D_80*80_image_domain.mat')
    parser.add_argument('--d_h_mat', default='D_80*80_image_domain_H.mat')
    parser.add_argument('--inv_d_mat', default='D_80*80_image_domain_Inv_norm.mat')
    parser.add_argument('--f_est', default='HQS_epoch_30.pth')
    args = parser.parse_args()
    config = parameter_setting(args)
    train(config)

