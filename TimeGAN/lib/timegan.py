import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from TimeGAN.lib.data import batch_generator
from TimeGAN.utils import extract_time, random_generator, NormMinMax
from TimeGAN.lib.model import Encoder, Recovery, Generator, Discriminator, Supervisor
from TimeGAN.lib.discriminative_metrics import discriminative_score_metrics
from TimeGAN.lib.predictive_metrics import predictive_score_metrics
from TimeGAN.lib.visualization_metrics import visualization


class BaseModel():
    def __init__(self, opt, ori_data):
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
        self.ori_time, self.max_seq_len = extract_time(self.ori_data)
        self.data_num, _, _ = np.asarray(ori_data).shape    # 3661; 24; 6
        self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        self.device = torch.device(
            "cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.metrics = {}

    def seed(self, seed_value):
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    def save_weights(self, epoch):
        weight_dir = os.path.join(
            self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
                   '%s/netE.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
                   '%s/netR.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
                   '%s/netS.pth' % (weight_dir))

    def train_one_iter_er(self):
        self.nete.train()
        self.netr.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size)
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train encoder & decoder
        self.optimize_params_er()

    def train_one_iter_er_(self):
        self.nete.train()
        self.netr.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size)
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train encoder & decoder
        self.optimize_params_er_()

    def train_one_iter_s(self):
        # self.nete.eval()
        self.nets.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size)
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)

        # train superviser
        self.optimize_params_s()

    def train_one_iter_g(self):
        self.netg.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size)
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
        self.Z = random_generator(
            self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

        # train superviser
        self.optimize_params_g()

    def train_one_iter_d(self):
        self.netd.train()

        # set mini-batch
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size)
        self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
        self.Z = random_generator(
            self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

        # train superviser
        self.optimize_params_d()

    def train(self):
        for iter in range(1, self.opt.iteration + 1):
            # Train for one iter
            self.train_one_iter_er()
            if iter % 100 == 0:    
                print('Encoder training step: ' + str(iter) +
                  '/' + str(self.opt.iteration))

        for iter in range(1, self.opt.iteration + 1):
            # Train for one iter
            self.train_one_iter_s()
            if iter % 100 == 0:
                print('Superviser training step: '  +
                  str(iter) + '/' + str(self.opt.iteration))

        for iter in range(1, self.opt.iteration + 1):
            # Train for one iter
            for kk in range(2):
                self.train_one_iter_g()
                self.train_one_iter_er_()

            self.train_one_iter_d()
            if iter % 100 == 0:
                print('Joint training step: ' +
                  str(iter) + '/' + str(self.opt.iteration))

        self.save_weights(self.opt.iteration)
        self.generated_data = self.generation(self.opt.generation_size)
        print('Finish Synthetic Data Generation')

    def evaluation(self):
        metric_results = dict()


        discriminative_score = list()
        for _ in range(self.opt.metric_iteration):
            temp_disc = discriminative_score_metrics(self.ori_data, self.generated_data)
            discriminative_score.append(temp_disc)

        metric_results['discriminative'] = np.mean(discriminative_score)

        predictive_score = list()
        for tt in range(self.opt.metric_iteration):
            temp_pred = predictive_score_metrics(self.ori_data, self.generated_data)
            predictive_score.append(temp_pred)

        metric_results['predictive'] = np.mean(predictive_score)
        self.metrics = metric_results
        
        
        # 3. Visualization (PCA and tSNE)
        visualization(self.ori_data, self.generated_data, 'pca')
        '''
        visualization(self.ori_data, self.generated_data, 'tsne')
        '''

    def generation(self, num_samples, mean=0.0, std=1.0):
        if num_samples == 0:
            return None, None
        # Synthetic data generation
        self.X0, self.T = batch_generator(
            self.ori_data, self.ori_time, self.opt.batch_size)
        self.Z = random_generator(
            num_samples, self.opt.z_dim, self.T, self.max_seq_len)
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)   
        self.H_hat = self.nets(self.E_hat)  
        generated_data_curr = self.netr(
            self.H_hat).cpu().detach().numpy()  

        generated_data = list()
        for i in range(num_samples):
            temp = generated_data_curr[i, :self.ori_time[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * self.max_val
        generated_data = generated_data + self.min_val
        return generated_data


class TimeGAN(BaseModel):
    @property
    def name(self):
        return 'TimeGAN'

    def __init__(self, opt, ori_data):
        super(TimeGAN, self).__init__(opt, ori_data)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        # Create and initialize networks.
        self.nete = Encoder(self.opt).to(self.device)
        self.netr = Recovery(self.opt).to(self.device)
        self.netg = Generator(self.opt).to(self.device)
        self.netd = Discriminator(self.opt).to(self.device)
        self.nets = Supervisor(self.opt).to(self.device)

        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(
                self.opt.resume, 'netG.pth'))['epoch']
            self.nete.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netE.pth'))['state_dict'])
            self.netr.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netR.pth'))['state_dict'])
            self.netg.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netD.pth'))['state_dict'])
            self.nets.load_state_dict(torch.load(os.path.join(
                self.opt.resume, 'netS.pth'))['state_dict'])
            print("\tDone.\n")

        # loss
        self.l_mse = nn.MSELoss()
        self.l_r = nn.L1Loss()
        self.l_bce = nn.BCELoss()

        # Setup optimizer
        if self.opt.isTrain:
            self.nete.train()
            self.netr.train()
            self.netg.train()
            self.netd.train()
            self.nets.train()
            self.optimizer_e = optim.Adam(self.nete.parameters(
            ), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_r = optim.Adam(self.netr.parameters(
            ), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(
            ), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_d = optim.Adam(self.netd.parameters(
            ), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_s = optim.Adam(self.nets.parameters(
            ), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def forward_e(self):
        self.H = self.nete(self.X)

    def forward_er(self):
        self.H = self.nete(self.X)
        self.X_tilde = self.netr(self.H)

    def forward_g(self):
        self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
        self.E_hat = self.netg(self.Z)

    def forward_dg(self):
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def forward_rg(self):
        self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
        self.H_supervise = self.nets(self.H)
        # print(self.H, self.H_supervise)

    def forward_sg(self):
        self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
        self.Y_real = self.netd(self.H)
        self.Y_fake = self.netd(self.H_hat)
        self.Y_fake_e = self.netd(self.E_hat)

    def backward_er(self):
        self.err_er = self.l_mse(self.X_tilde, self.X)
        self.err_er.backward(retain_graph=True)
        #print("Loss: ", self.err_er)

    def backward_er_(self):
        self.err_er_ = self.l_mse(self.X_tilde, self.X)
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
        self.err_er.backward(retain_graph=True)

    #  print("Loss: ", self.err_er_, self.err_s)
    def backward_g(self):
        self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))

        self.err_g_U_e = self.l_bce(
            self.Y_fake_e, torch.ones_like(self.Y_fake_e))
        self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat, [0])[
                                   1] + 1e-6) - torch.sqrt(torch.std(self.X, [0])[1] + 1e-6)))   # |a^2 - b^2|
        self.err_g_V2 = torch.mean(torch.abs(
            (torch.mean(self.X_hat, [0])[0]) - (torch.mean(self.X, [0])[0])))  # |a - b|
        self.err_s = self.l_mse(self.H_supervise[:, :-1, :], self.H[:, 1:, :])
        self.err_g = self.err_g_U + \
            self.err_g_U_e * self.opt.w_gamma + \
            self.err_g_V1 * self.opt.w_g + \
            self.err_g_V2 * self.opt.w_g + \
            torch.sqrt(self.err_s)
        self.err_g.backward(retain_graph=True)
        #print("Loss G: ", self.err_g)

    def backward_s(self):
        self.err_s = self.l_mse(self.H[:, 1:, :], self.H_supervise[:, :-1, :])
        self.err_s.backward(retain_graph=True)
        #print("Loss S: ", self.err_s)
     #   print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def backward_d(self):
        self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
        self.err_d_fake = self.l_bce(
            self.Y_fake, torch.zeros_like(self.Y_fake))
        self.err_d_fake_e = self.l_bce(
            self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
        self.err_d = self.err_d_real + \
            self.err_d_fake + \
            self.err_d_fake_e * self.opt.w_gamma
        if self.err_d > 0.15:
            self.err_d.backward(retain_graph=True)

         # print("Loss D: ", self.err_d)

    def optimize_params_er(self):
        # Forward-pass
        self.forward_er()

        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_er_(self):
        # Forward-pass
        self.forward_er()
        self.forward_s()
        # Backward-pass
        # nete & netr
        self.optimizer_e.zero_grad()
        self.optimizer_r.zero_grad()
        self.backward_er_()
        self.optimizer_e.step()
        self.optimizer_r.step()

    def optimize_params_s(self):
        # Forward-pass
        self.forward_e()
        self.forward_s()

        # Backward-pass
        # nets
        self.optimizer_s.zero_grad()
        self.backward_s()
        self.optimizer_s.step()

    def optimize_params_g(self):
        # Forward-pass
        self.forward_e()
        self.forward_s()
        self.forward_g()
        self.forward_sg()
        self.forward_rg()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_g.zero_grad()
        self.optimizer_s.zero_grad()
        self.backward_g()
        self.optimizer_g.step()
        self.optimizer_s.step()

    def optimize_params_d(self):
        # Forward-pass
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        self.forward_d()
        self.forward_dg()

        # Backward-pass
        # nets
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
