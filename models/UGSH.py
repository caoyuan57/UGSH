from torch import nn
import torch
from torch.nn import functional as F
from torchvision import models
import os

import pdb


class UGSH(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim, hash_dim):
        super(UGSH, self).__init__()
        self.module_name = 'UGSH'
        # image_dim = 768
        self.image_dim = image_dim
        # text_dim = 768
        self.text_dim = text_dim
        # hidden_dim = 1024 * 4
        self.hidden_dim = hidden_dim
        # hash_dim = 64
        self.hash_dim = hash_dim

        self.gcn_dim = 1024 * 2

        self.middle_dim = 512

        # resnet = models.resnet18(weights= models.ResNet18_Weights.DEFAULT)

        # self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        # 图像模块
        
        self.image_module = nn.Sequential(
            nn.Linear(image_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.gcn_dim, bias=True),
            nn.BatchNorm1d(self.gcn_dim),
            nn.ReLU(True)
        )

        # 文本模块
        self.text_module = nn.Sequential(
            nn.Linear(text_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, self.gcn_dim, bias=True),
            nn.BatchNorm1d(self.gcn_dim),
            nn.ReLU(True)
        )

        # hash discriminator
        self.hash_dis = nn.Sequential(
            nn.Linear(self.hash_dim, self.hash_dim * 2, bias=True),
            nn.BatchNorm1d(self.hash_dim * 2),
            nn.ReLU(True),
            nn.Linear(self.hash_dim * 2, 1, bias=True)
        )

        # GCN part
        # self.gcnI1 = nn.Linear(self.gcn_dim + self.raw_dim, self.gcn_dim)
        self.gcnI1 = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.BNI1 = nn.BatchNorm1d(self.gcn_dim)
        self.actI1 = nn.ReLU(True)
                
        self.gcnT1 = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.BNT1 = nn.BatchNorm1d(self.gcn_dim)
        self.actT1 = nn.ReLU(True)

        self.text_Fusion = nn.Sequential(            
            nn.Linear(self.gcn_dim * 2, self.middle_dim, bias=True),
            nn.BatchNorm1d(self.middle_dim),
            nn.ReLU(True),
            nn.Linear(self.middle_dim, hash_dim, bias=True),
            nn.Tanh()
        )
        
        self.image_Fusion = nn.Sequential(
            nn.Linear(self.gcn_dim * 2, self.middle_dim, bias=True),
            nn.BatchNorm1d(self.middle_dim),
            nn.ReLU(True),
            nn.Linear(self.middle_dim, hash_dim, bias=True),
            nn.Tanh()
        )

        

        self.beta = 0.95
        # self.a = 0.5

    def forward(self, *args):
        if len(args) == 4:
            res = self.forward_img_img_txt_txt(*args)
        else:
            raise Exception('Method must take 2, 3 or 4 arguments')
        return res

    def forward_img_img_txt_txt(self, r_img1, r_img2, r_txt1, r_txt2):

        # shallow_feature = self.resnet(raw_img)
        h_img1 = self.image_module(r_img1)

        h_img2 = self.image_module(r_img2)

        h_txt1 = self.text_module(r_txt1)

        h_txt2 = self.text_module(r_txt2)

        f_img1 = F.normalize(h_img1, dim=1)
        Matrix_img1 = (f_img1.matmul(f_img1.transpose(0, 1)) >= self.beta).type(torch.cuda.FloatTensor)
        # g_img1 = self.gcnI1(Matrix_img1.mm(torch.cat((h_img1, shallow_feature), dim=1)))
        g_img1 = self.gcnI1(Matrix_img1.mm(h_img1))
        g_img1 = self.BNI1(g_img1)
        g_img1 = self.actI1(g_img1)

        f_img2 = F.normalize(h_img2, dim=1)
        Matrix_img2 = (f_img2.matmul(f_img2.transpose(0, 1)) >= self.beta).type(torch.cuda.FloatTensor)
        # g_img2 = self.gcnI1(Matrix_img2.mm(torch.cat((h_img2, shallow_feature), dim=1)))
        g_img2 = self.gcnI1(Matrix_img2.mm(h_img2))        
        g_img2 = self.BNI1(g_img2)
        g_img2 = self.actI1(g_img2)

        f_txt1 = F.normalize(h_txt1, dim=1)
        Matrix_txt1 = (f_txt1.matmul(f_txt1.transpose(0, 1)) >= self.beta).type(torch.cuda.FloatTensor)
        g_txt1 = self.gcnT1(Matrix_txt1.mm(h_txt1))
        g_txt1 = self.BNT1(g_txt1)
        g_txt1 = self.actT1(g_txt1)


        f_txt2 = F.normalize(h_txt2, dim=1)
        Matrix_txt2 = (f_txt2.matmul(f_txt2.transpose(0, 1)) >= self.beta).type(torch.cuda.FloatTensor)
        g_txt2 = self.gcnT1(Matrix_txt2.mm(h_txt2))
        g_txt2 = self.BNT1(g_txt2)
        g_txt2 = self.actT1(g_txt2)

        hg_img1 = self.image_Fusion(torch.cat((h_img1, g_img1), dim=1)).squeeze()
        hg_img2 = self.image_Fusion(torch.cat((h_img2, g_img2), dim=1)).squeeze()
        hg_txt1 = self.text_Fusion(torch.cat((h_txt1, g_txt1), dim=1)).squeeze()
        hg_txt2 = self.text_Fusion(torch.cat((h_txt2, g_txt2), dim=1)).squeeze()

        return hg_img1, hg_img2, hg_txt1, hg_txt2, Matrix_img1, Matrix_img2, Matrix_txt1, Matrix_txt2


    def generate_img_code(self, i):

        # hallow_feature = self.resnet(raw_img)
        # shallow_feature = shallow_feature.view(shallow_feature.shape[0], shallow_feature.shape[1])
        
        h_img = self.image_module(i)

        f_img = F.normalize(h_img, dim=1)
        Matrix_img = (f_img.matmul(f_img.transpose(0, 1)) >= self.beta).type(torch.cuda.FloatTensor)
        # g_img = self.gcnI1(Matrix_img.mm(torch.cat((h_img, shallow_feature), dim=1)))
        g_img = self.gcnI1(Matrix_img.mm(h_img))
        g_img = self.BNI1(g_img)
        g_img = self.actI1(g_img)

        hg_img =self.image_Fusion(torch.cat((h_img, g_img), dim=1))

        return hg_img.detach()

    def generate_txt_code(self, t):
        h_txt = self.text_module(t)

        f_txt= F.normalize(h_txt, dim=1)
        Matrix_txt = (f_txt.matmul(f_txt.transpose(0, 1)) >= self.beta).type(torch.cuda.FloatTensor)
        g_txt = self.gcnT1(Matrix_txt.mm(h_txt))
        g_txt = self.BNT1(g_txt)
        g_txt = self.actT1(g_txt)

        hg_txt =self.text_Fusion(torch.cat((h_txt, g_txt), dim=1))

        return hg_txt.detach()

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, name=None, path='./checkpoints', cuda_device=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if cuda_device.type == 'cpu':
            torch.save(self.state_dict(), os.path.join(path, name))
        else:
            with torch.cuda.device(cuda_device):
                torch.save(self.state_dict(), os.path.join(path, name))
        return name

    def preprocess(self, A_hat):

        size = len(A_hat)

        I = torch.eye(size).to("cuda:0")

        A_hat = A_hat.clone().detach()

        A = A_hat - I

        degrees = []
        for node_adjaceny in A:
            num = 0
            for node in node_adjaceny:
                if node == 1.0:
                    num = num + 1
            num = num + 1
            degrees.append(num)
        # print(degrees)
        degrees = torch.tensor(degrees).float()
    
        D = torch.diag(degrees.clone().detach())
        # print(D)
        D = torch.linalg.cholesky(D)
        # print(D)
        D_inv = torch.linalg.inv(D).to('cuda:0')
        # print(D)


        return D_inv
    
    def discriminate_hash(self, h):
        return self.hash_dis(h).squeeze()


class GCN(nn.Module):
    def __init__(self , dim_in=20 , dim_out=20, dim_embed = 512):
        super(GCN,self).__init__()

        self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)

        self.out = nn.Linear(dim_out * dim_in, dim_embed)

    def forward(self, A, X):
        batch, objects, rep = X.shape[0], X.shape[1], X.shape[2]

        # first layer
        tmp = (A.bmm(X)).view(-1, rep)
        X = F.relu(self.fc1(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # second layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc2(tmp))
        X = X.view(batch, -1, X.shape[-1])

        # third layer
        tmp = (A.bmm(X)).view(-1, X.shape[-1])
        X = F.relu(self.fc3(tmp))
        X = X.view(batch, -1)

        return l2norm(self.out(X), -1)
    
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
