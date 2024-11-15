import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../optimal_transport')
from typing import Callable, Optional
from backbone import PositionalEncoding, particleTransformer, MLP, RNN, CNN
from emdloss import *
import torch.nn.functional as F
import pytorch_lightning as pl
from emdloss import *
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
import pickle
from geomloss import SamplesLoss
import ot


#PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class ManifoldEmbedder(pl.LightningModule):
    """docstring for ManifoldEmbedder"""
    def __init__(self, data_type, data_npair, backbone_type, learning_rate, modelparams):
        super(ManifoldEmbedder, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.data_npair = data_npair
        if data_type == 'jets':
            if backbone_type == 'MLP':
                self.encoder = MLP(*modelparams)
            elif backbone_type == 'Transformer':
                self.encoder = particleTransformer(3, *modelparams)
            elif backbone_type == 'LSTM':
                self.encoder = RNN('LSTM',*modelparams)
            elif backbone_type == 'GRU':
                self.encoder = RNN('GRU',*modelparams)
            else:
                raise LookupError('only support MLP, Transformer, LSTM and GRU for jets')

        elif data_type == 'MNIST':
            if backbone_type == 'MLP':
                self.encoder = MLP(*modelparams)

            elif backbone_type == 'CNN':
                self.encoder = CNN(*modelparams)

            else:
                raise LookupError('only support MLP and CNN for MNIST')

        else:
            raise LookupError('only support Jets and MNIST dataembedding')



    def forward(self, x):

        embedding = self.encoder(x)
        return embedding


    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "val")
        return loss
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            print('start')
            self.distortion_measure = []
            self.pairwise_ratio = []
            self.original_metric = []
        self._common_step(batch, batch_idx, "test")
        
    
    def _common_step(self, batch, batch_idx, stage: str):
        if self.data_npair == 2:
            x, y, dist = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist = torch.where(dist > 0, dist, torch.tensor(0.001,dtype=torch.float).to(device))
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist = pdist(x_embed,y_embed)

            if stage == 'test':
                loss = (euclidean_dist-dist).abs() / dist.double().abs()
                self.distortion_measure.append(loss)
                loss = (euclidean_dist).abs() / dist.float().abs()
                self.pairwise_ratio.append(loss)
                self.original_metric.append(dist.float().abs())
                
            
            else:
                loss = torch.sum((euclidean_dist - dist.float()).abs() / (dist.float().abs()))/(len(euclidean_dist))


        elif self.data_npair == 3:
            x, y, z,  dist1, dist2, dist3 = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist1 = torch.where(dist1 > 0, dist1, torch.tensor(0.01,dtype=torch.float).to(device))
            dist2 = torch.where(dist2 > 0, dist2, torch.tensor(0.01,dtype=torch.float).to(device))
            dist3 = torch.where(dist3 > 0, dist3, torch.tensor(0.01,dtype=torch.float).to(device))

            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            z_embed = self.encoder(z)

            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist1 = pdist(x_embed,y_embed)
            euclidean_dist2 = pdist(y_embed,z_embed)
            euclidean_dist3 = pdist(z_embed,x_embed)

            loss = (torch.sum((euclidean_dist1 - dist1.float()).abs() / (dist1.float().abs() + 1e-8)+(euclidean_dist2 - dist2.float()).abs() / (dist2.float().abs() + 1e-8)+(euclidean_dist3 - dist3.float()).abs() / (dist3.float().abs() + 1e-8))/(len(euclidean_dist1))) / 3.0

        if stage != 'test':
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        return loss



    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, label = batch
        return self(x), label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        return optimizer


    
class HyperbolicEmbedder(ManifoldEmbedder):
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            print('true')
            nn.init.uniform_(m.weight.data, -self.init_weights, self.init_weights)
    
    def __init__(self, data_type, data_npair, backbone_type, learning_rate, epsilon, init_weights, modelparams):
        super(HyperbolicEmbedder, self).__init__(data_type, data_npair, backbone_type, learning_rate, modelparams)
        self.double()
        self.init_weights = init_weights
        self.epsilon = epsilon
        self.encoder.apply(self.initialize_weights)

    def hypdist(self, u, v):
        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1)
        sqvnorm = torch.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(x ** 2 - 1)

        return torch.log(x + z)
    
    def _common_step(self, batch, batch_idx, stage: str):
        if self.data_npair == 2:
            x, y, dist = batch
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist = dist.type(torch.DoubleTensor).to(device)
            dist = torch.where(dist > 0, dist, torch.tensor(1e-8,dtype=torch.float64).to(device))
            x,y, dist=x.type(torch.DoubleTensor).to(device),y.type(torch.DoubleTensor).to(device), dist.type(torch.DoubleTensor).to(device)

            x_embed = self.encoder(x.double())
            y_embed = self.encoder(y.double())

            hyperbolic_dist = self.hypdist(x_embed,y_embed)

            


            if stage == 'test':
                loss = (hyperbolic_dist-dist).abs() / dist.double().abs()
                self.distortion_measure.append(loss)
                loss = (hyperbolic_dist).abs() / dist.float().abs()
                self.pairwise_ratio.append(loss)
                self.original_metric.append(dist.float().abs())
            else:
                loss = torch.sum(torch.square(hyperbolic_dist - dist.double()))


        elif self.data_npair == 3:
            x, y, z,  dist1, dist2, dist3 = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist1 = torch.where(dist1 > 0, dist1, torch.tensor(0.01,dtype=torch.double).to(device))
            dist2 = torch.where(dist2 > 0, dist2, torch.tensor(0.01,dtype=torch.double).to(device))
            dist3 = torch.where(dist3 > 0, dist3, torch.tensor(0.01,dtype=torch.double).to(device))
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            z_embed = self.encoder(z)

            hyperbolic_dist1 = self.hypdist(x_embed,y_embed)
            hyperbolic_dist2 = self.hypdist(y_embed,z_embed)
            hyperbolic_dist3 = self.hypdist(z_embed,x_embed)

            loss = (torch.sum((hyperbolic_dist1 - dist1.double()).abs() / (dist1.double().abs() + 1e-8)+(hyperbolic_dist2 - dist2.float()).abs() / (dist2.double().abs() + 1e-8)+(hyperbolic_dist3 - dist3.double()).abs() / (dist3.double().abs() + 1e-8))/(len(hyperbolic_dist1))) / 3.0


        if stage != 'test':
            self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True)
        return loss
    
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, label = batch
        return self(x.double()), label

class JetDataset(torch.utils.data.Dataset):
    """It returns pair of jet data X, Y and the target emd(X,Y)"""
    def __init__(self, from_file, data_dir, isToy, jet1_data, jet2_data, num_part):
        super(JetDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet1_data), 'rb') as handle:
                self.jet1_data = pickle.load(handle)

            with open(os.path.join(data_dir,jet2_data), 'rb') as handle:
                self.jet2_data = pickle.load(handle)

        else:
            self.jet1_data = jet1_data
            self.jet2_data = jet2_data
        
        emdcalc = EMDLoss(num_particles=num_part,device='cpu')
        if torch.cuda.is_available():
            emdcalc = EMDLoss(num_particles=num_part,device='cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if isToy:
            self.jet1_data = self.process_data(self.jet1_data, num_part, 3, True)
            self.jet2_data = self.process_data(self.jet2_data, num_part, 3, True)
        else:
            self.jet1_data = self.process_jet_data_all(self.jet1_data, num_part)
            self.jet2_data = self.process_jet_data_all(self.jet2_data, num_part)
        paired_data = torch.utils.data.TensorDataset(self.jet1_data, self.jet2_data)
        dataloader = DataLoader(paired_data, batch_size=5000, shuffle=False)
        emd = torch.zeros(0)
        for x,y in tqdm(dataloader):
            #print(x.shape, y.shape)
            emd = torch.cat((emd.to(device), emdcalc(x.to(device),y.to(device))))
        self.emd = emd.to("cpu").float()

    def process_data(self, dt, num_part, num_feat, doNormalize):
        data = np.copy(dt)
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def process_jet_data_all(self, dt, num_part):
        def fix_phi(phi):
            phi %= (2*np.pi)
            if phi > np.pi:
                phi -= 2*np.pi
            return phi

        def rotate_eig(evt, num_part):
            new = np.copy(evt)
            cov_mat = np.cov(evt[:3*num_part].reshape(-1,3)[:, 1:3], aweights=evt[:3*num_part].reshape(-1,3)[:, 0] , rowvar=False)

            if np.isnan(np.sum(cov_mat)):
                return new
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            idx = eig_vals.argsort()[::1]   
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:,idx]
            new[:3*num_part].reshape(-1,3)[:, 1:3] = np.matmul(evt[:3*num_part].reshape(-1,3)[:, 1:3], eig_vecs)

            return new

        def flip(evt, num_part):
            new = np.copy(evt)
            upper_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,2]>0)
            lower_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,2]<=0)
            upper_sum = np.sum(evt[:3*num_part].reshape(-1,3)[upper_quadrant,0])
            lower_sum = np.sum(evt[:3*num_part].reshape(-1,3)[lower_quadrant,0])
            if lower_sum > upper_sum:
                new[:3*num_part].reshape(-1,3)[:,2] *= -1
            return new

        def flip_eta(evt, num_part):
            new = np.copy(evt)
            right_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,1]>0)
            left_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,1]<=0)
            right_sum = np.sum(evt[:3*num_part].reshape(-1,3)[right_quadrant,0])
            left_sum = np.sum(evt[:3*num_part].reshape(-1,3)[left_quadrant,0])
            if left_sum > right_sum:
                new[:3*num_part].reshape(-1,3)[:,1] *= -1
            return new   

        temp = np.copy(dt)
        pt = temp[:,3*num_part]
        eta = temp[:,3*num_part+1]
        phi = temp[:,3*num_part+2]
        fix_phi_vec = np.vectorize(fix_phi)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,0] /= pt.reshape(-1,1)

        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,1] -= eta.reshape(-1,1)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,2] = fix_phi_vec(temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,2] - phi.reshape(-1,1) )
        temp2 = np.apply_along_axis(rotate_eig, 1, temp, num_part)
        temp3 = np.apply_along_axis(flip, 1, temp2, num_part)
        temp4 = np.apply_along_axis(flip_eta, 1, temp3, num_part)

        return torch.FloatTensor(temp4[:,:3*num_part].reshape(-1, num_part, 3)[:, :, [1, 2, 0]])


    def __len__(self):
        return len(self.emd)

    def __getitem__(self, index):
        return self.jet1_data[index], self.jet2_data[index], self.emd[index]

class JetTripletDataset(torch.utils.data.Dataset):
    """It returns  of jet data X, Y, Z and the target emd(X,Y), emd(Y,Z), emd(Z,X)"""
    def __init__(self, from_file, data_dir, jet1_data, jet2_data, jet3_data, num_part):
        super(JetTripletDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet1_data), 'rb') as handle:
                self.jet1_data = pickle.load(handle)

            with open(os.path.join(data_dir,jet2_data), 'rb') as handle:
                self.jet2_data = pickle.load(handle)

            with open(os.path.join(data_dir,jet3_data), 'rb') as handle:
                self.jet3_data = pickle.load(handle)

        else:
            self.jet1_data = jet1_data
            self.jet2_data = jet2_data
            self.jet3_data = jet3_data

        emdcalc = EMDLoss(num_particles=num_part,device='cpu')
        if torch.cuda.is_available():
            emdcalc = EMDLoss(num_particles=num_part,device='cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.jet1_data = self.process_data(self.jet1_data, num_part, 3, True )
        self.jet2_data = self.process_data(self.jet2_data, num_part, 3, True )
        self.jet3_data = self.process_data(self.jet3_data, num_part, 3, True )
        paired_data_1 = torch.utils.data.TensorDataset(self.jet1_data, self.jet2_data)
        paired_data_2 = torch.utils.data.TensorDataset(self.jet2_data, self.jet3_data)
        paired_data_3 = torch.utils.data.TensorDataset(self.jet3_data, self.jet1_data)
        dataloader_1 = DataLoader(paired_data_1, batch_size=128, shuffle=False)
        dataloader_2 = DataLoader(paired_data_2, batch_size=128, shuffle=False)
        dataloader_3 = DataLoader(paired_data_3, batch_size=128, shuffle=False)
        emd_1 = torch.zeros(0)
        for x,y in tqdm(dataloader_1):
            emd_1 = torch.cat((emd_1.to(device), emdcalc(x.to(device),y.to(device))))

        emd_2 = torch.zeros(0)
        for x,y in tqdm(dataloader_2):
            emd_2 = torch.cat((emd_2.to(device), emdcalc(x.to(device),y.to(device))))


        emd_3 = torch.zeros(0)
        for x,y in tqdm(dataloader_3):
            emd_3 = torch.cat((emd_3.to(device), emdcalc(x.to(device),y.to(device))))



        self.emd_1 = emd_1.to("cpu").float()
        self.emd_2 = emd_2.to("cpu").float()
        self.emd_3 = emd_3.to("cpu").float()


    def process_data(self, data, num_part, num_feat, doNormalize):
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.emd_1)

    def __getitem__(self, index):
        return self.jet1_data[index], self.jet2_data[index], self.jet3_data[index], self.emd_1[index], self.emd_2[index] , self.emd_3[index]

class JetPredictDataset(torch.utils.data.Dataset):
    """docstring for JetPredictDataset"""
    def __init__(self, from_file, data_dir, isToy, jet_data, label_data, num_part):
        super(JetPredictDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet_data), 'rb') as handle:
                self.jet_data = pickle.load(handle)

        else:
            self.jet_data = jet_data

        if isToy:
            self.jet_data = self.process_data(self.jet_data, num_part, 3, True)
        else:
            self.jet_data = self.process_jet_data_all(self.jet_data, num_part)

        
        self.label_data = torch.FloatTensor(label_data)
        
    def process_data(self, data, num_part, num_feat, doNormalize):
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def process_jet_data_all(self, dt, num_part):
        def fix_phi(phi):
            phi %= (2*np.pi)
            if phi > np.pi:
                phi -= 2*np.pi
            return phi

        def rotate_eig(evt, num_part):
            new = np.copy(evt)
            cov_mat = np.cov(evt[:3*num_part].reshape(-1,3)[:, 1:3], aweights=evt[:3*num_part].reshape(-1,3)[:, 0] , rowvar=False)
            if np.isnan(np.sum(cov_mat)):
                return new
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            idx = eig_vals.argsort()[::1]   
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:,idx]
            new[:3*num_part].reshape(-1,3)[:, 1:3] = np.matmul(evt[:3*num_part].reshape(-1,3)[:, 1:3], eig_vecs)

            return new

        def flip(evt, num_part):
            new = np.copy(evt)
            upper_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,2]>0)
            lower_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,2]<=0)
            upper_sum = np.sum(evt[:3*num_part].reshape(-1,3)[upper_quadrant,0])
            lower_sum = np.sum(evt[:3*num_part].reshape(-1,3)[lower_quadrant,0])
            if lower_sum > upper_sum:
                new[:3*num_part].reshape(-1,3)[:,2] *= -1
            return new

        def flip_eta(evt, num_part):
            new = np.copy(evt)
            right_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,1]>0)
            left_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,1]<=0)
            right_sum = np.sum(evt[:3*num_part].reshape(-1,3)[right_quadrant,0])
            left_sum = np.sum(evt[:3*num_part].reshape(-1,3)[left_quadrant,0])
            if left_sum > right_sum:
                new[:3*num_part].reshape(-1,3)[:,1] *= -1
            return new   

        temp = np.copy(dt)
        pt = temp[:,3*num_part]
        eta = temp[:,3*num_part+1]
        phi = temp[:,3*num_part+2]
        fix_phi_vec = np.vectorize(fix_phi)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,0] /= pt.reshape(-1,1)

        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,1] -= eta.reshape(-1,1)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,2] = fix_phi_vec(temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,2] - phi.reshape(-1,1) )
        temp2 = np.apply_along_axis(rotate_eig, 1, temp, num_part)
        temp3 = np.apply_along_axis(flip, 1, temp2, num_part)
        temp4 = np.apply_along_axis(flip_eta, 1, temp3, num_part)

        return torch.FloatTensor(temp4[:,:3*num_part].reshape(-1, num_part, 3)[:, :, [1, 2, 0]])

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        return self.jet_data[index], self.label_data[index]


class JetDataModule(LightningDataModule):
    def __init__(self, file_dict, batch_size :int = BATCH_SIZE):
        super().__init__()
        self.file_dict = file_dict
        self.batch_size = batch_size


    def setup(self, stage: Optional[str] = None):

        if stage in (None, "fit"):
            self.jetpair_train = torch.load(os.path.join(self.file_dict['train']))
            self.jetpair_val = torch.load(os.path.join(self.file_dict['val']))
        if stage == "test":
            self.jetpair_test = torch.load(os.path.join(self.file_dict['test']))
        if stage == "predict":
            self.jetpair_predict = torch.load(os.path.join(self.file_dict['predict']))

    def train_dataloader(self):
        jetpair_train = DataLoader(self.jetpair_train, batch_size=self.batch_size,shuffle=True,num_workers=4)
        return jetpair_train

    def val_dataloader(self):
        jetpair_val = DataLoader(self.jetpair_val, batch_size=self.batch_size,num_workers=4)
        return jetpair_val

    def test_dataloader(self):
        jetpair_test = DataLoader(self.jetpair_test, batch_size=self.batch_size)
        return jetpair_test

    def predict_dataloader(self):
        jetpair_predict = DataLoader(self.jetpair_predict, batch_size=self.batch_size)
        return jetpair_predict






class PrintCallbacks(Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("Trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("Training ended")







