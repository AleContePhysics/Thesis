import torch
from torch import nn, optim
from scipy import special
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial
import os
from tqdm import tqdm
from scipy.stats.qmc import Sobol
from torch.utils.data import TensorDataset
from torch.cuda.amp import autocast, GradScaler
import copy
import ctypes
import sys
import os 
import random

matplotlib.use('Agg') 

MeVfm = 197.3269804440410602 #hbar * c in MeV
PROTONmass  = 938.27208943
NEUTRONmass = 939.565378
mu = PROTONmass * NEUTRONmass / (PROTONmass + NEUTRONmass)

# Use GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if train_on_gpu else "cpu")
print(device)

def define_tensor(x):                                                      # just wraps x into a float64 torch tensor on the right device, nothing more
    return torch.tensor(x, dtype = torch.float64, device=device)
#  Constants for the Minnesota potential [Nucl. Phys. A286, 53 (1977)]

v0r = 200.0;    # MeV
v0t = 178.0;    # MeV
v0s =  91.85;   # MeV
kr  =   1.487;  # fm**-2
kt  =   0.639;  # fm**-2
ks  =   0.465;  # fm**-2

def set_reproducibility(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_reproducibility(10)

def VNN_Minnesota_r(r):
    #  r [fm] is de distance among nucleons
    #  The result is the interaction in MeV
    

    rsq = r*r

    vr_dir =   v0r * torch.exp(-kr*rsq)

    vt_dir = - v0t * torch.exp(-kt*rsq)

    #vs_dir = - v0s * np.exp(-ks*rsq)

    #xV1 = xV2 = xV3 = xV4 = 0.0

    #x1 = (xV1+xV2+xV3+xV4)

    return (vr_dir + vt_dir)


def VNN_Argonne18_r(r):
    
    # AV18 projected on the deuteron channel: gives back the three radial pieces
    # V_SS, V_SD, V_DD (in MeV) of the coupled 3S1-3D1 system
    r = torch.as_tensor(r, dtype=torch.float64)
    r = torch.clamp(torch.abs(r), min=1.0e-4)   # stay away from r=0, the one-pion terms go like 1/r

    hc   = 197.327053
    mpi0 = 134.9739
    mpic = 139.5675
    mp   = 938.27231
    mn   = 939.56563
    alpha = 1.0/137.035989
    mup  =  2.7928474
    mun  = -1.9130427


    mpi = (mpi0 + 2.0*mpic)/3.0
    mu0 = mpi0/hc
    muc = mpic/hc
    mux = mpi/hc
    fsq = 0.075
    cpi = 2.1
    rws = 0.5
    aiws = 5.0
    x  = mux*r
    x0 = mu0*r
    xc = muc*r

    rcut = 1.0 - torch.exp(-cpi*r*r)
    ypi_a = torch.exp(-x)*rcut/x
    tpi   = (1.0 + (3.0 + 3.0/x)/x)*ypi_a*rcut          # average-pion tensor piece, used for tpi2 below
    ypi0  = (mpi0/mpic)**2*(mpi0/3.0)*torch.exp(-x0)*rcut/x0
    tpi0  = (1.0 + (3.0 + 3.0/x0)/x0)*ypi0*rcut
    ypic  = (mpic/3.0)*torch.exp(-xc)*rcut/xc
    tpic  = (1.0 + (3.0 + 3.0/xc)/xc)*ypic*rcut

    ypi0 = fsq*ypi0
    ypic = fsq*ypic
    tpi0 = fsq*tpi0
    tpic = fsq*tpic
    tpi2 = tpi*tpi
    ws   = 1.0/(1.0 + torch.exp((r - rws)*aiws))
    ws0  = 1.0/(1.0 + math.exp(-rws*aiws))
    wsp  = ws*(1.0 + aiws*math.exp(-rws*aiws)*ws0*r)
    wsx  = ws*x
    wsx2 = wsx*x
    dypi00 = (mpi0/mpic)**2*(mpi0/3.0)*cpi/mu0
    dypic0 = (mpic/3.0)*cpi/muc
    ypi0p = ypi0 - fsq*dypi00*ws*r/ws0
    ypicp = ypic - fsq*dypic0*ws*r/ws0

    p11 = ((-7.62701*tpi2 + 1815.4920*wsp + 1847.8059*wsx2 + ypi0p)
         + (-7.62701*tpi2 + 1811.5710*wsp + 1847.8059*wsx2 + ypi0p)
         + (-7.62701*tpi2 + 1813.5315*wsp + 1847.8059*wsx2 - ypi0p + 2*ypicp))/3.0
    pt1 = ((1.07985*tpi2 - 190.0949*wsx - 811.2040*wsx2 + tpi0)
         + (1.07985*tpi2 - 190.0949*wsx - 811.2040*wsx2 + tpi0)
         + (1.07985*tpi2 - 190.0949*wsx - 811.2040*wsx2 - tpi0 + 2*tpic))/3.0
    pls1  = -0.62697*tpi2 - 570.5571*wsp + 819.1222*wsx2
    pl211 =  0.06709*tpi2 + 342.0669*wsp - 615.2339*wsx2
    pls21 =  0.74129*tpi2 +   9.3418*wsp - 376.4384*wsx2
    p10   = -8.62770*tpi2 + 2605.2682*wsp + 441.9733*wsx2 - ypi0p - 2*ypicp
    pt0   =  1.485601*tpi2 - 1126.8359*wsx + 370.1324*wsx2 - tpi0 - 2*tpic
    pls0  =  0.10180*tpi2 +  86.0658*wsp - 356.5175*wsx2
    pl210 = -0.13201*tpi2 + 253.4350*wsp -   1.0076*wsx2
    pls20 =  0.07357*tpi2 - 217.5791*wsp +  18.3935*wsx2
    p01 = ((-11.27028*tpi2 + 3346.6874*wsp - 3*ypi0p)
         + (-11.27028*tpi2 + 3342.7664*wsp - 3*ypi0p)
         + (-10.66788*tpi2 + 3126.5542*wsp - 3*(-ypi0p + 2*ypicp)))/3.0
    pl201 = 0.12472*tpi2 + 16.7780*wsp
    p00   = -2.09971*tpi2 + 1204.4301*wsp - 3*(-ypi0p - 2*ypicp)
    pl200 = -0.31452*tpi2 + 217.4559*wsp

    vnn1  = 0.0625*(9*p11 + 3*p10 + 3*p01 + p00)
    vnn2  = 0.0625*(3*p11 - 3*p10 + p01 - p00)
    vnn3  = 0.0625*(3*p11 + p10 - 3*p01 - p00)
    vnn4  = 0.0625*(p11 - p10 - p01 + p00)
    vnn5  = 0.25*(3*pt1 + pt0)
    vnn6  = 0.25*(pt1 - pt0)
    vnn7  = 0.25*(3*pls1 + pls0)
    vnn8  = 0.25*(pls1 - pls0)
    vnn9  = 0.0625*(9*pl211 + 3*pl210 + 3*pl201 + pl200)
    vnn10 = 0.0625*(3*pl211 - 3*pl210 + pl201 - pl200)
    vnn11 = 0.0625*(3*pl211 + pl210 - 3*pl201 - pl200)
    vnn12 = 0.0625*(pl211 - pl210 - pl201 + pl200)
    vnn13 = 0.25*(3*pls21 + pls20)
    vnn14 = 0.25*(pls21 - pls20)

    # electromagnetic part 
    b = 4.27
    br = b*r
    me = 0.510999
    mr = mp*mn/(mp+mn)
    gamma = 0.577216
    beta = 0.0189
    ftr3   = (1.0 - (1 + br + br**2/2 + br**3/6 + br**4/24 + br**5/144)*torch.exp(-br))/r**3
    flsr3  = (1.0 - (1 + br + br**2/2 + 7*br**3/48 + br**4/48)*torch.exp(-br))/r**3
    fdelta = b**3*(1 + br + br**2/3)*torch.exp(-br)/16.0
    fnpr   = b**3*(15 + 15*br + 6*br**2 + br**3)*torch.exp(-br)/384.0
    vem5  =  alpha*hc*beta*fnpr
    vem8  = -alpha*hc**3*mup*mun*fdelta/(6*mn*mp)
    vem11 = -alpha*hc**3*mup*mun*ftr3/(4*mp*mn)
    vem14 = -alpha*hc**3*mun*flsr3/(2*mn*mr)

    # projection on the 3S1-3D1 channel (l=0, s=1, j=1, t=0, t1z=-1, t2z=+1) 
    s1ds2 = 1     # 4s-3
    t1dt2 = -3    # 4t-3
    vc   = vnn1 + t1dt2*vnn2 + s1ds2*vnn3 + s1ds2*t1dt2*vnn4
    vt   = vnn5 + t1dt2*vnn6
    vls  = vnn7 + t1dt2*vnn8
    vl2  = vnn9 + t1dt2*vnn10 + s1ds2*vnn11 + s1ds2*t1dt2*vnn12
    vls2 = vnn13 + t1dt2*vnn14
    # EM terms 
    vc  = vc + vem5 + s1ds2*vem8
    vt  = vt + vem11
    vls = vls + vem14
    # coupling for j=1: s12 = sqrt(36 j(j+1))/(2j+1) = sqrt(72)/3. This is the tensor
    # matrix element between 3S1 and 3D1: it is what generates the D wave
    s12 = math.sqrt(72.0)/3.0
    V_SS = vc
    V_SD = s12*vt
    V_DD = vc - 2.0*vt - 3.0*vls + 6.0*vl2 + 9.0*vls2
    return V_SS, V_SD, V_DD

# Global parameters
torch.set_float32_matmul_precision('high')
nets = []
E_prev = -2
L = 50
SMALLER_SIZE = 13
SMALL_SIZE = 18
MEDIUM_SIZE = 22
# BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=0)  # fontsize of the figure title

def perturbPoints(grid,t0,tf,sig=0.5):
    # stochastic perturbation of the evaluation points
    # force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]  
    noise = delta_t * torch.randn_like(grid)*sig
    t = grid + noise
    #t.data[2] = torch.ones(1,1)*(-1)
    t.data[t<t0]=t0 - t.data[t<t0]
    t.data[t>tf]=2*tf - t.data[t>tf]
    t.data[0] = torch.ones(1,1)*t0

    t.data[-1] = torch.ones(1,1)*tf
    #t.requires_grad = True
    return t


############## Test for an adaptive learning rate. This does not really work, I'm keeping it here in case something comes from it ############################

class schedule_energy():
    def __init__(self, optimizer, lr_neg_min = 1e-7, lr_neg_max = 1e-6, lr_pos_min = 1e-6, lr_pos_max = 1e-4, shift_up = 0.001, nepochs = 1e5):
        self.optim = optimizer

        self.lrnegmin = lr_neg_min
        self.lrnegmax = lr_neg_max
        self.lrposmin = lr_pos_min
        self.lrposmax = lr_pos_max

        self.shiftpos = shift_up

        self.Tmax = nepochs

        self.Tcurrpos = 0
        self.Tcurrneg = 0

    def cosine_annealing(self, eta, etamin, Tcurr):
        if Tcurr < self.Tmax:
            return etamin + (eta-etamin) * (1+np.cos((Tcurr+1)*np.pi/self.Tmax)) / (1+np.cos(Tcurr*np.pi/self.Tmax))
        else:
            return etamin


    def update(self, E, Elim):
        if E > Elim: # If unstable, increase learning rate
            self.lrposmax = self.cosine_annealing(self.lrposmax, self.lrposmin, self.Tcurrpos)
            self.optim.param_groups[-1]['lr'] = np.min([self.optim.param_groups[-1]['lr'] * (1 + self.shiftpos), self.lrposmax])
            self.Tcurrpos += 1
        else:
            self.lrnegmax = self.cosine_annealing(self.lrnegmax, self.lrnegmin, self.Tcurrneg)
            self.optim.param_groups[-1]['lr'] = self.lrnegmax
            self.Tcurrneg += 1

    def get_last_lr(self):
        return self.optim.param_groups[-1]['lr']

####################################################################################################################


###########
## model ##
###########
# Example of an alternative activation function. TODO: implement ad activation function that is infinitely differentiable but has the properties of ReLU
# class mySin(torch.nn.Module):
#     @staticmethod
#     def forward(input):
#         return torch.sin(input)


########################################################### Neural Network ###############################################


'''
The following class contains the neural network and everything needed
to compute the total loss, the wave function and the energy

'''

class Net(nn.Module):
    def __init__(self, n_iter = 1000, d_hid = 2**8, E_prev = 0, n_points = 2**8, combins = 8):
        super(Net, self).__init__()


        '''
        n_iter: number of iterations before reaching the minimum learning rate
        d_hid: number of neurons in the hidden layers
        E_prev: hyperparameter for the stability loss
        n_points: number of collocation points per batch. Unused
        combins: number of combinations of spin and isospin. Unused here.
        '''

        # Eigenfunction

        self.hidden_layer1 = nn.Linear(1, d_hid)

        self.hidden_layer2 = nn.Linear(d_hid, d_hid)
        
        self.hidden_layer3 = nn.Linear(d_hid, d_hid)
        
        self.hidden_layer4 = nn.Linear(d_hid, d_hid)
        
        self.hidden_layer5 = nn.Linear(d_hid, d_hid)

        self.hidden_layer6 = nn.Linear(d_hid, d_hid)
        
        self.output_layer_1 = nn.Linear(d_hid, 2)   # 2 channels: u_S (3S1) and u_D (3D1)

        # Weights

        # Secondary network for adaptive weights. This does not seem to work and is thus unused, 
        # however the infostructure to utilize it is mantained in case I find a way to use it

        self.hidden_layer_w_1 = nn.Linear(6, d_hid) # 6 losses

        self.hidden_layer_w_2 = nn.Linear(d_hid, d_hid)

        self.hidden_layer_w_3 = nn.Linear(d_hid, d_hid)

        self.hidden_layer_w_4 = nn.Linear(d_hid, d_hid)

        self.output_layer_3 = nn.Linear(d_hid, 5) # 6 weights, ignore energy and orthogonality


        # Nonlinearities for the neural networks. They must be infinitely differentiable

        self.nonlinearity = nn.Tanh()

        self.nonlinearity_W = nn.Tanh()
        





        # Hyperparameters 

        self.E_prev = E_prev
        self.combins = combins

        # Initialize stuff for patience condition


        # Starting values for the patience
        self.patience_norm = 1
        self.patience_bc = 1
        self.patience_eq = 1
        self.patience_grad = 1
        self.patience_grad_int = 1
        self.patience_en = 1


        # How many epochs have passed since the given loss improved

        self.curr_patience_norm = 0
        self.curr_patience_bc = 0
        self.curr_patience_eq = 0
        self.curr_patience_grad = 0
        self.curr_patience_grad_int = 0
        self.curr_patience_en = 0


        # Minimum historical loss

        self.best_norm = 1e9
        self.best_bc = 1e9
        self.best_eq = 1e9
        self.best_grad = 1e9
        self.best_grad_int = 1e9

        self.do_patience_eq = False # If the patience condition should be applied to the differential equation loss from the beginning
        self.max_epochs_pat = 10000 # Maximum number of consecutive epochs in which the patience is increased. This avoids the patience spiraling out of control

        self.tol_reduction = 0.1 # A loss will be considered better the current best one if it is at most 90% it

        # Scale of each loss (the network will not converge if the losses are too diffent from each other, but it is also suggested
        # to balance this weights in order to ensure that the losses converge in order)
        #self.scale_en = 1e6
        self.scale_en = 0

        self.scale_norm = 1.
        self.scale_eq = 1e-4
        self.scale_bc = 1e2
        self.scale_pos = 1e4
        self.scale_stab = 0.0   # weight of loss_stability (the second "energy loss"): kept at 0 here, so only the Schrodinger loss sees E
        self.peso_canale_D = 2.5   # extra weight on the D-wave residual inside loss_diff_eq. Without it the S residual dominates the MSE. Note it enters squared, so the contribution goes like weight^2

        # energy held from the Metropolis estimate. It is the one fed to the Schrodinger loss from epoch 15100 on (initialised at the true value)
        self.E_hold = -2.2245758106075337

        self.tol = 200.


        
        # Parameter that controls the "inertia" of the gradient

        # betas = [0.9999, 0.99999]
        betas = [0.999, 0.9999]
        # betas = [0.99, 0.999]

        self.lr = 1e-4 # learning rate
        self.min_lr_scheduler = 6e-5 # learning rate after n_epochs
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=betas)#, amsgrad = True) # Standard Adam optimizer. Seems to be the best
        

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                            T_max = n_iter, # Maximum number of iterations.
                            eta_min = self.min_lr_scheduler) 
        # self.scheduler = schedule_energy(self.optimizer, lr_neg_min = lr_scheduler_neg_min, lr_neg_max=lr_scheduler_neg_max, lr_pos_min= lr_scheduler_pos_min, lr_pos_max =  lr_scheduler_pos_max, shift_up= shift_scheduler_up, nepochs=n_iter)

        # Default criterion for losses. This utilizes Sum of Squared Errors
        self.criterion = nn.MSELoss(reduction='sum')

    def metropolis(self, x0):
        # one Metropolis step: propose x1 = x0 + gaussian noise and accept it with
        # probability |u(x1)|^2 / |u(x0)|^2, summed over the two channels.
        # Proposals falling outside [0, L] are rejected right away
        with torch.no_grad():
            x0 = x0.to(device)
            L=50
            sigma = 2
            epsilon = torch.randn_like(x0)
            x1 = x0 + epsilon * sigma

            if x1>=L or x1<=0:
                x_new = x0
            else:
                psi_old = self.forward(x0.unsqueeze(1))*x0.unsqueeze(1)   # [1,2]: u_S, u_D
                psi_new = self.forward(x1.unsqueeze(1))*x1.unsqueeze(1)
                p = (psi_new**2).sum() / ((psi_old**2).sum() + 1e-10)     # |u_S|^2 + |u_D|^2
                r = torch.rand_like(p)
                x_new = torch.where(r <= p, x1, x0)
            
        return x_new.cpu()

    def initialize(self): # Use kaiming initialization for the starting weights
        nl = 'tanh'
        mode = 'fan_in'
        torch.nn.init.kaiming_normal_(self.hidden_layer1.weight, mode=mode, nonlinearity=nl)
        torch.nn.init.kaiming_normal_(self.hidden_layer2.weight, mode=mode, nonlinearity=nl)
        torch.nn.init.kaiming_normal_(self.hidden_layer3.weight, mode=mode, nonlinearity=nl)
        torch.nn.init.kaiming_normal_(self.hidden_layer4.weight, mode=mode, nonlinearity=nl)
        torch.nn.init.kaiming_normal_(self.hidden_layer5.weight, mode=mode, nonlinearity=nl)
        torch.nn.init.kaiming_normal_(self.hidden_layer6.weight, mode=mode, nonlinearity=nl)
        torch.nn.init.kaiming_normal_(self.output_layer_1.weight, mode=mode, nonlinearity=nl)

    def init_weights(self): # Use kaiming initialization for the starting weights
        nl = 'tanh'
        mode = 'fan_in'
        torch.nn.init.xavier_uniform_(self.hidden_layer_w_1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_w_2.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_w_3.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_w_4.weight)
        torch.nn.init.xavier_uniform_(self.output_layer_3.weight)




    def forward(self, x):    # computes psi(x): takes x of shape [nbatch,1] and returns psi of shape [nbatch,2]

        # Forward method for the neural network

        x = x#/L

        layer_out_first = self.nonlinearity(self.hidden_layer1(x))
        #layer_out_first = self.nonlinearity(self.hidden_layer1(torch.cat((x,E),1)))
        
        layer_out_first = self.nonlinearity(self.hidden_layer2(layer_out_first))

        layer_out_first = self.nonlinearity(self.hidden_layer3(layer_out_first))

        layer_out_first = self.nonlinearity(self.hidden_layer4(layer_out_first))

        layer_out_first = self.nonlinearity(self.hidden_layer5(layer_out_first))

        layer_out_first = self.nonlinearity(self.hidden_layer6(layer_out_first))

        psi = self.output_layer_1(layer_out_first)


        return psi


# Unused
    
    def forward_weights(self, l):

        layer_out_third = self.nonlinearity_W(self.hidden_layer_w_1(l))
        
        layer_out_third = self.nonlinearity_W(self.hidden_layer_w_2(layer_out_third))

        layer_out_third = self.nonlinearity_W(self.hidden_layer_w_3(layer_out_third))

        layer_out_third = self.nonlinearity_W(self.hidden_layer_w_4(layer_out_third))

        W = self.output_layer_3(layer_out_third)

        return torch.exp(W)

    def diff_eq(self, x, psi, E, Hpsi):

        # The Schrodinger equation
        
        return Hpsi - E * psi
    



    def train_step(self, x, V, x_energy, E_prev = 0, n_points = 2**4):
        '''
        This is the main function for the neural network.
        This computes the losses and performs the training given the collocation points
        and the potential at those points
        '''

        self.optimizer.zero_grad()
        outputs = self.forward(x)
        psi = outputs * torch.abs(x) # [N,2]: u_S, u_D (= R*r for each channel)
        u_S = psi[:, 0:1]; u_D = psi[:, 1:2]

        ########### Compute energy

        # kinetic term: second derivatives channel by channel (ones_like on one channel at a time)
        u_S_x  = torch.autograd.grad(u_S, x, torch.ones_like(u_S), create_graph=True, retain_graph=True, materialize_grads=True)[0]
        u_S_xx = torch.autograd.grad(u_S_x, x, torch.ones_like(u_S_x), create_graph=True, retain_graph=True, materialize_grads=True)[0]
        u_D_x  = torch.autograd.grad(u_D, x, torch.ones_like(u_D), create_graph=True, retain_graph=True, materialize_grads=True)[0]
        u_D_xx = torch.autograd.grad(u_D_x, x, torch.ones_like(u_D_x), create_graph=True, retain_graph=True, materialize_grads=True)[0]

        # coupled AV18 potential (3 components)
        V_SS, V_SD, V_DD = VNN_Argonne18_r(x)
        C = (MeVfm ** 2)/(2 * mu)

        # coupled 3S1-3D1 Hamiltonian (only the D channel carries the 6/r^2 centrifugal barrier)
        Hu_S = -C * u_S_xx + V_SS * u_S + V_SD * u_D
        Hu_D = -C * u_D_xx + C * 6.0/(x**2 + 1e-12) * u_D + V_SD * u_S + V_DD * u_D
        Hpsi = torch.cat([Hu_S, Hu_D], dim=1)   # [N,2] -> norm and residual already work on both channels

        # Compute energy with a finite difference method
        spacing = torch.diff(x.squeeze(1))
        weights_int_1 = torch.cat((spacing, define_tensor([0])))
        weights_int_2 = torch.cat((define_tensor([0]), spacing))
        weights_int = ((weights_int_1 + weights_int_2)/2).unsqueeze(1)

        norm = torch.sum((psi) ** 2 * weights_int) 

        E_true=-2.2245758106075337

        """if epoch_tot <1000:
            E = torch.tensor(0.0, dtype=psi.dtype, device=psi.device)

        elif epoch_tot<4000:
            alpha = (epoch_tot - 1000) / (3000)
            E = torch.tensor(alpha * E_true, dtype=psi.dtype, device=psi.device)"""

        # === Energy estimated on the Metropolis points (Rayleigh local energy, every 200 epochs) ===
        #   E_loc = (u_S H u_S + u_D H u_D) / (u_S^2 + u_D^2), averaged over the grid_energy points (~|psi|^2)
        E_metropolis = float('nan')
        if epoch_tot % 200 == 0:
            psi_e = self.forward(x_energy) * torch.abs(x_energy)        # [M,2]
            uS_e = psi_e[:, 0:1]; uD_e = psi_e[:, 1:2]
            uS_e_x  = torch.autograd.grad(uS_e, x_energy, torch.ones_like(uS_e),  create_graph=True, retain_graph=True, materialize_grads=True)[0]
            uS_e_xx = torch.autograd.grad(uS_e_x, x_energy, torch.ones_like(uS_e_x), create_graph=True, retain_graph=True, materialize_grads=True)[0]
            uD_e_x  = torch.autograd.grad(uD_e, x_energy, torch.ones_like(uD_e),  create_graph=True, retain_graph=True, materialize_grads=True)[0]
            uD_e_xx = torch.autograd.grad(uD_e_x, x_energy, torch.ones_like(uD_e_x), create_graph=True, retain_graph=True, materialize_grads=True)[0]
            V_SS_e, V_SD_e, V_DD_e = VNN_Argonne18_r(x_energy)
            C_en = (MeVfm ** 2)/(2 * mu)
            HuS_e = -C_en * uS_e_xx + V_SS_e * uS_e + V_SD_e * uD_e
            HuD_e = -C_en * uD_e_xx + C_en * 6.0/(x_energy**2 + 1e-12) * uD_e + V_SD_e * uS_e + V_DD_e * uD_e
            E_loc = (uS_e * HuS_e + uD_e * HuD_e) / (uS_e**2 + uD_e**2 + 1e-12)   # coupled local energy (Rayleigh)
            E_metropolis = torch.mean(E_loc[1:-1]).item()                         # average over the ~|psi|^2 points
            beta_E = 0.3   # how much the new measurement counts: a moving average with inertia, to damp the noise of the estimate
            self.E_hold = beta_E * E_metropolis + (1.0 - beta_E) * self.E_hold    # moving average: the E handed to the loss comes out smooth, without sudden jumps

        # === Energy that goes into the Schrodinger loss ===
        #  - before epoch 15100: the true value E_true, to give the network time to settle
        #  - from 15100 on: the value estimated on the Metropolis points (self.E_hold), held fixed for 200 epochs
        if epoch_tot < 15100:
            E = torch.tensor(E_true, dtype=psi.dtype, device=psi.device)
        else:
            E = torch.tensor(self.E_hold, dtype=psi.dtype, device=psi.device)

        #Losses:

        zeros_diff_eq = torch.zeros(size=(len(x), 1), requires_grad=False, dtype=torch.float64).to(device)

        # Differential equation loss. Note that it weighted in order to give more importance to the parts closer to 0

        f_out = torch.abs(self.diff_eq(x, psi, E, Hpsi))*(L/(1e-6+torch.abs(x)))   # [N,2]: residual of the two channels
        f_out = f_out * torch.tensor([1.0, self.peso_canale_D], dtype=f_out.dtype, device=f_out.device)   # the D channel (col. 1) is weighted more, otherwise the S residual dominates the MSE
        loss_diff_eq = self.criterion(f_out, torch.zeros_like(f_out))

        # Normalization loss and boundary conditions loss
        if epoch_tot<10000:
            extremes_r = psi[(torch.abs(x) >= L).nonzero(as_tuple=True)[0]]#/integ_final #Last 1%%
        else :
            extremes_r = psi[(torch.abs(x) >= 17*L/20).nonzero(as_tuple=True)[0]]



        loss_bc = self.criterion(extremes_r, torch.zeros_like(extremes_r))*len(psi)
        
        loss_norm = (norm - torch.log(norm) - 1) ** 2

        loss_pos = self.criterion(torch.relu(-psi), torch.zeros_like(psi))   # positivity on both channels (u_S and u_D >= 0)



        # Two parts of the stability loss

        exp_en = 8e-1
        loss_en = torch.exp(exp_en * (E - E_prev))
        loss_stability = torch.max(define_tensor(0.),(1e4*(E+0.1)))

            
            

        # This is where weights would be computed if they did not make the training worse

        # losses_all = torch.cat((loss_norm * self.scale_norm * self.patience_norm, loss_diff_eq.unsqueeze(0) * self.patience_eq * self.scale_eq, loss_bc.unsqueeze(0) * self.patience_bc * self.scale_bc, loss_integ.unsqueeze(0) * self.patience_integ * self.scale_integ, ss_loss.unsqueeze(0) * self.patience_simm * self.scale_simm, (loss_en * self.scale_en).unsqueeze(0))).to(device)
        # weights = self.forward_weights(losses_all/torch.max(losses_all))
        weights = torch.ones(5, dtype=torch.float64, device=device)



        # Compute the total loss with the relevant weights

        loss_total = loss_norm * self.scale_norm * self.patience_norm * weights[0] + loss_diff_eq * self.scale_eq * self.patience_eq * weights[1] + loss_bc * self.patience_bc * self.scale_bc * weights[2] + loss_en * self.scale_en  + loss_stability * self.patience_en * self.scale_stab + loss_pos * self.scale_pos
        
        

        # Here we compute the backpropagation
        loss_total.backward()
        self.optimizer.step()
        if self.scheduler.get_last_lr()[0] != self.min_lr_scheduler:
            self.scheduler.step()



        # Patience conditon: increase weights for the losses that are not improving


        patience_before_increase = 300 # Number of epochs before the patience increases
        self.scale_en = self.scale_en * 0.9999 # You want to force the first part of the stability loss to go to 0


         ###### Normalization ######
        if loss_norm < self.best_norm - (self.best_norm * self.tol_reduction) or loss_norm <= self.tol or self.curr_patience_norm > self.max_epochs_pat: # If the loss improves, reset patience
             self.curr_patience_norm = 0
             #self.patience_norm = np.max([self.patience_norm-1,1])
             self.best_norm = loss_norm
        
        else:
             self.curr_patience_norm += 1
             if self.curr_patience_norm > patience_before_increase: # Give 100 epochs for the loss to improve, then start increasing the weight
                 self.patience_norm += 1



        # ###### Boundary conditions ######
        if loss_bc < self.best_bc - (self.best_bc * self.tol_reduction) or loss_bc <= self.tol or self.curr_patience_bc > self.max_epochs_pat:
             self.curr_patience_bc = 0
             #self.patience_bc = np.max([self.patience_bc-1,1])
             self.best_bc = loss_bc
        
        else:
             self.curr_patience_bc += 1
             if self.curr_patience_bc > patience_before_increase:
                 self.patience_bc += 1

   
        #### Once all other losses are close to convergence, start using patience also for the equation ####

        if self.do_patience_eq:
            if loss_diff_eq < self.best_eq - (self.best_eq * self.tol_reduction) or loss_diff_eq <= self.tol:
                    self.curr_patience_eq = 0
                    #self.patience_eq = np.max([self.patience_eq-1,1])
                    self.best_eq = loss_diff_eq
            
            else:
                    self.curr_patience_eq += 1
                    if self.curr_patience_eq > patience_before_increase:
                        self.patience_eq += 1
            
            if E < -0.05:
                self.curr_patience_en = 0
            else:
                self.curr_patience_en += 1
                if self.curr_patience_en > patience_before_increase:
                    self.patience_en += 1   
            
        
        else:
             if (loss_norm <= self.tol*100).long() + (loss_bc <= self.tol*100).long() == 2:
                 self.do_patience_eq = True
                 self.best_eq = loss_diff_eq
        

        return loss_total.item(), loss_norm.item(), loss_diff_eq.item(), loss_bc.item(), (loss_en*self.scale_en).item(), E.item(), weights, self.optimizer.param_groups[-1]['lr'], norm, E_metropolis


##############
## TRAINING ##
##############
# def main(): 

min_loss = np.inf
min_loss_eq = np.inf
last_plotted_epoch = -np.inf
loss = np.inf
losseq = np.inf
best_norm = 1.

# Make folders to store the results
os.makedirs("images",exist_ok=True)
os.makedirs("images/plots",exist_ok=True)
os.makedirs("model",exist_ok=True)

# Number of epochs
n_epochs = 15000
n_epochs_batch = 100000000 # To generate a new batch. At the moment multiple batches seem to lead to worse performance. It is proably due to the finite difference method
n_points = 2 ** 10
n_batches = 1
n_relax=50
n_burn_in=2000


############# Prepare metrics #################

# losses history
loss_history = np.array([])
# fid_history = np.array([])
loss_norm_history = np.array([])
losseq_history = np.array([])
lossbc_history = np.array([])
lossor_history = np.array([])
en_min_history = np.array([])
grad_history = np.array([])
grad_int_history = np.array([])

# Weights history (unused)

weight_loss_norm_history = np.array([])
weight_losseq_history = np.array([])
weight_lossbc_history = np.array([])
weight_lossor_history = np.array([])
weight_grad_history = np.array([])
weight_grad_int_history = np.array([])


# Patience history

patience_lossbc_history = np.array([])
patience_loss_norm_history = np.array([])
patience_losseq_history = np.array([])
patience_grad_history = np.array([])
patience_grad_int_history = np.array([])

# Energy history

E_history = np.array([])
E_metropolis_history = np.array([])   # Metropolis energy estimate: one point every 200 epochs
E_metropolis_epochs  = np.array([])   # matching epochs (x axis of the plot)


# Two separate sets of points: grid_loss is where the residual and the constraints are
# imposed, grid_energy is where the energy is estimated. Both start uniform and are
# replaced by the Metropolis samples once the sampling kicks in
grid_loss = torch.linspace(start = 0, end = L, steps = n_points, dtype = torch.float64)
grid_energy = torch.linspace(start = 0, end = L, steps = n_points, dtype = torch.float64)
ancore = torch.linspace(0, L, round(0.5* n_points), dtype=torch.float64)   # uniform fixed anchors (half of n_points): they cover the whole domain, tail included

# from here on both grids get rebuilt by Metropolis inside the training loop 

# Define network
net = Net(n_iter = n_epochs, d_hid= 2 ** 7, E_prev=E_prev, n_points=n_points, combins=0)

net.double()        
net.initialize()
net.to(device)
batch_num = 0
epoch_tot = 0  

x0 = torch.tensor([10.0], dtype=torch.float64)

while loss > 100 or losseq > 400: # One of the two stopping conditions

    all_batched = []
    V = torch.zeros((n_points), dtype = torch.float64)
    print('Generating V...')
    V = VNN_Minnesota_r(grid_loss)
    print('Generated V')
    all_batched = all_batched + [(grid_loss, V)]
    batch_num += 1
    prog_bar = tqdm(range(n_epochs_batch), total=n_epochs_batch)

    for epoch in prog_bar:

        # Here the training happens
        epoch_tot+=1

        # Every 1000 epochs, and only after 6000 (before that the wave function is still
        # meaningless and sampling it would be pointless), the collocation points are
        # regenerated with Metropolis: burn-in first, then one point kept every n_relax
        # steps so that consecutive points are not too correlated
        if epoch_tot % 1000 == 0 and epoch_tot >= 6000 :
    
    
            lista_copia = []
            for i in range(n_burn_in):
                x0 = net.metropolis(x0)
            print("Burn-In Terminato")
                
            for i in range(n_points * n_relax):
                x0 = net.metropolis(x0)
                if i % n_relax == 0:
                    lista_copia.append(x0.clone())
            print("Griglia Completata")

                
                
            grid_loss_copia = torch.stack(lista_copia).view(-1)
            grid_loss_copia_ordinata = torch.sort(grid_loss_copia)[0]
            inizio = torch.tensor([0.0], dtype=grid_loss_copia.dtype, device=grid_loss_copia.device)
            fine = torch.tensor([50.0], dtype=grid_loss_copia.dtype, device=grid_loss_copia.device)
            grid_energy = torch.cat([inizio, grid_loss_copia_ordinata, fine])              # only |u|^2 points (used for the energy)
            grid_loss = torch.sort(torch.cat([ancore, grid_loss_copia_ordinata]))[0]     # fixed anchors + |u|^2 points (used for the losses)

            n_relax=20
            n_burn_in=500

            # Generate the potential

            V = torch.zeros((n_points), dtype = torch.float64)
            print('Generating V...')
            V = VNN_Minnesota_r(grid_loss)
            print('Generated V')
            all_batched = all_batched + [(grid_loss, V)]
            batch_num += 1

            # local style, used only for these histograms
            stile_locale = {
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "figure.figsize": (6, 4),    # plot size
            }

            # the 'with' block keeps the style change local
            with plt.style.context(stile_locale):
                fig, ax = plt.subplots()

                # histogram of the sampled points
                ax.hist(
                    grid_loss.cpu().detach().numpy(), 
                    bins=80, 
                    color='#2b5c8f',
                    edgecolor='#1d3f63',
                    linewidth=0.5,
                    alpha=0.8,
                    density=True,
                    zorder=3
                )  

                ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc', zorder=0)
                ax.set_axisbelow(True) 

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # axis labels as raw strings, so the LaTeX math renders
                ax.set_xlabel(r'$r$ (fm)', fontsize=12)
                ax.set_ylabel(r'Probability Density $P(r)$', fontsize=12)

                ax.set_title(f'Distribution r_i loss — epoch {epoch_tot}', fontsize=12, pad=10)

                plt.tight_layout()
                
                # save as png, switch the extension to .pdf if you want vector output for LaTeX
                plt.savefig(f'images/plots/dist_loss_{epoch_tot}.png', bbox_inches='tight', dpi=300)
                plt.close()

            with plt.style.context(stile_locale):
                fig, ax = plt.subplots()

                # histogram of the sampled points
                ax.hist(
                    grid_energy.cpu().detach().numpy(), 
                    bins=80, 
                    color="#ce1313",
                    edgecolor="#990707",
                    linewidth=0.5,
                    alpha=0.8,
                    density=True,
                    zorder=3
                )  

                ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc', zorder=0)
                ax.set_axisbelow(True) 

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # axis labels as raw strings, so the LaTeX math renders
                ax.set_xlabel(r'$r$ (fm)', fontsize=12)
                ax.set_ylabel(r'Probability Density $P(r)$', fontsize=12)

                ax.set_title(f'Distribution energy $r_i$ — epoch {epoch_tot}', fontsize=12, pad=10)

                plt.tight_layout()
                
                # save as png, switch the extension to .pdf if you want vector output for LaTeX
                plt.savefig(f'images/plots/dist_energy_{epoch_tot}.png', bbox_inches='tight', dpi=300)
                plt.close()
        
        # batch = all_batched[epoch%n_batches]
        batch = all_batched[-1]
        x = batch[0].to(device)
        #x = x/L
        x.requires_grad = True

        V = batch[1].to(device)
        V.requires_grad = True

        
        net.train()
        # Train step. Pass training points as a single batch
        x_energy = grid_energy.to(device).unsqueeze(1)
        x_energy.requires_grad = True
        loss_nn, loss_norm, losseq, lossbc, en_min, E_est, weights, lr, norm, E_metropolis = net.train_step(x = x.unsqueeze(1),V = V, x_energy = x_energy, E_prev=E_prev, n_points= n_points)#, indexes_swap = indexes_swap)
        # update the loss values beside the progress bar and other various metrics for each iteration

        loss = loss_norm * weights[0] * net.scale_norm + losseq * weights[1] * net.scale_eq + lossbc * weights[2] * net.scale_bc  #+ loss_grad_int * net.scale_grad_int + en_min #+ torch.sum(1/weights**2) 
        loss = loss.item()

        prog_bar.set_description(desc=f"Epoch:{epoch_tot}, batch:{batch_num}, Loss: {loss:.4E},  loss_best: {min_loss:.4E}, loss_nn: {loss_nn:.4E}, loss_norma: {loss_norm:.4E}, loss_eq: {losseq:.4E}, loss_bc: {lossbc:.3E}, lr: {lr:.3E}, E(MeV): {(E_est):.4E}")

        # Collect metrics
        
        loss_history = np.append(loss_history, loss)

        loss_norm_history = np.append(loss_norm_history, loss_norm)
        losseq_history = np.append(losseq_history, losseq)
        lossbc_history = np.append(lossbc_history, lossbc)

        en_min_history = np.append(en_min_history, en_min)

 

        weight_loss_norm_history = np.append(weight_loss_norm_history, weights[0].item())
        weight_losseq_history = np.append(weight_losseq_history, weights[1].item())
        weight_lossbc_history = np.append(weight_lossbc_history, weights[2].item())

        patience_lossbc_history = np.append(patience_lossbc_history,net.patience_bc)
        patience_loss_norm_history = np.append(patience_loss_norm_history,net.patience_norm)
        patience_losseq_history = np.append(patience_losseq_history,net.patience_eq)
        patience_grad_history = np.append(patience_grad_history, net.patience_grad)

        
        E_history = np.append(E_history, E_est)

        # Store and print the Metropolis energy every 200 epochs. This history is only for the
        # plots: the value that actually enters the loss is self.E_hold, kept inside the net
        if epoch_tot % 200 == 0:
            E_metropolis_history = np.append(E_metropolis_history, E_metropolis)
            E_metropolis_epochs  = np.append(E_metropolis_epochs, epoch_tot)
            print(f"[Metropolis] epoch {epoch_tot}: E = {E_metropolis:.5f} MeV ")



        # Update the best model if needed
        if loss < 1000: # Set a minimum requirement for a successful model
            if losseq < min_loss_eq:
                best = net.state_dict()
                torch.save(best, "model/best_model.pth")
                best_epoch = epoch
                min_loss_eq = losseq
            if loss < 100 and losseq < 400 and epoch_tot>=19500:
                break

        # Plot the partial best model. 
        if loss < min_loss:
            min_loss = loss
            best_norm = norm
        if epoch_tot - last_plotted_epoch >= 799:
            last_plotted_epoch = epoch_tot
            net.eval()

            ##### Total loss #####
            plt.semilogy(loss_history)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.title('Loss history')
            plt.savefig('images/loss.png', bbox_inches='tight')
            plt.clf()
            np.save('images/loss_history', loss_history)


            ##### Partial losses #####
            plt.semilogy(loss_norm_history, color = 'orange')
            plt.semilogy(lossbc_history, color = 'darkgoldenrod')
            plt.semilogy(losseq_history, color = 'black')
            plt.hlines(400, color="blue", xmin=0, xmax=len(losseq_history), linestyles='dashed')

            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Normalization', 'BCs', 'Equation', 'Convergence Line'])
            plt.title('Partial losses history')
            plt.savefig('images/partial_losses.png', bbox_inches='tight')
            plt.clf()

            ##### Partial losses weighted #####
            plt.semilogy(loss_norm_history * net.scale_norm * weight_loss_norm_history * patience_loss_norm_history, color = 'orange')
            plt.semilogy(lossbc_history * net.scale_bc * weight_lossbc_history * patience_lossbc_history, color = 'darkgoldenrod')
            plt.semilogy(losseq_history * net.scale_eq * weight_losseq_history * patience_losseq_history, color = 'black')
            plt.hlines(400, color="blue", xmin=0, xmax=len(losseq_history), linestyles='dashed')

            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Normalization', 'BCs', 'Equation', 'Convergence Line'])
            plt.title('Partial losses history (weighted)')
            plt.savefig('images/partial_losses_weighted.png', bbox_inches='tight')
            plt.clf()


            #### Weights #####
            plt.semilogy(weight_loss_norm_history, color = 'orange')
            plt.semilogy(weight_lossbc_history, color = 'darkgoldenrod')
            plt.semilogy(weight_losseq_history, color = 'black')

            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Normalization', 'BCs', 'Equation'])
            plt.title('Weights history')
            plt.savefig('images/partial_losses_weights.png', bbox_inches='tight')
            plt.clf()
            

            ##### Energy #####
            plt.plot(E_history,label="NN Value")
            plt.hlines(-2.2245758106075337, xmin=0, xmax=len(E_history),color="red",label="Real Value")
            plt.ylabel('E')
            plt.ylim(-4,1)
            plt.legend()
            plt.xlabel('Epoch')
            plt.title('Energy history')
            plt.savefig('images/Energy.png', bbox_inches='tight')
            plt.clf()

            ##### Energy estimated on the Metropolis points (one point every 200 epochs) #####
            # Single plot: the same file gets overwritten every time this block runs (~every 800 epochs)
            # It starts at epoch 6000: before that the sampling is off and the estimate means nothing
            maschera_E = E_metropolis_epochs >= 6000
            plt.plot(E_metropolis_epochs[maschera_E], E_metropolis_history[maschera_E], marker='o', ms=3, color='dodgerblue', label='Energia Metropolis')
            plt.hlines(-2.2245758106075337, xmin=6000, xmax=max(epoch_tot, 6001), color="red", label="Valore Reale")
            plt.ylabel('E (MeV)')
            plt.ylim(-2.6, 1)
            plt.legend()
            plt.xlabel('Epoca')
            plt.title('Energia Metropolis')
            plt.savefig('images/Energy_metropolis.png', bbox_inches='tight')
            plt.clf()

            ##### Patiences ####

            plt.semilogy(patience_loss_norm_history, color = 'orange')
            plt.semilogy(patience_lossbc_history, color = 'darkgoldenrod')
            plt.semilogy(patience_losseq_history, color = 'black')

            plt.ylabel('Patience')
            plt.xlabel('Epoch')
            plt.legend(['Normalization', 'BCs', 'Equation'])
            plt.title('Patience history')
            plt.savefig('images/patience.png', bbox_inches='tight')
            plt.clf()



            ##### Wafeunction #####

            output = net(x.unsqueeze(1))                              # [N,2]
            uS_plot = (output[:, 0] * torch.abs(x)).cpu().detach().numpy()
            uD_plot = (output[:, 1] * torch.abs(x)).cpu().detach().numpy()
            r_plot  = batch[0].detach().numpy()
            V_SS_plot = VNN_Argonne18_r(x.unsqueeze(1))[0].squeeze(1).cpu().detach().numpy()   # V_3S1 (AV18)
            V_scaled  = V_SS_plot/(np.max(np.abs(V_SS_plot))) * np.max(np.abs(uS_plot))        # rescaled just for display
            plt.scatter(r_plot, uS_plot, color='red',  s=0.1)
            plt.scatter(r_plot, uD_plot, color='blue', s=0.1)
            plt.plot(r_plot, V_scaled, color='green', linewidth=0.8)
            plt.ylabel('u(r) / V (scal.)')
            plt.grid(True)
            plt.xlabel('r(fm)')
            plt.title('Wf, epoch: ' + str(epoch_tot) + ' E = ' + str(E_est))
            plt.legend(['u_S', 'u_D', 'V_SS (scaled)'])
            plt.savefig('images/plots/wf_' + str(epoch_tot) + '_E_' + str(E_est)+'_Loss_'+str(loss)+'.png', bbox_inches='tight')

            plt.clf()


            ##### u_S(r) and u_D(r) on a uniform grid #####
            r_unif = torch.linspace(0, L, 1000, dtype=torch.float64, device=device)
            out_unif = net(r_unif.unsqueeze(1))                       # [1000,2]
            uS_unif = (out_unif[:, 0] * r_unif).cpu().detach().numpy()
            uD_unif = (out_unif[:, 1] * r_unif).cpu().detach().numpy()
            r_unif_np = r_unif.cpu().detach().numpy()
            plt.plot(r_unif_np, uS_unif, color='red', label='u_S')
            plt.plot(r_unif_np, uD_unif, color='blue',    label='u_D')
            plt.grid(True)
            plt.ylabel('u(r)')
            plt.xlabel('r (fm)')
            plt.legend()
            plt.title('u(r), epoch: ' + str(epoch_tot))
            plt.savefig('images/plots/u_' + str(epoch_tot) + '.png', bbox_inches='tight')
            plt.clf()


            ##### Zoom of u_S(r) and u_D(r) on 0-5 fm, to look at the l=2 onset of the D wave #####
            r_zoom = torch.linspace(0, 5, 600, dtype=torch.float64, device=device)   # dense grid only at short range
            out_zoom = net(r_zoom.unsqueeze(1))                       # [600,2]
            uS_zoom = (out_zoom[:, 0] * r_zoom).cpu().detach().numpy()
            uD_zoom = (out_zoom[:, 1] * r_zoom).cpu().detach().numpy()
            r_zoom_np = r_zoom.cpu().detach().numpy()
            plt.plot(r_zoom_np, uS_zoom, color='red', label='u_S')
            plt.plot(r_zoom_np, uD_zoom, color='blue', label='u_D')
            plt.grid(True)
            plt.ylabel('u(r)')
            plt.xlabel('r (fm)')
            plt.xlim(0, 5)
            plt.legend()
            plt.title('u(r) zoom 0-5 fm, epoch: ' + str(epoch_tot))
            plt.savefig('images/plots/u_zoom_' + str(epoch_tot) + '.png', bbox_inches='tight')
            plt.clf()



######## Final plots after convergence (also see fix_plots.py to see better versions of them) #######

net.eval()

# ===== Quadrupole moment of the final solution =====
# Q = (1/20) * integral[ r^2 * u_D (sqrt(8) u_S - u_D) ] dr, computed on the normalized psi.
# It is dominated by the u_S*u_D interference term, and the norm has to be 1 because Q scales
# with the amplitude. Same formula as the reference solver, which gives +0.2696
with torch.no_grad():
    r_q = torch.linspace(0, L, 4000, dtype=torch.float64, device=device)
    out_q = net(r_q.unsqueeze(1))
    uS_q = (out_q[:, 0] * r_q).cpu().numpy()
    uD_q = (out_q[:, 1] * r_q).cpu().numpy()
    r_q_np = r_q.cpu().numpy()
dr_q = r_q_np[1] - r_q_np[0]
norm_q = np.sum(uS_q**2 + uD_q**2) * dr_q
uS_q = uS_q / np.sqrt(norm_q)
uD_q = uD_q / np.sqrt(norm_q)
Q_d = (1.0/20.0) * np.sum(r_q_np**2 * uD_q * (np.sqrt(8.0)*uS_q - uD_q)) * dr_q
print("=" * 56)
print(f"  Momento di quadrupolo Q_d = {Q_d:+.4f} fm^2")
print(f"  atteso: +0.270 ")
print("=" * 56)

os.makedirs('images' + "/final",exist_ok=True)
np.save('images/final/loss_history', loss_history)

os.makedirs('images' + "/final"+'/partial_losses',exist_ok=True)
np.save('images/final/partial_losses/loss_norm_history', loss_norm_history)
np.save('images/final/partial_losses/lossbc_history', lossbc_history)
np.save('images/final/partial_losses/losseq_history', losseq_history)

scientific_style = {
    "font.family": "serif",      # serif font, to match the LaTeX document
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 4.2),  # same size for every plot
}

# same colours for the same quantities across all the plots
color_norm = "#c11d0b"
color_bc = "#1a83b0"
color_eq = '#222222'


with plt.style.context(scientific_style):
    
    # ------------------------------------------------------------------
    # 1. Total Loss History
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(loss_history, color='#2b5c8f', linewidth=1.5, label='Total Loss')
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_title('Loss History', fontsize=12, pad=10)
    
    plt.tight_layout()
    # png at dpi=300 is good enough; switching the extension to .pdf gives vector output for LaTeX
    plt.savefig('images/final/loss.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


    # ------------------------------------------------------------------
    # 2. Partial Losses
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(loss_norm_history, color=color_norm, linewidth=1.5, label='Normalization')
    ax.semilogy(lossbc_history, color=color_bc, linewidth=1.5, label='BCs')
    ax.semilogy(losseq_history, color=color_eq, linewidth=1.5, label='Equation')
    
    # horizontal convergence line
    ax.axhline(
        400,
        color='#1f77b4',
        linestyle='--',
        linewidth=1.2,
        label='Convergence Limit'
    )
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.8, loc='upper right')
    ax.set_title('Partial Losses History', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig('images/final/partial_losses.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


    # ------------------------------------------------------------------
    # 3. Weights History
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(weight_loss_norm_history, color=color_norm, linewidth=1.5, label='Normalization')
    ax.semilogy(weight_lossbc_history, color=color_bc, linewidth=1.5, label='BCs')
    ax.semilogy(weight_losseq_history, color=color_eq, linewidth=1.5, label='Equation')
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 'Weight' instead of the generic 'Loss': that is what is actually on this axis
    ax.set_ylabel('Weight', fontsize=12) 
    ax.set_xlabel('Epoch', fontsize=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.8, loc='upper right')
    ax.set_title('Weights History', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig('images/final/partial_losses_weights.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


    # ------------------------------------------------------------------
    # 4. Energy History
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    
    ax.plot(
        E_history, 
        color='#2b5c8f', 
        linewidth=1.8, 
        alpha=0.9, 
        label='NN Value'
    )
    
    # exact binding energy, drawn as a horizontal reference
    ax.axhline(
        -2.2245758106075337, 
        color='#9e2a2b', 
        linestyle='--', 
        linewidth=1.5, 
        label='Exact Value'
    )
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel(r'Energy $E$ (MeV)', fontsize=12) 
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_title('Energy History', fontsize=12, pad=10)
    
    ax.legend(
        frameon=True, 
        facecolor='white', 
        edgecolor='none', 
        framealpha=0.8, 
        loc='upper right'
    )
    
    plt.tight_layout()
    plt.savefig('images/final/Energy.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


    # ------------------------------------------------------------------
    # 5. Patience History
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.semilogy(patience_loss_norm_history, color=color_norm, linewidth=1.5, label='Normalization')
    ax.semilogy(patience_lossbc_history, color=color_bc, linewidth=1.5, label='BCs')
    ax.semilogy(patience_losseq_history, color=color_eq, linewidth=1.5, label='Equation')
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel('Patience', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.8, loc='upper right')
    ax.set_title('Patience History', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig('images/final/patience.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


    # ------------------------------------------------------------------
    # 6. Wavefunction and Potential
    # ------------------------------------------------------------------
    grid_test = grid_loss
    output = net(grid_test.unsqueeze(1).to(device))                          # [M,2]
    uS_pos = (output[:, 0] * torch.abs(grid_test.to(device))).cpu().detach().numpy()
    uD_pos = (output[:, 1] * torch.abs(grid_test.to(device))).cpu().detach().numpy()
    psi_pos = uS_pos
    V_plot = VNN_Minnesota_r(grid_test)
    
    fig, ax = plt.subplots()
    
    # wave function: the two channels u_S and u_D
    ax.scatter(
        grid_test.detach().numpy(),
        uS_pos,
        color='#e31a1c',
        s=1.5,
        alpha=0.7,
        label=r'$u_S(r)$'
    )
    ax.scatter(
        grid_test.detach().numpy(),
        uD_pos,
        color='#1f1adc',
        s=1.5,
        alpha=0.7,
        label=r'$u_D(r)$'
    )
    
    # potential rescaled to the wave function, just so both fit on the same axes
    V_np = V_plot.detach().numpy()
    V_scaled = V_np / (np.max(np.abs(V_np))) * np.max(np.abs(psi_pos))
    
    ax.plot(
        grid_test.detach().numpy(), 
        V_scaled, 
        color='#1f77b4',
        linewidth=1.5, 
        label=r'Potential $V(r)$'
    )
    
    ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_ylabel(r'$\psi(r)$ or $V(r)$ ', fontsize=12)
    ax.set_xlabel(r'$r$ (fm)', fontsize=12)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.8, loc='upper right')
    ax.set_title('Wavefunction of the Deuteron in Position Space', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig(
        f'images/final/wf_{epoch_tot}_E_{E_est}_Loss_{loss}.png', 
        bbox_inches='tight', 
        dpi=300
    )
    plt.close(fig)