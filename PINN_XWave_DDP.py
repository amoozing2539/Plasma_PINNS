import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.jit as jit

import numpy as np
import matplotlib.pyplot as plt

import os
import time
import pickle
import tqdm
import random
import typing
import itertools
from typing import Union
from datetime import datetime

plt.style.use('seaborn-v0_8-poster')


def set_seed(seed: int = 42) -> None: #meaning of life 

    """
    Function to set the PRNG seed of the sampling, NN initialization, etc.
    Used for reproducibility.

    Inputs: 
    - seed (int): PRNG seed. Default: 42 (Meaning of Life, the Universe, and Everything)

    Outputs:
    None
    """

    if type(seed) != int:
        raise Exception("Input seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def np_to_tensor(arr: np.ndarray, reshape: bool = False, dims: tuple = (-1,1), device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32, requires_grad: bool = False) -> torch.Tensor:

    """
    Function to convert NumPy array to Pytorch tensor. Optionally, the tensor can be reshaped,
    which is specified with the boolean flag 'reshape'. If 'reshape' is True, the tensor is reshaped
    to have dimension 'dims', which should be passed as a tuple.

    Inputs:
    - arr (array): NumPy array to be converted to a PyTorch tensor.
    - reshape (bool): boolean flag indicating whether or not to reshape the array. Default: False
    - dims (tuple): tuple indicating what shape to reshape the array to, if desired. Default: (-1,1)
    - device (torch.device): indicates whether resulting tensor will be on CPU or GPU, and if GPU, which one. Default: torch.device("cpu")
    - dtype (torch.dtype): indicates what dtype the elements of the resulting tensor will have. Default: torch.float32
    - requires_grad (bool): boolean flag indicating whether or not operations on the resulting tensor will be recorded with autograd. Default: False

    Outputs:
    - PyTorch tensor corresponding to the input 'arr'.
    """

    if reshape:
        return torch.tensor(arr.reshape(dims), dtype=dtype, requires_grad=requires_grad, device=device)
    else:
        return torch.tensor(arr, dtype=dtype, requires_grad=requires_grad, device=device)


def tensor_to_np(tensor: torch.Tensor, reshape: bool = False, dims: tuple = (-1,1)) -> np.ndarray:

    """
    Function to convert PyTorch tensor to NumPy array. Optionally, the array can be reshaped,
    which is specified with the boolean flag 'reshape'. If 'reshape' is True, the array is reshaped
    to have dimension 'dims', which should be passed as a tuple.

    Inputs:
    - tensor (tensor): PyTorch tensor to be converted to a NumPy array.
    - reshape (bool): boolean flag indicating whether or not to reshape the array.
    - dims (tuple): tuple indicating what shape to reshape the array to, if desired.

    Outputs:
    - NumPy array corresponding to the input 'tensor'.
    """

    if reshape:
        return tensor.reshape(dims).detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def get_lr(optimizer) -> float:
    
    """
    Function gives the current (global) learning rate of the passed optimizer.

    Inputs:
    - optimizer (instance of torch.optim class)

    Outputs:
    - (Global) learning rate associated with 'optimizer'.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    # return optimizer.param_groups[0]['lr']
  
    
#calculate k from dispersion relation
def dispersion(omega, omega_ce, omega_pe, vth):
    """
    Function to calculate the wave number k and the normalized wave number k_lambda_D.
    
    Inputs:
    -omega(float):  wave frequency normalized to plasma frequency (omega_pe)
    -omega_ce(float): electron cyclotron frequency normalized to plasma frequency (omega_pe)
    -omega_pe(float): plasma frequency normalized to itself (usually set to 1)
    
    Outputs:
    -k(float): wave number in units of omega_pe/c
    -k_lambda_D(float): Unitless product k*lambda_D, where lambda_D is the Debye length.
    """
    
    omega_h_sq = omega_pe**2 + omega_ce**2
    omega_h_factor = omega_h_sq/(omega_pe**2)
    denom = omega**2 - omega_h_factor
    
    k_sq_over_omega_sq = 1 - (1/omega**2)*((omega**2-1)/denom)
    
    k = np.sqrt(omega**2 * k_sq_over_omega_sq)

    k_lambda_D = k*vth
    
    return k, k_lambda_D


def check_size_eq(lst):
    """Check if all lists in lst have the same size."""
    return not any(len(lst[0])!= len(i) for i in lst)


def phase_factor(delta):
    """Calculate the phase factor for the wave solution."""
    return np.exp(1j*delta)


def analytical_solution(x, t, omega, k, phi_amp, delta):
    
    phi_amp *= phase_factor(delta)
    
    B0_sq_num = ((omega**2)-(k**2)-1)*((omega**2)-1)
    B0_sq_denom = ((omega**2)-(k**2))
    B0 = np.sqrt(B0_sq_num / B0_sq_denom)                                     #We Choose A1y and B0 such that all constraints are satisfied.
        
    phase = 1j*(k*x - omega*t)
    factor = np.exp(phase)
    
    phi = phi_amp * factor
    
    # Perturbed Vector Potential A1
    A1y_amp = (phi_amp/B0) * (((omega**2)/k) - (1/k))
    A1y = A1y_amp * factor
    A1x = (omega/k)*phi
    
    
    # Electron density perturbation N = n_e - n_0
    N = ((omega**2) - (k**2))*phi
    
    # Perturbed electron velocities v1
    vx = (omega/k)*((omega**2) - (k**2))*phi
    vy = ((omega**2) - (k**2))*A1y
    
    # Time derivative of perturbed B field (from curl of A1)
    Bdot = omega*k*A1y
    
    #Background field B0 is constant.
    B0_value = np.full_like(Bdot, B0, dtype=np.complex128)
     
    return np.real(phi), np.imag(Bdot), np.real(A1x), np.imag(A1y), np.real(vx), np.imag(vy), np.real(N), np.real(B0_value)

def derivative_t(f, dt):

    f_t = np.zeros_like(f)

    f_t[1:-1,:] = (0.5*f[2:,:] - 0.5*f[0:-2,:]) / dt             # 2nd order accurate central difference stencil for interior points
    f_t[0,:] = (-1.5*f[0,:] + 2.0*f[1,:] - 0.5*f[2,:]) / dt      # 2nd order accurate forward difference stencil for t_0 edge
    f_t[-1,:] = (0.5*f[-3,:] - 2.0*f[-2,:] + 1.5*f[-1,:]) / dt   # 2nd order accurate backward difference stencil for t_f edge

    return f_t

def derivative_tt(f, dt):
    f_tt = np.zeros_like(f)
    
    f_tt = np.zeros_like(f)

    # 2nd order accurate central difference stencil for interior points
    f_tt[1:-1, :] = (f[2:, :] - 2.0 * f[1:-1, :] + f[0:-2, :]) / (dt**2)

    # 2nd order accurate forward difference stencil for the t_0 edge
    f_tt[0, :] = (2.0 * f[0, :] - 5.0 * f[1, :] + 4.0 * f[2, :] - f[3, :]) / (dt**2)

    # 2nd order accurate backward difference stencil for the t_f edge
    f_tt[-1, :] = (-f[-4, :] + 4.0 * f[-3, :] - 5.0 * f[-2, :] + 2.0 * f[-1, :]) / (dt**2)

    return f_tt

def derivative_x(f, dx):

    f_x = np.zeros_like(f)

    f_x[:,1:-1] = (0.5*f[:,2:] - 0.5*f[:,0:-2]) / dx             # 2nd order accurate central difference stencil for interior points
    f_x[:,0] = (-1.5*f[:,0] + 2.0*f[:,1] - 0.5*f[:,2]) / dx      # 2nd order accurate forward difference stencil for x_0 edge
    f_x[:,-1] = (0.5*f[:,-3] - 2.0*f[:,-2] + 1.5*f[:,-1]) / dx   # 2nd order accurate backward difference stencil for x_f edge

    return f_x

def derivative_xx(f, dx):

    f_xx = np.zeros_like(f)

    f_xx[:,1:-1] = (f[:,2:] - 2.0*f[:,1:-1] + f[:,0:-2]) / dx**2                  # 2nd order accurate central difference stencil (interior)
    f_xx[:,0] = (2.0*f[:,0] - 5.0*f[:,1] + 4.0*f[:,2] - 1.0*f[:,3]) / dx**2       # 2nd order accurate forward difference stencil (x_0 edge)
    f_xx[:,-1] = (2.0*f[:,-1] - 5.0*f[:,-2] + 4.0*f[:,-3] - 1.0*f[:,-4]) / dx**2  # 2nd order accurate backward difference stencil (x_f edge)

    return f_xx

def derivative_xt(f,dx,dt):
    f_xt = np.zeros_like(f)
    
    f_xt[:,1:-1] = (f[:,2:] - 2.0*f[:,1:-1] + f[:,0:-2]) / dx*dt                  # 2nd order accurate central difference stencil (interior)
    f_xt[:,0] = (2.0*f[:,0] - 5.0*f[:,1] + 4.0*f[:,2] - 1.0*f[:,3]) / dx*dt       # 2nd order accurate forward difference stencil (x_0 edge)
    f_xt[:,-1] = (2.0*f[:,-1] - 5.0*f[:,-2] + 4.0*f[:,-3] - 1.0*f[:,-4]) / dx*dt  # 2nd order accurate backward difference stencil (x_f edge)
    return f_xt


def generate_data(xmin, xmax, tmin, tmax, nx, nt, vth, omega_ce, omega_pe, omega_list, phi_amp_list, delta_list):

    if not check_size_eq([omega_list, phi_amp_list, delta_list]):
        raise Exception("Parameter lists are not the same size!")

    x = np.linspace(xmin, xmax, nx)
    t = np.linspace(tmin, tmax, nt)
    
    x_arr, t_arr = np.meshgrid(x,t)
    
    phi = np.zeros_like(x_arr, dtype=np.complex128)
    Bdot = np.zeros_like(x_arr, dtype=np.complex128)
    A1x = np.zeros_like(x_arr, dtype=np.complex128)
    A1y = np.zeros_like(x_arr, dtype=np.complex128)
    Vx = np.zeros_like(x_arr, dtype=np.complex128)
    Vy = np.zeros_like(x_arr, dtype=np.complex128)
    N = np.zeros_like(x_arr, dtype=np.complex128)
    B0 = np.zeros_like(x_arr, dtype=np.complex128)

    for i in range(len(omega_list)):
        k, _ = dispersion(omega=omega_list[i], omega_ce=omega_ce, omega_pe=omega_pe, vth=vth)
        temp_phi, temp_Bdot, temp_A1x, temp_A1y, temp_Vx, temp_Vy, temp_N, temp_B0 = analytical_solution(x=x_arr, t=t_arr, omega=omega_list[i],
                                                                                                k=k, phi_amp=phi_amp_list[i],
                                                                                                delta=delta_list[i])
        phi += temp_phi
        Bdot += temp_Bdot
        A1x += temp_A1x
        A1y += temp_A1y
        Vx += temp_Vx
        Vy += temp_Vy
        N += temp_N
        B0 += temp_B0
        
        #Return real part of all quantities
    # return x_arr.flatten(), t_arr.flatten(), np.real(phi).flatten(), np.real(Bdot).flatten(), \
    #         np.real(A1x).flatten(), np.real(A1y).flatten(), \
    #         np.real(Vx).flatten(), np.real(Vy).flatten(), np.real(N).flatten(), np.real(B0).flatten()
        
        #Return full complex quantities of all quantities  
    return x_arr.flatten(), t_arr.flatten(), phi.flatten(), Bdot.flatten(), \
            A1x.flatten(), A1y.flatten(), \
            Vx.flatten(), Vy.flatten(), N.flatten(), B0.flatten()


def sparse_measurements(x, t, phi, Bdot, num_samples):

    indices = np.random.choice(x.shape[0], num_samples, replace=False)

    return x[indices], t[indices], phi[indices], Bdot[indices]


def collocation_points(xmin, xmax, tmin, tmax, Nx, Nt, L, tau):

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)

    x_coll, t_coll = np.meshgrid(x, t)

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    return x_coll.flatten(), t_coll.flatten(), dx, dt


# def spectral_derivative(f,x):
#     N = len(x)
#     dx = x[1] - x[0]
    
#     f_hat = np.fft.fft(f)
#     k = 2*np.pi*np.fft.fftfreq(N, d=dx)
#     df_hat = 1j * k * f_hat
#     df = np.fft.ifft(df_hat)
    
#     return df.real
#     # L is the length of the domain (e.g., L = x_max - x_min)
#     N = f.shape[1] # Number of points in x
    
#     # Create the wavenumber array
#     k_x = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    
#     # Take the FFT along the x-axis
#     f_hat = np.fft.fft(f, axis=1)
    
#     # Multiply by ik
#     dfdx_hat = 1j * k_x * f_hat
    
#     # Inverse FFT to get the derivative
#     dfdx = np.fft.ifft(dfdx_hat, axis=1)
    
#     return np.real(dfdx)


#Plot GTs
def plot_analytical(quantities, sparse, titles, extent):
    """Plot the analytical solution."""
    
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
    colors = ['PiYG', 'PRGn', 'BrBG',
              'PuOr', 'RdGy', 'RdBu',
              'RdYlBu', 'RdYlGn', 'bwr',
              'seismic', 'berlin', 'vanimo']
    x_sparse = sparse[0]
    t_sparse = sparse[1]
    phi_sparse = sparse[2]
    bdot_sparse = sparse[3]
    
                      
    for ax, quantity, title, color in zip(axes.flatten(), quantities, titles, colors):
        im = ax.imshow(quantity, aspect='auto', extent=extent, origin='lower', cmap=color)
        ax.set_title(title, fontsize = 25)
        ax.set_xlabel(f'$x [\\frac{{c}}{{\omega_{{pe}}}}]$')
        ax.set_ylabel('$t [\\omega_{{pe}}^{{-1}}]$')
        fig.colorbar(im, ax=ax)
        
        if f'$\phi [\\frac{{m_{{e}}c^2}}{{e}}]$' in title:
            ax.scatter(x_sparse, t_sparse, c='black', s=20, label=f'500 φ Measurements', alpha=1.)
            ax.legend()
        if f'$\dot{{B}} [\\frac{{e}}{{m_{{e}}c}}]$' in title:
            ax.scatter(x_sparse, t_sparse, c='black', s=20, label=f'500 Bdot Measurements', alpha=1.)
            ax.legend()
            
    
    plt.tight_layout()
    plt.show()
    plt.suptitle("Ground Truth")
    # plt.savefig("XWave_Ground_Truth_Plots")



def plot_constraints(constraints, titles, extent):
    "Plots the constraints in their current form."
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))
    
    #adjusted titles to remove the average
    adjusted_titles = [r'$|\partial_t V_x - \partial_t A_{1x} - \partial_x \phi + V_y B_0|$',
                       r'$|\partial_t V_y - \partial_t A_{1y} - V_x B_0|$']
    
    for ax, constraint, title in zip(axes.flatten(), constraints, titles):
        
        #make a copy to avoid modifying the original constraint
        plot_data = np.abs(constraint.copy())
        
        if title in adjusted_titles:
            mean_value = np.mean(plot_data)
            plot_data -= mean_value  # Remove the mean value from the constraint
            print(f"Adjusting title: {title} by subtracting mean value: {mean_value:.4f}")
            
        im = ax.imshow(plot_data, aspect='auto', extent=extent, origin='lower', cmap='Reds')
        ax.set_title(title, fontsize = 15)
        ax.set_xlabel(f'$x [\\frac{{c}}{{\omega_{{pe}}}}]$')
        ax.set_ylabel('$t [\\omega_{{pe}}^{{-1}}]$')
        fig.colorbar(im, ax=ax)
        
    
    plt.tight_layout()
    plt.show()
    # plt.savefig("XWave_Constraints.png")
    
# print("------Plotting Constraints as is--------------")
# plot_constraints(constraints = constraints, titles = constraint_titles, extent=constraints_ext)


# Helper functions for initialization of network parameters

def xavier_init_weights(network: nn.Module) -> nn.Module:
    with torch.no_grad():
        print("Initializing NN with glorot initialization...\n")
        for m in network:
            if hasattr(m, 'weight'):
                nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)

    return network

def kaiming_init_weights(network: nn.Module) -> nn.Module:
    with torch.no_grad():
        print("Initializing NN with he initialization...\n")
        for m in network:
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)

    return network

# Helper function for initialization of activation function

def init_param(w0: float, device: torch.device, adaptive: bool) -> nn.Parameter:
    return nn.Parameter(torch.tensor(w0, dtype=torch.float32, requires_grad=adaptive, device=device))

# Helper function for Fourier Feature embedding

def encode_input(inp: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    inp = inp.to(B.device, dtype=torch.float32)  # Ensure inp is on the same device as B
    B = B.to(inp.device, dtype=torch.float32)
    input_proj = (2.0*torch.pi*inp) @ B.t()
    return torch.cat([torch.sin(input_proj), torch.cos(input_proj)], dim=-1)

# Activation function definitions

class Sin(nn.Module):

    def __init__(self, device: torch.device, w0: float = 1., adaptive: bool = False) -> None:
        super(Sin, self).__init__()
        self.w0 = init_param(w0, device, adaptive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

class Tanh(nn.Module):

    def __init__(self, device: torch.device, w0: float = 1., adaptive: bool = False) -> None:
        super(Tanh, self).__init__()
        self.w0 = init_param(w0, device, adaptive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.w0 * x)

class Swish(nn.Module):

    def __init__(self, device: torch.device, w0: float = 1., adaptive: bool = False) -> None: 
        super(Swish, self).__init__()
        self.w0 = init_param(w0, device, adaptive)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.w0 * x)

#Actual construction of the MLP class
class MLP(nn.Module):
    
    def __init__(self, extent: np.ndarray, device: torch.device, num_hidden_layers: int = 3, hidden_size: int = 64, activation: str = 'tanh', adaptive_af: bool = False, init: Union[None, str] = None, input_encoding: bool = False, sigma: float = 1.0, single_output: bool = False) -> None:
        
        super(MLP, self).__init__()
        
        self.tmin = torch.tensor(extent[0], dtype=torch.float32, device=device, requires_grad=False)
        self.tmax = torch.tensor(extent[1], dtype=torch.float32, device=device, requires_grad=False)
        self.xmin = torch.tensor(extent[2], dtype=torch.float32, device=device, requires_grad=False)
        self.xmax = torch.tensor(extent[3], dtype=torch.float32, device=device, requires_grad=False)
        
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.input_size = 2  # spatiotemporal input (x, t)
        self.elapsed_iterations = 0
        
        if single_output:
            self.output_size = 1
        else:
            self.output_size = 7 # 7 outputs: phi, Bdot, Ax, Ay, Vx, Vy, N
              
        if activation == 'tanh' or activation == 'Tanh':
            self.activation_fn = Tanh(device, adaptive=adaptive_af)
            
        elif activation == 'sin' or activation == 'Sin':
            self.activation_fn = Sin(device, adaptive=adaptive_af)
            
        elif activation == 'swish' or activation == 'Swish':
            self.activation_fn = Swish(device, adaptive=adaptive_af)

        else:
            raise Exception("Invalid choice of activation fn. Choices are 'tanh', 'sin', or 'swish'.")
        
        self.init = init 
        self.input_encoding = input_encoding #convert raw input data into suitable numerical representation for NN processing
        
        self.sigma = sigma #enhance activation fns. Rather than tanh(x) -> tanh(sigma*x). Especially for high frequency stuff. 
        self.encoded_size = hidden_size 
        
        if self.input_encoding:
            self.input_size = 2 * hidden_size #if input encoding is used, the input size is doubled
        
        self.B = np_to_tensor(np.random.normal(scale=self.sigma, size=(self.encoded_size, 2)), device=device, dtype=torch.float32, requires_grad=False)
        
        #now making the real MLP network
        self.network = nn.Sequential() 
        self.network.add_module("input_layer", nn.Linear(self.input_size, hidden_size, bias=True))
        self.network.add_module("input_activation", self.activation_fn)
        for i in range(num_hidden_layers):
            self.network.add_module(f"hidden_layer_{i}", nn.Linear(hidden_size, hidden_size, bias=True))
            self.network.add_module(f"hidden_activation_{i}", self.activation_fn)
        self.network.add_module("output_layer", nn.Linear(hidden_size, self.output_size)) #single output for each input
        
        if init is not None:
            if init == 'kaiming' or init == 'he': #Random initialization from a Gaussian distribution W -> N(0, sqrt(2/n)) n = number of inputs to the node
                self.network = kaiming_init_weights(self.network)
            elif init == 'xavier' or init == 'glorot': #
                self.network = xavier_init_weights(self.network)
            else:
                raise Exception("Invalid choice of initialization method. Choices are 'kaiming / he' or 'xavier / glorot'.")
            
        self.__repr__()
        
    def __getstate__(self):
        return self.__dict__.copy()
    def __setstate__(self, state):
        return self.__dict__.update(state)
    
    def forward(self, t:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        
        t = 2.0*((t - self.tmin) / (self.tmax - self.tmin)) - 1
        x = 2.0*((x - self.xmin) / (self.xmax - self.xmin)) - 1

        inp = torch.cat((t, x), dim=1)

        if self.input_encoding:
            inp = encode_input(inp, self.B)
            
        u = self.network(inp)
            
        return u
    
    def __repr__(self) -> str:

        """
        Class method to print all relevant attributes for the MLP class.
        
        Inputs:
        None
        
        Outputs:
        Blank string (since all other relevant info is printed in-line)
        """

        print('\n')
        print(f'MLP tmin: {self.tmin.item():.4f}, MLP tmax: {self.tmax.item():.4f}, MLP xmin: {self.xmin.item():.4f}, MLP xmax: {self.xmax.item():.4f}\n')
        print(f'Number of hidden layers: {self.num_hidden_layers}')
        print(f'Hidden layer size: {self.hidden_size} nodes / layer')
        print(f'Input size: {self.input_size}')
        print(f'Output size: {self.output_size}\n')
        print(f'Activation function used: {str(self.activation_fn)}')
        print(f'NN parameter initialization mode used: {str(self.init)}')
        print(f'Use Fourier feature input encoding? {self.input_encoding}')

        if self.input_encoding:
            print(f'Sigma of Fourier feature input encoding = {self.sigma}')

        print('\n')
        print(f'Network architecture: {self.network}')
        print('\n')

        return ''


def pde_residuals(model, t, x, B0_amp, means, stds):
    
    #Check tracking gradients
    grad_enabled = torch.is_grad_enabled()
    
    device = t.device
    B0_amp = B0_amp.to(device)
    B0_amp_value = B0_amp
    
    # Set requires_grad on inputs if needed
    if grad_enabled and not t.requires_grad:
        t = t.detach().requires_grad_(True)
    if grad_enabled and not x.requires_grad:
        x = x.detach().requires_grad_(True)
        
    residuals = torch.zeros(t.size(dim=0), 11).to(device)
    
    phi, Bdot, Ax, Ay, Vx, Vy, N = ((model(t, x)*stds + means).T).to(device)
    
    phi = phi.reshape(-1, 1)
    Bdot = Bdot.reshape(-1, 1)
    Ax = Ax.reshape(-1,1)
    Ay = Ay.reshape(-1,1)
    Vx = Vx.reshape(-1, 1)
    Vy = Vy.reshape(-1, 1)
    N = N.reshape(-1, 1)
    
    
    if grad_enabled:
        
        phi_x = grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_xx = grad(phi_x, x, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0]
        phi_t = grad(phi, t, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        phi_tt = grad(phi_t, t, grad_outputs=torch.ones_like(phi_t), create_graph=True)[0]
        phi_xt = grad(phi_x, t, grad_outputs=torch.ones_like(phi_x), create_graph=True)[0]
        
        Ax_x = grad(Ax, x, grad_outputs=torch.ones_like(Ax), create_graph=True)[0]
        Ax_xx = grad(Ax_x, x, grad_outputs=torch.ones_like(Ax_x), create_graph=True)[0]
        Ax_t = grad(Ax, t, grad_outputs=torch.ones_like(Ax), create_graph=True)[0]
        Ax_tt = grad(Ax_t, t, grad_outputs=torch.ones_like(Ax_t), create_graph=True)[0]
        Ax_xt = grad(Ax_x, t, grad_outputs=torch.ones_like(Ax_x), create_graph=True)[0]

        
        Ay_x = grad(Ay, x, grad_outputs=torch.ones_like(Ay), create_graph=True)[0]
        Ay_xx = grad(Ay_x, x, grad_outputs=torch.ones_like(Ay_x), create_graph=True)[0]
        Ay_t = grad(Ay, t, grad_outputs=torch.ones_like(Ay), create_graph=True)[0]
        Ay_tt = grad(Ay_t, t, grad_outputs=torch.ones_like(Ay_t), create_graph=True)[0]
        Ay_xt = grad(Ay_x, t, grad_outputs=torch.ones_like(Ay_x), create_graph=True)[0]
        
        Vx_t = grad(Vx, t, grad_outputs=torch.ones_like(Vx), create_graph=True)[0]
        Vx_x = grad(Vx, x, grad_outputs=torch.ones_like(Vx), create_graph=True)[0]
        
        Vy_t = grad(Vy, t, grad_outputs=torch.ones_like(Vy), create_graph=True)[0]
        Vy_x = grad(Vy, x, grad_outputs=torch.ones_like(Vy), create_graph=True)[0]
        
        N_t = grad(N, t, grad_outputs=torch.ones_like(N), create_graph=True)[0]
        
        
        #Don't need derivatives for Vx, Vy, N, or Bdot
        
        residuals[:, 0:1] = Ax_x + phi_t                        #Gauge
        residuals[:, 1:2] = phi_xx + Ax_xt - N                  #Gauss's Law
        residuals[:, 2:3] = Ax_tt + phi_xt - Vx                 #Ampere's Law x
        residuals[:, 3:4] = Ay_tt - Ay_xx + Vy                  #Ampere's Law y
        # residuals[:, 4:5] = Ax_xx - Ax_tt - Vx                  #Wave_Ax
        # residuals[:, 5:6] = Ay_xx - Ay_tt - Vy                  #Wave_Ay  
        # residuals[:, 6:7] = phi_xx - phi_tt - N                 #Wave_phi 
        residuals[:, 4:5] = Bdot - Ay_xt                        #time_derivative of B field
        residuals[:, 5:6] = Vx_t - phi_x - Ax_t - Vy*B0_amp     #Momentum_x
        residuals[:, 6:7] = Vy_t - Ay_t - Vx*B0_amp             #Momentum_y
        residuals[:, 7:8] = N_t + Vx_x                          #continuity
        
    else:
        residuals.fill_(0.0)  # If gradients are not enabled, set residuals to zero
            
    return residuals


#With DDP (What makes our PINNs PINNs and not NNs)
def get_physics_loss(model, t_coll, x_coll, B0_amp, means, stds, rank):
    
    #DDP part
    t_coll_d = t_coll.to(rank)
    x_coll_d = x_coll.to(rank)
    t_coll = t_coll_d.detach().requires_grad_(True)
    x_coll = x_coll_d.detach().requires_grad_(True)
    
    residuals = pde_residuals(model, t_coll, x_coll, B0_amp, means, stds)
    
    gauge_res = residuals[:, 0:1]
    gauss_law_res = residuals[:, 1:2]
    ampere_x_res = residuals[:, 2:3]
    ampere_y_res = residuals[:, 3:4]
    bdot_curlA_res = residuals[:, 4:5]
    momentum_x_res = residuals[:, 5:6]
    momentum_x_res_copy = torch.abs(momentum_x_res.clone())
    momentum_x_res -= torch.mean(momentum_x_res_copy)  # Remove the mean value from the momentum_x residuals
    momentum_y_res = residuals[:, 5:6]
    momentum_y_res_copy = torch.abs(momentum_y_res.clone())
    momentum_y_res -= torch.mean(momentum_y_res_copy) #remove the mean value from the momentum_y residuals
    
    continuity_res = residuals[:, 6:7]
    
    gauge_loss = torch.mean(gauge_res**2)
    gauss_law_loss = torch.mean(gauss_law_res**2)
    ampere_x_loss = torch.mean(ampere_x_res**2)
    ampere_y_loss = torch.mean(ampere_y_res**2)
    bdot_curlA_loss = torch.mean(bdot_curlA_res**2)
    momentum_x_loss = torch.mean(momentum_x_res**2)
    momentum_y_loss = torch.mean(momentum_y_res**2)
    continuity_loss = torch.mean(continuity_res**2)
    
    #coefficients are loss weighting
    physics_loss =  gauge_loss + gauss_law_loss + ampere_x_loss + ampere_y_loss + \
                    bdot_curlA_loss + momentum_x_loss + momentum_y_loss + continuity_loss
                    
    
    return  physics_loss, gauge_loss, gauss_law_loss, ampere_x_loss, ampere_y_loss, \
            bdot_curlA_loss, momentum_x_loss, momentum_y_loss, continuity_loss


def get_sm_loss(model, t_sparse, x_sparse, phi_sparse, Bdot_sparse, means, stds, rank):
    
    #DDP part
    t_sparse_d = t_sparse.to(rank)
    x_sparse_d = x_sparse.to(rank)
    phi_sparse_d = phi_sparse.to(rank)
    Bdot_sparse_d = Bdot_sparse.to(rank)
    
    
    phi_sparse_preds = model(t_sparse, x_sparse)[:, 0:1]
    Bdot_sparse_preds = model(t_sparse, x_sparse)[:, 1:2]
    
    phi_loss = torch.mean(torch.square(phi_sparse_preds - ((phi_sparse - means[:,0:1])/stds[:,0:1])))
    Bdot_loss = torch.mean(torch.square(Bdot_sparse_preds - ((Bdot_sparse - means[:,1:2])/stds[:,1:2])))
    
    sm_loss = phi_loss + Bdot_loss
    
    return sm_loss

def get_total_loss(model, t_sparse, x_sparse, phi_sparse, Bdot_sparse, t_coll, x_coll, B0_amp, means, stds, rank, lamda=1.0):
    
    physics_loss, gauge_loss, gauss_law_loss, ampere_x_loss, ampere_y_loss, \
    bdot_curlA_loss, momentum_x_loss, momentum_y_loss, continuity_loss = get_physics_loss(model, t_coll, x_coll, B0_amp, means, stds, rank)
    
    sm_loss = get_sm_loss(model, t_sparse, x_sparse, phi_sparse, Bdot_sparse, means, stds, rank)
    
    loss = lamda*sm_loss + physics_loss #lamda is the weighting factor for the physics loss (default = 1.0)
    
    return  loss, sm_loss, physics_loss, \
            gauge_loss, gauss_law_loss, ampere_x_loss, ampere_y_loss, \
            bdot_curlA_loss, momentum_x_loss, momentum_y_loss, continuity_loss

def write_loss(hist, loss, sm_loss, physics_loss, 
               gauge_loss, gauss_law_loss, ampere_x_loss, ampere_y_loss,
                bdot_curlA_loss, momentum_x_loss, momentum_y_loss, continuity_loss):
    
    """Writes the loss history to a dictionary"""
    
    hist['loss'].append(loss.item())
    hist['sm_loss'].append(sm_loss.item())
    hist['physics_loss'].append(physics_loss.item())
    hist['gauge_loss'].append(gauge_loss.item())
    hist['gauss_loss'].append(gauss_law_loss.item())
    hist['ampere_x_loss'].append(ampere_x_loss.item())
    hist['ampere_y_loss'].append(ampere_y_loss.item())
    hist['bdot_curlA_loss'].append(bdot_curlA_loss.item())
    hist['momentum_x_loss'].append(momentum_x_loss.item())
    hist['momentum_y_loss'].append(momentum_y_loss.item())
    hist['continuity_loss'].append(continuity_loss.item())
    
    return hist

#optimization function
def optimize(model, optimizer, scheduler, hist, num_epochs, n_batches,
             t_sparse, x_sparse, phi_sparse, Bdot_sparse,
             t_coll, x_coll, B0_amp, means, stds, rank, lamda=1.0):
    
    # Ensure all data tensors are on the correct device for this process
    t_sparse_d = t_sparse.to(rank)
    x_sparse_d = x_sparse.to(rank)
    phi_sparse_d = phi_sparse.to(rank)
    Bdot_sparse_d = Bdot_sparse.to(rank)
    t_coll_d_full = t_coll.to(rank)
    x_coll_d_full = x_coll.to(rank)
    means_d = means.to(rank)
    stds_d = stds.to(rank)
    
    for epoch in range(num_epochs):
        
        n_coll = t_coll.shape[0]
        i_idxs = np.random.choice(n_coll, n_batches, replace=False) #indexes for the collocation points, randomly chosen for each batch
        
        for i in range(n_batches):
            
            t_coll_batch = t_coll[i_idxs[i]:i_idxs[i]+1] #separate the collocation points into batches
            x_coll_batch = x_coll[i_idxs[i]:i_idxs[i]+1]
            
            optimizer.zero_grad() #clear the gradients of our optimizer 
            
            loss, sm_loss, physics_loss, \
            gauge_loss, gauss_law_loss, ampere_x_loss, ampere_y_loss, bdot_curlA_loss, \
            momentum_x_loss, momentum_y_loss, continuity_loss = get_total_loss(model, t_sparse, x_sparse, phi_sparse, 
                                                                        Bdot_sparse, t_coll, x_coll, B0_amp, means, stds, 
                                                                        rank, lamda=10.)
            
            loss.backward() #calculate the gradients of the loss w.r.t. the model parameters. Backwards pass
            optimizer.step() #update the model parameters using the optimizer *magic*
            
            #update our loss history only for rank 0 
            if rank == 0:
                hist = write_loss(hist, loss, sm_loss, physics_loss,
                                    gauge_loss, gauss_law_loss, ampere_x_loss, ampere_y_loss,
                                    bdot_curlA_loss, momentum_x_loss, momentum_y_loss, continuity_loss)
            
            old_lr = get_lr(optimizer) #get the current learning rate
            scheduler.step(loss) #update the learning rate using the scheduler based on the loss
            
            if get_lr(optimizer) < old_lr:
                print(f"LR has been set to {get_lr(optimizer):.4e}.")
            
        if rank == 0 and epoch % 50 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Total Loss: {loss.item():.4e}, SM Loss: {sm_loss.item():.4e}, Physics Loss: {physics_loss.item():.4e}")
        
    return model, optimizer, hist


def plot_loss_histories(hist):
    "Total Loss, SM Loss, and Physics Loss histories."
    
    plt.figure()
    
    plt.title("PINN Loss Evolution Histories")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (arbitrary units)")
    
    plt.semilogy(hist['sm_loss'], label=r'$\mathcal{L}_{Data}$', color='blue')
    plt.semilogy(hist['physics_loss'], label=r'$\mathcal{L}_{Physics}$', color='green')
    plt.semilogy(hist['loss'], label=r'$\mathcal{L}_{Total}$', color='orange')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #physics loss components
    
    plt.figure()
    plt.title("PINN Physics Loss Evolution Histories")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (arbitrary units)")
    
    plt.semilogy(hist['gauge_loss'], label=r'$\mathcal{L}_{Gauge}$')
    plt.semilogy(hist['gauss_loss'], label=r'$\mathcal{L}_{Gauss}$')
    plt.semilogy(hist['ampere_x_loss'], label=r'$\mathcal{L}_{ampere_{x}}$')
    plt.semilogy(hist['ampere_y_loss'], label = r'$\mathcal{L}_{ampere_{y}}$')
    plt.semilogy(hist['bdot_curlA_loss'], label=r'$\mathcal{L}_{bdot_curl(A)}$')
    plt.semilogy(hist['momentum_x_loss'], label=r'$\mathcal{L}_{Momentum_x}$')
    plt.semilogy(hist['momentum_y_loss'], label=r'$\mathcal{L}_{Momentum_y}$')
    plt.semilogy(hist['continuity_loss'], label=r'$\mathcal{L}_{continuity}$')    
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig("XWave_loss_hist")
    
    return




def run_ddp_training(rank, world_size, params):
    
    print("-----Training Model----------")
    
    #initialize the process group
    dist.init_process_group(backend = 'nccl', rank = rank, world_size = world_size)
    torch.cuda.set_device(rank) #set the specific GPU for this process. 
    
    #Define physical parameters 
    omega = params['omega_list'][0]
    omega_ce = params['omega_ce'][0]
    omega_pe = params['omega_pe'][0]
    phi_amp = params['phi_amp_list'][0]
    vth = params['vth']
    k, k_lambda_D = dispersion(omega, omega_ce, omega_pe,vth)
    lamda_wave = 2*np.pi/k
    T_wave = (2*np.pi)/omega
    xmin, xmax = 0, 3*lamda_wave
    tmin, tmax = 0, 3*T_wave
    
    L = xmax-xmin
    tau = tmax - tmin
    Nt = params['Nt']
    Nx = params['Nx']
    Nt_coll = params['Nt_coll']
    Nx_coll = params['Nx_coll']
    # dt_val = T_wave/float(Nt) 
    # dx_val = L/float(Nx) 
    
    X_flat_np, T_flat_np, phi_flat_np, Bdot_flat_np, Ax_flat_np, Ay_flat_np, Vx_flat_np, Vy_flat_np, N_flat_np, B0_flat_np  = generate_data(
        xmin, xmax, tmin, tmax, Nx, Nt, params['vth'], params['omega_ce'][0], params['omega_pe'][0], params['omega_list'], params['phi_amp_list'], params['delta_list']
    )
    
    x_sparse_np, t_sparse_np, phi_sparse_np, Bdot_sparse_np = sparse_measurements(X_flat_np, T_flat_np, phi_flat_np, Bdot_flat_np, num_samples=params['num_sparse_samples'])
    x_coll_np, t_coll_np, dx_coll, dt_coll = collocation_points(xmin, xmax, tmin, tmax, Nx_coll, Nt_coll, L, tau)
    
    # Scalar tensor for the background magnetic field
    B0_amp_val = np_to_tensor(np.array(B0_flat_np[0]), device=rank, requires_grad=False)
    
    #Convert to tensors and move to correct device (rank)
    t_sparse_tensor = np_to_tensor(t_sparse_np, reshape = True, device = rank, requires_grad=True)
    x_sparse_tensor = np_to_tensor(x_sparse_np, reshape = True, device = rank, requires_grad=True)
    phi_sparse_tensor = np_to_tensor(phi_sparse_np, reshape = True, device = rank, requires_grad=True)
    Bdot_sparse_tensor = np_to_tensor(Bdot_sparse_np, reshape = True, device = rank, requires_grad=True)
    
    #Collocation points only require grad when calculating physics loss,
    # but it's safe to set here if they're always used that way.
    t_coll_tensor = np_to_tensor(t_coll_np, reshape=True, device=rank, requires_grad=True)
    x_coll_tensor = np_to_tensor(x_coll_np, reshape =True, device=rank, requires_grad=True)
    
    # Means and Stds for normalization (ensure these are on the correct device)
    data_flat_GT = np.array([phi_flat_np, Bdot_flat_np, Ax_flat_np, Ay_flat_np, Vx_flat_np, Vy_flat_np, N_flat_np])
    means = np_to_tensor(np.mean(data_flat_GT, axis=1), reshape=True, dims=(1,7), device=rank)
    stds = np_to_tensor(np.std(data_flat_GT, axis=1) + 1e-8, reshape=True, dims=(1,7), device=rank) #epsilon to prevent division by zero. 
    
    # Initialize Model for this process
    model = MLP(
        extent=params['extent'],
        device=torch.device(f'cuda:{rank}'), # Explicitly assign GPU
        num_hidden_layers=params['num_hidden_layers'],
        hidden_size=params['hidden_size'],
        activation=params['activation'],
        init=params['init'],
        input_encoding=params['input_encoding'],
        sigma=params['sigma']
    ).to(rank) # Move model to assigned GPU
    
    #Wrap the model with DDP
    #device_ids and output_device are necessary for DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8, verbose=True)
    
    hist = {
        'loss': [], 'sm_loss': [], 'physics_loss': [],
        'gauge_loss': [], 'gauss_loss': [], 'ampere_x_loss': [], 'ampere_y_loss': [],
        'bdot_curlA_loss':[], 'momentum_x_loss': [], 'momentum_y_loss': [], 'continuity_loss':[]}
    
    # Optimize the model
    model, optimizer, hist = optimize(
        model, optimizer, scheduler, hist, num_epochs=params['num_epochs'], n_batches=params['n_batches'],
        t_sparse=t_sparse_tensor, x_sparse=x_sparse_tensor,
        phi_sparse=phi_sparse_tensor, Bdot_sparse=Bdot_sparse_tensor,
        t_coll=t_coll_tensor, x_coll=x_coll_tensor, B0_amp=B0_amp_val, # Pass full tensors for batching inside optimize
        means=means, stds=stds, lamda=params['lamda'], rank=rank
    )
    
    # Save model only from rank 0
    if rank == 0:
        print("\nTraining complete. Saving model and history...")
        # Save state_dict for DDP model. Use model.module.state_dict() for actual model weights
        # because DDP wraps the original model.
        torch.save(model.module.state_dict(), "ddp_xwave_model.pt")
        # Save history
        with open(f"ddp_xwave_hist.pkl", "wb") as f:
            pickle.dump(hist, f)
            
    # Clean up process group
    dist.destroy_process_group()
    
    pass


def main():
    
    #Set a random seed for reproducibility
    set_seed(42) #meaning of life 
    
    # Set up environment variables for torch.distributed.launch
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'     

    world_size = 2 # Number of GPUs to use (your 2 RTX 4070s)

    # Define common parameters for all processes
    omega_list = [2.5] # Example values based on your notebook
    phi_amp_list = [1.0]
    delta_list = [0.0]

    # Temporarily calculate k for lamda_wave and T_wave calculation
    # These global vars need to be set or params passed
    # It's better to pass them via `params` dict.
    vth = 0.1 # This is used by x_wave_dispersion
    omega_ce = 2.0 # This is used by x_wave_dispersion
    omega_pe = 1.0 # This is used by x_wave_dispersion
    
    # Calculate initial physical constants to define extent
    k_init, _ = dispersion(omega_list[0], omega_ce, omega_pe, vth)
    lamda_wave_init = 2 * np.pi / k_init
    T_wave_init = (2 * np.pi) / omega_list[0]
    
    xmin_init, xmax_init = 0, 3 * lamda_wave_init
    tmin_init, tmax_init = 0, 3 * T_wave_init   

    params = {
        'extent': np.array([tmin_init, tmax_init, xmin_init, xmax_init]), 
        'num_hidden_layers': 3,
        'hidden_size': 50,
        'activation': 'tanh',
        'init': 'xavier',
        'input_encoding': True,
        'sigma': 1.0,
        'output_size': 7, # Number of outputs: phi, Bdot, Ax, Ay, Vx, Vy, N_e

        # Training parameters
        'num_epochs': 2000,
        'n_batches': 64, # Number of collocation batches per epoch per GPU
        'lr': 1e-3, 
        'lamda': 10.0,

        # Data generation parameters
        'Nt': 2000,
        'Nx': 2000,
        'Nt_coll': 200, # Total collocation points = Nt_coll * Nx_coll
        'Nx_coll': 200, 
        'num_sparse_samples': 500, #~5% of our collocation points
        'omega_list': omega_list,
        'omega_ce': [omega_ce],     # List to match expected input format
        'omega_pe': [omega_pe],     # List to match expected input format
        'vth': vth,                 # This is used by x_wave_dispersion
        'phi_amp_list': phi_amp_list,
        'delta_list': delta_list,
    }

    # Use torch.multiprocessing.spawn to launch DDP processes
    torch.multiprocessing.spawn(
        run_ddp_training,
        args=(world_size, params),
        nprocs=world_size,
        join=True
    )
    
if __name__ == "__main__":
    main()
