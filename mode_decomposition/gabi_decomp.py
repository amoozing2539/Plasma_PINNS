import numpy as np
import os, h5py, gc
from scipy import integrate, interpolate

def slice_decompose(Fs, modes, ext, cart_vals, cyl_vals, AngAng):
    norm = 2.*np.pi
    d_phi = norm/ext[0]
    X, Y = cart_vals
    XX, YY = cyl_vals
    
    fr_m = np.zeros((ext[1], 2*modes+1))
    fp_m = np.zeros((ext[1], 2*modes+1))
    fz_m = np.zeros((ext[1], 2*modes+1))

    func_f1 = interpolate.RectBivariateSpline(X, Y, Fs[0])
    func_f2 = interpolate.RectBivariateSpline(X, Y, Fs[1])
    func_f3 = interpolate.RectBivariateSpline(X, Y, Fs[2])

    fx = func_f1(XX.flatten(), YY.flatten(), grid = False).reshape(ext)
    fy = func_f2(XX.flatten(), YY.flatten(), grid = False).reshape(ext)

    fr =  fx*np.cos(AngAng) + fy*np.sin(AngAng)
    fp = -fx*np.sin(AngAng) + fy*np.cos(AngAng)
    fz = func_f3(XX.flatten(), YY.flatten(), grid = False).reshape(ext)

    # zeroth mode
    fr_m[:, 0] = integrate.simpson(fr, axis=0, dx = d_phi)
    fp_m[:, 0] = integrate.simpson(fp, axis=0, dx = d_phi)
    fz_m[:, 0] = integrate.simpson(fz, axis=0, dx = d_phi)

    for m in range(1, modes+1):
        # Real part
        k = 2*m - 1
        cos_th = np.cos(m * AngAng)
        sin_th = np.sin(m * AngAng)

        fr_m[:, k] = integrate.simpson(fr * cos_th, axis = 0, dx = d_phi)
        fp_m[:, k] = integrate.simpson(fp * cos_th, axis = 0, dx = d_phi)
        fz_m[:, k] = integrate.simpson(fz * cos_th, axis = 0, dx = d_phi)

        fr_m[:, k+1] = integrate.simpson(fr * sin_th, axis = 0, dx = d_phi)
        fp_m[:, k+1] = integrate.simpson(fp * sin_th, axis = 0, dx = d_phi)
        fz_m[:, k+1] = integrate.simpson(fz * sin_th, axis = 0, dx = d_phi)

    # m = 0 mode
    fr_m[:, 0] /= (2*np.pi)
    fp_m[:, 0] /= (2*np.pi)
    fz_m[:, 0] /= (2*np.pi)

    # m >= 1 modes
    fr_m[:, 1:] /= np.pi
    fp_m[:, 1:] /= np.pi
    fz_m[:, 1:] /= np.pi

    return(fr_m, fp_m, fz_m)

def make_modes_interp(Fx_3d, Fy_3d, Fz_3d, X, Y, Z, modes = 2, NP = 125):
    '''
    Fx_3d, Fy_3d, Fz_3d = 3D arrays with indexes (i, j, k) corresponding to position (x_i, y_j, z_k)
    X, Y, Z = 1D arrays corresponding to (x, y, z) values in 3D arrays
    modes = # azimuthal modes to calculate 
    NP = number azimuthal points in phi, should be ~ 10-100x (# modes) for good/smooth statistics
    '''
    
    dr = Y[1]-Y[0]

    Rmax = (Y[-1]-Y[0])/2. 
    R_vals = np.arange(0, Rmax, dr)

    NR = len(R_vals)
    NZ = len(Z)

    int_angles = np.linspace(-np.pi, np.pi, NP)
    RR, AngAng = np.meshgrid(R_vals, int_angles)
    XX, YY = RR*np.cos(AngAng), RR*np.sin(AngAng)

    Fr = np.zeros((NZ, NR, 2*modes+1))
    Fp = np.zeros((NZ, NR, 2*modes+1))
    Fz = np.zeros((NZ, NR, 2*modes+1))
    
    print(Fx_3d.shape, Fy_3d.shape, Fz_3d.shape)
    for i in range(NZ):
        fr_m, fp_m, fz_m = slice_decompose([Fx_3d[:, :, i], Fy_3d[:, :, i], Fz_3d[:, :, i]], modes, 
                                               (NP, NR), [X, Y], [XX, YY], AngAng)
        #fr_m, fp_m, fz_m = slice_decompose(i)
        Fr[i, :, :] = fr_m[:, :]
        Fp[i, :, :] = fp_m[:, :]
        Fz[i, :, :] = fz_m[:, :]
        
    return(R_vals, Fr, Fp, Fz)

def reconstruct_modes(Fr, Fp, Fz, phi_grid, modes):
    NZ, NR, _ = Fr.shape
    NP = len(phi_grid)

    Fr_rec = np.zeros((NZ, NR, NP))
    Fp_rec = np.zeros((NZ, NR, NP))
    Fz_rec = np.zeros((NZ, NR, NP))


    # constant mode
    Fr_rec += Fr[:, :, 0][:, :, None]
    Fp_rec += Fp[:, :, 0][:, :, None]
    Fz_rec += Fz[:, :, 0][:, :, None]

    # higher modes 
    for m in range(1, modes+1):
        cos_term = np.cos(m * phi_grid)[None, None, :] 
        sin_term = np.sin(m * phi_grid)[None, None, :]

        Fr_rec += Fr[:, :, 2*m - 1][:, :, None] * cos_term
        Fr_rec += Fr[:, :, 2*m][:, :, None] * sin_term

        Fp_rec += Fp[:, :, 2*m - 1][:, :, None] * cos_term
        Fp_rec += Fp[:, :, 2*m][:, :, None] * sin_term

        Fz_rec += Fz[:, :, 2*m - 1][:, :, None] * cos_term
        Fz_rec += Fz[:, :, 2*m][:, :, None] * sin_term

    return Fr_rec, Fp_rec, Fz_rec