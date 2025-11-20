from scipy.interpolate import griddata
import numpy as np
from matplotlib import colors

def get_LAPD_domain(spatial_coords):
    x = np.unique(spatial_coords['X'].flatten())
    y = np.unique(spatial_coords['Y'].flatten())
    z = np.unique(spatial_coords['Z'].flatten())

    return x, y, z

def gen_3D_Bvec(time_index, Bvec, x, y, z, M):
    # stacks all z slices for each B component at time_index
    Bx_3d = np.stack([Bvec[time_index][z_ind]['DATA_X'] for z_ind in range(len(z))], axis=2)
    By_3d = np.stack([Bvec[time_index][z_ind]['DATA_Y'] for z_ind in range(len(z))], axis=2)
    Bz_3d = np.stack([Bvec[time_index][z_ind]['DATA_Z'] for z_ind in range(len(z))], axis=2)

    return Bx_3d, By_3d, Bz_3d

def get_rec_cartesian_domain(r_vals, phi):
    # gets the spatial domain in reconstruction (different from LAPD spatial grid)
    RR, PHI = np.meshgrid(r_vals, phi, indexing='xy')
    X_rec = RR * np.cos(PHI)
    Y_rec = RR * np.sin(PHI)

    return X_rec, Y_rec

def gen_phi(nphi, min_phi=-np.pi, max_phi=np.pi):
    return np.linspace(min_phi, max_phi, nphi)

def pred_interp(X_rec, Y_rec, X_true, Y_true, F_rec):
    # interpolates reconstruction domain onto LAPD domain
    points = np.column_stack((X_rec.ravel(), Y_rec.ravel()))
    values = F_rec.T.ravel()
    return griddata(points, values, (X_true, Y_true), method='linear').T

def norm_L2_error(Fx_true, Fy_true, Fz_true, Fx_pred, Fy_pred, Fz_pred):
    # calculates normalized L2 error on not nan values of fields
    valid_mask = ~np.isnan(Fx_pred) & ~np.isnan(Fy_pred) & ~np.isnan(Fz_pred)
    L2 = np.sum(((Fx_true-Fx_pred)**2+(Fy_true-Fy_pred)**2+(Fz_true-Fz_pred)**2)[valid_mask])
    norm = np.sum((Fx_true**2 + Fy_true**2 + Fz_true**2)[valid_mask])

    return L2/norm

def energy(Bx_pred, By_pred, Bz_pred, dx, dy):
    # doesn't include constants of field energy (1/2mu)

    return np.nansum(Bx_pred**2+By_pred**2+Bz_pred**2)*dx*dy

def cartesian_rec(Br_rec, Bp_rec, Bz_rec, z_ind, phi):
    # converts cylindrical reconstructed components to cartesian for a z slice
    Bx_rec_slice = Br_rec[z_ind] * np.cos(phi) - Bp_rec[z_ind] * np.sin(phi)
    By_rec_slice = Br_rec[z_ind] * np.sin(phi) + Bp_rec[z_ind] * np.cos(phi)
    Bz_rec_slice = Bz_rec[z_ind]

    return Bx_rec_slice, By_rec_slice, Bz_rec_slice

def get_color_norm(F_true, F_pred):
    # gives global color norm for PCM color bars in final decomp figure
    # F_true_masked is returned here as an afterthought, as I need it for plotting as well
    F_true_masked = np.where(np.isnan(F_pred), np.nan, F_true)
    F_color_norm = colors.Normalize(vmin=np.nanmin(F_true_masked), vmax=np.nanmax(F_true_masked))
    return F_true_masked, F_color_norm

def B_true_slice(Bvec, z_slice, t_index):
    # returns "true" components at given z_slice and t_index
    Bx_true = Bvec[t_index][z_slice]["DATA_X"]
    By_true = Bvec[t_index][z_slice]["DATA_Y"]
    Bz_true = Bvec[t_index][z_slice]["DATA_Z"]

    return Bx_true, By_true, Bz_true