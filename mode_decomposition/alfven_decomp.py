from scipy.interpolate import griddata
import numpy as np
from matplotlib import colors
import scipy
import h5py
from gabi_decomp import *


def get_LAPD_domain(spatial_coords):
    x = np.unique(spatial_coords['X'].flatten())
    y = np.unique(spatial_coords['Y'].flatten())
    z = np.unique(spatial_coords['Z'].flatten())

    return x, y, z

def gen_3D_Bvec(time_index, Bvec, x, y, z, M, low_pass, sigma):
    # stacks all z slices for each B component at time_index

    Bx_list = []
    By_list = []
    Bz_list = []

    for z_ind in range(len(z)):
        Bx_slice, By_slice, Bz_slice = B_true_slice(Bvec, z_ind, time_index, low_pass=low_pass, sigma=sigma)
        Bx_list.append(Bx_slice)
        By_list.append(By_slice)
        Bz_list.append(Bz_slice)

    Bx_3d = np.stack(Bx_list, axis=2)
    By_3d = np.stack(By_list, axis=2)
    Bz_3d = np.stack(Bz_list, axis=2)

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

def B_true_slice(Bvec, z_slice, t_index, low_pass, sigma):
    # returns "true" components at given z_slice and t_index
    Bx_true = Bvec[t_index][z_slice]["DATA_X"]
    By_true = Bvec[t_index][z_slice]["DATA_Y"]
    Bz_true = Bvec[t_index][z_slice]["DATA_Z"]

    if low_pass:
        Bx_true = scipy.ndimage.gaussian_filter(Bx_true, sigma=sigma)
        By_true = scipy.ndimage.gaussian_filter(By_true, sigma=sigma)
        Bz_true = scipy.ndimage.gaussian_filter(Bz_true, sigma=sigma)

    return Bx_true, By_true, Bz_true


if __name__ == '__main__':
    # BASIC PARAMS
    filename = '/home/stsoukalas/shared/data/LAPD_ALfven_2025-02/b37-40.hdf5'


    f_raw = h5py.File(filename, 'r')
    Bvec = f_raw['B vectors/Bvec']
    times = f_raw['B amplitudes/Timesteps']
    spatial_coords = f_raw['Spatial grid']

    M = 10
    x, y, z = get_LAPD_domain(spatial_coords)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    Nx = len(x)
    Ny = len(y)
    Nt = len(times)
    Nz = len(z)
    nphi = 256
    phi = gen_phi(nphi)
    # write metadata
    with h5py.File("mode_profiles.hdf5", "w") as f:

        f.create_dataset('coords/t', data=times)
        f.create_dataset("coords/x", data=x)
        f.create_dataset("coords/y", data=y)
        f.create_dataset("coords/z", data=z)

        # placeholder for r_vals (will resize in a moment bc Nr is not known until interpolation)
        f.create_dataset("coords/r_vals", shape=(0,), maxshape=(None,), dtype="f8")

        # f.create_dataset('rec_coords/x_rec', shape=(0,), maxshape=(None,), dtype="f8")
        # f.create_dataset('rec_coords/y_rec', shape=(0,), maxshape=(None,), dtype="f8")
        
        # f.create_dataset('rec_coords/x', shape=(0,), maxshape=(None,), dtype="f8")

    # get Nr on t=0
    Bx_3d, By_3d, Bz_3d = gen_3D_Bvec(
        time_index=0, Bvec=Bvec,
        x=x, y=y, z=z, M=M, low_pass=False, sigma=0
    )

    r_vals, Br0, Bp0, Bz0 = make_modes_interp(
        Bx_3d, By_3d, Bz_3d,
        x, y, z, modes=M
    )

    X_rec, Y_rec = get_rec_cartesian_domain(r_vals, phi)

    Nr = len(r_vals)
    Nm = 2*M + 1
    # Nx_rec = len(X_rec.ravel())
    # Ny_rec = len(Y_rec.ravel())

    # -----------------------------------------------
    #   3. Create the mode datasets NOW that Nr is known
    # -----------------------------------------------
    with h5py.File("mode_profiles.hdf5", "r+") as f:

        # store r-values
        f["coords/r_vals"].resize((Nr,))
        f["coords/r_vals"][...] = r_vals

        # f['rec_coords/x_rec'].resize((Nx_rec,))
        # f['rec_coords/y_rec'].resize((Ny_rec,))

        # f['rec_coords/x_rec'][...] = X_rec
        # f['rec_coords/y_rec'][...] = Y_rec

        # Create datasets: (Nt, Nz, Nr, Nmodes)
        f.create_dataset("modes/Br", shape=(Nt, Nz, Nr, Nm), dtype="f4")
        f.create_dataset("modes/Bp", shape=(Nt, Nz, Nr, Nm), dtype="f4")
        f.create_dataset("modes/Bz", shape=(Nt, Nz, Nr, Nm), dtype="f4")

        f.create_dataset('rec_raw/Br', shape=(Nt, Nz, Nr, nphi, Nm), dtype="f4")
        f.create_dataset('rec_raw/Bp', shape=(Nt, Nz, Nr, nphi, Nm), dtype="f4")
        f.create_dataset('rec_raw/Bz', shape=(Nt, Nz, Nr, nphi, Nm), dtype="f4")

        f.create_dataset('rec_interp/Br', shape=(Nt, Nz, Nx, Ny, Nm), dtype="f4")
        f.create_dataset('rec_interp/Bp', shape=(Nt, Nz, Nx, Ny, Nm), dtype="f4")
        f.create_dataset('rec_interp/Bz', shape=(Nt, Nz, Nx, Ny, Nm), dtype="f4")

        dBr = f["modes/Br"]
        dBp = f["modes/Bp"]
        dBz = f["modes/Bz"]

        # save first timestep
        dBr[0] = Br0
        dBp[0] = Bp0
        dBz[0] = Bz0

        for m in range(M+1):
            Br_rec0, Bp_rec0, Bz_rec0 = reconstruct_modes(Br0, Bp0, Bz0, phi, m)
            f["rec_raw/Br"][0][:,:,:,m] = Br_rec0
            f["rec_raw/Bp"][0][:,:,:,m] = Bp_rec0
            f["rec_raw/Bz"][0][:,:,:,m] = Bz_rec0

        # -----------------------------------------------
        #   4. Loop remaining timesteps normally
        # -----------------------------------------------
        for it in range(1, Nt):
            Bx_3d, By_3d, Bz_3d = gen_3D_Bvec(
                time_index=it, Bvec=Bvec,
                x=x, y=y, z=z, M=M, low_pass=False, sigma=0
            )

            r_vals, Br, Bp, Bz = make_modes_interp(
                Bx_3d, By_3d, Bz_3d,
                x, y, z, modes=M
            )

            assert Br.shape == (Nz, Nr, nphi) # sanity check

            dBr[it] = Br
            dBp[it] = Bp
            dBz[it] = Bz

            for m in range(M+1):
                Br_rec, Bp_rec, Bz_rec = reconstruct_modes(Br, Bp, Bz, phi, m)
                f["rec_raw/Br"][it][:,:,:,m] = Br_rec
                f["rec_raw/Bp"][it][:,:,:,m] = Bp_rec
                f["rec_raw/Bz"][it][:,:,:,m] = Bz_rec

                for z_slice, _ in enumerate(z):
                    Br_interp = pred_interp(X_rec, Y_rec, X, Y, Br_rec[z_slice])
                    Bp_interp = pred_interp(X_rec, Y_rec, X, Y, Bp_rec[z_slice])
                    Bz_interp = pred_interp(X_rec, Y_rec, X, Y, Bz_rec[z_slice])

                    f['rec_interp/Br'][it][z_slice,:,:,m] = Br_interp
                    f['rec_interp/Bp'][it][z_slice,:,:,m] = Bp_interp
                    f['rec_interp/Bz'][it][z_slice,:,:,m] = Bz_interp

        # optional metadata
        f.attrs["n_modes"] = M
        f.attrs["n_timesteps"] = Nt
        f.attrs["description"] = "Mode decomposition profiles over z slices and timesteps"