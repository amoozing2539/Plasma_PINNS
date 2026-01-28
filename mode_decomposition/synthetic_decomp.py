import numpy as np
import matplotlib.pyplot as plt
from osiris_data_interp.osiris_filename import *
from osiris_data_interp.osiris_h5data_obj import *
import scipy

def gen_meshgrids(r_vals, dr):
    r_max = r_vals[-1]

    x = np.arange(-r_max, r_max + dr, dr)
    y = np.arange(-r_max, r_max + dr, dr)

    X, Y = np.meshgrid(x,y)

    RR = np.sqrt(X**2+Y**2)
    PHI = np.arctan2(Y,X) % (2*np.pi)

    return X, Y, RR, PHI


def interp(f_before, r_before, r_after, phi_grid):
    return scipy.interpolate.interp1d(r_before, f_before, axis=0)(r_after).reshape(phi_grid.shape)

def reconstruct_plane(phi_grid, r_before, r_after, parent_field_r, parent_field_phi, parent_field_z, z_slice, M=1):

    field_r = interp(parent_field_r[(0, 're')][1:, z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)
    field_phi = interp(parent_field_phi[(0, 're')][1:, z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)
    field_z = interp(parent_field_z[(0, 're')][1:, z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)

    for m in range(1, M+1):
        field_z_re = interp(parent_field_z[(m, 're')][1:,z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)
        field_r_re = interp(parent_field_r[(m, 're')][1:,z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)
        field_phi_re = interp(parent_field_phi[(m, 're')][1:,z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)

        field_z_im = interp(parent_field_z[(m, 'im')][1:,z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)
        field_r_im = interp(parent_field_r[(m, 'im')][1:,z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)
        field_phi_im = interp(parent_field_phi[(m, 'im')][1:,z_slice], r_before=r_before, r_after=r_after, phi_grid=phi_grid)

        cos_term = np.cos(m * phi_grid)
        sin_term = np.sin(m * phi_grid)

        field_r += field_r_re * cos_term - field_r_im * sin_term
        field_phi += field_phi_re * cos_term - field_phi_im * sin_term
        field_z += field_z_re * cos_term - field_z_im * sin_term


    return field_r, field_phi, field_z

def cartesian_rec(Br_rec, Bp_rec, Bz_rec, phi):
    # converts cylindrical reconstructed components to cartesian for a z slice
    Bx_rec_slice = Br_rec * np.cos(phi) - Bp_rec * np.sin(phi)
    By_rec_slice = Br_rec * np.sin(phi) + Bp_rec * np.cos(phi)
    Bz_rec_slice = Bz_rec

    return Bx_rec_slice, By_rec_slice, Bz_rec_slice


def plot(X, Y, Bx, By, Bz, z, t, savepath=None):
    bound = np.max(np.abs(Bz))

    fig, ax = plt.subplots()

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_aspect('equal')

    pcm = ax.pcolormesh(X, Y, Bz, cmap='RdBu_r', vmin=-bound, vmax=bound)
    ax.streamplot(X, Y, Bx, By, density=0.8, color='black')
    fig.colorbar(pcm)
    title = r'Synthetic $\vec{B}, z = $' + str(round(z, 2)) + r' $c / \omega_{pe}, t = $' + str(round(t, 2)) + r' $1/\omega_{pe}$'
    ax.set_title(title)
    if savepath is not None:
        fig.savefig(savepath)


def get_data(data_dir, ndump=50):

    # getting data attrs
    last_dump_num = int(sorted(os.listdir(data_dir +'/MS/FLD/MODE-0-RE/b1_cyl_m/'))[-1][-9:][:-3])
    last_dump_simulation_idx = ndump * last_dump_num

    filename = make_osiris_filename(
        'b1', last_dump_num, data_type='field', cylindrical=True,
        mode_type='re', folder=data_dir, mode=0
    )

    # This object has all the attributes we will need
    # See osiris_h5data_obj.py for reference
    data_obj = get_cylindrical_data(filename)

    dz = data_obj.DX
    dr = data_obj.DY
    dt = data_obj.DT[0]

    nz = data_obj.NX
    nr = data_obj.NY

    z_grid = np.linspace(data_obj.AXIS1[0], data_obj.AXIS1[1], nz) # create linspaced 1D z-array
    r_grid = np.linspace(data_obj.AXIS2[0], data_obj.AXIS2[1], nr) # create linspaced 1D z-array
    t_sim = last_dump_simulation_idx * dt


    quantity = ['b1', 'b2', 'b3', 'e1', 'e2', 'e3', 'j1', 'j2', 'j3'] # -> 1 = z, 2 = r, 3 = phi
    modes = [(0, 're'), (1, 're'), (1, 'im')]
    data_dict = {quantity: {mode: np.zeros((nr, nz)) for mode in modes} for quantity in quantity}

    for quantity in quantity:
        for mode_tuple in modes:

            data_type = 'field'
            particle  = None

            filename = make_osiris_filename(
                data_name=quantity,
                timestep=last_dump_num, 
                data_type=data_type, 
                folder=data_dir, 
                particle=particle,
                cylindrical=True,
                mode=mode_tuple[0],
                mode_type=mode_tuple[1], 
            )

            data_obj = get_cylindrical_data(filename)

            data_dict[quantity][mode_tuple] = data_obj.DATA # shape: (nr, nz)

    return data_dict, z_grid, r_grid, t_sim


if __name__ == '__main__':
    working_dir = '/home/stsoukalas/Plasma_PINNS/mode_decomposition'
    data_dir = f'{working_dir}/osiris_data_interp/'
    data_dict, z_grid, r_grid, t_sim = get_data(data_dir)

    X, Y, RR, PHI = gen_meshgrids(r_grid, r_grid[1]-r_grid[0])
    r_before = r_grid[1:]
    r_flat = RR.flatten()
    r_after = np.clip(r_flat, r_before.min(), r_before.max())
    iz = len(z_grid) // 5

    br_rec, bp_rec, bz_rec = reconstruct_plane(parent_field_r = data_dict['b2'], 
                                               parent_field_phi=data_dict['b3'], 
                                               parent_field_z=data_dict['b1'], 
                                               M=1, z_slice=iz, phi_grid=PHI,
                                               r_before=r_before, r_after=r_after)


    bx, by, bz = cartesian_rec(br_rec, bp_rec, bz_rec, PHI)
    plot(X, Y, bx, by, bz, z=z_grid[iz], t=t_sim, savepath=f'{working_dir}/plots/synthetic/z{iz}.png')