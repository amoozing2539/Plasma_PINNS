import numpy as np
from scipy import integrate, interpolate
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def cartesian_rec(Br_rec, Bp_rec, Bz_rec, phi):
    # converts cylindrical reconstructed components to cartesian for a z slice
    Bx_rec_slice = Br_rec * np.cos(phi) - Bp_rec * np.sin(phi)
    By_rec_slice = Br_rec * np.sin(phi) + Bp_rec * np.cos(phi)
    Bz_rec_slice = Bz_rec

    return Bx_rec_slice, By_rec_slice, Bz_rec_slice

def decompose(f_polar, M, PHI):
    norm = 2.*np.pi
    Nr, Nphi = PHI.shape
    d_phi = norm/Nphi
    
    f_m = np.zeros((2*M+1, Nr))
    f_m[0] = integrate.simpson(f_polar, axis=1, dx = d_phi)

    for m in range(1, M+1):
        # Real part
        k = 2*m - 1
        cos_term = np.cos(m * PHI)
        sin_term = np.sin(m * PHI)

        f_m[k] = integrate.simpson(f_polar * cos_term, axis = 1, dx = d_phi)
        f_m[k+1] = integrate.simpson(f_polar * sin_term, axis = 1, dx = d_phi)

    # m = 0 mode
    f_m[0] /= (2*np.pi)

    # m >= 1 modes
    f_m[1:] /= np.pi

    return f_m

def cart_to_polar_mesh(f, x, y, pts, nr, nphi):
    interp = interpolate.RegularGridInterpolator((y,x), f, bounds_error=False, fill_value=0.0)
    return interp(pts).reshape(nr, nphi)

def get_polar_comps(fx_polar, fy_polar, fz_polar, PHI, XX, YY):
    fr_polar =  fx_polar*np.cos(PHI) + fy_polar*np.sin(PHI)
    fp_polar = -fx_polar*np.sin(PHI) + fy_polar*np.cos(PHI)
    return fr_polar, fp_polar, fz_polar

def reconstruct(f_m, PHI, modes):
    Nr, Nphi = PHI.shape
    # f_rec = f_m[0][:,None]
    f_rec = f_m[0][:, None] * np.ones((Nr, Nphi))

    for m in range(1, modes+1):
        cos_term = np.cos(m * PHI)
        sin_term = np.sin(m * PHI)

        f_rec += f_m[2*m - 1][:, None] * cos_term
        f_rec += f_m[2*m][:, None] * sin_term

    return f_rec

def polar_field_to_cartesian(r, phi, f_polar, X, Y, fill_value=0.0):

    interp_rphi = interpolate.RegularGridInterpolator(
        (r, phi),
        f_polar,
        bounds_error=False,
        fill_value=fill_value
    )

    # convert experimental (X,Y) -> (r,phi)
    Rxy = np.sqrt(X**2 + Y**2)
    PHIxy = np.mod(np.arctan2(Y, X), 2*np.pi)

    # evaluate
    pts = np.stack([Rxy.ravel(), PHIxy.ravel()], axis=-1)
    f_xy = interp_rphi(pts).reshape(X.shape)

    return f_xy

def get_LAPD_domain(spatial_coords):
    x = np.unique(spatial_coords['X'].flatten())
    y = np.unique(spatial_coords['Y'].flatten())
    z = np.unique(spatial_coords['Z'].flatten())

    return x, y, z

def get_comps(Bvec, it, iz):
    Bx = Bvec[it, iz]['DATA_X']
    By = Bvec[it, iz]['DATA_Y']
    Bz = Bvec[it, iz]['DATA_Z']
    return Bx, By, Bz

def experimental_pipeline(Bx, By, Bz, spatial_coords):
    x, y, z = get_LAPD_domain(spatial_coords)
    X, Y = np.meshgrid(x, y, indexing='xy')

    RR = np.sqrt(X**2+Y**2)
    PHI = np.arctan2(Y,X) % (2*np.pi)

    Rmax = min(np.max(np.abs(x)), np.max(np.abs(y)))
    Nr   = 64
    r = np.linspace(0.0, Rmax, Nr)
    Nphi = 256
    phi = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

    R, PHI = np.meshgrid(r, phi, indexing="ij")
    Xp = R * np.cos(PHI)
    Yp = R * np.sin(PHI)
    pts = np.stack([Yp.ravel(), Xp.ravel()], axis=-1)

    Bx_polar = cart_to_polar_mesh(Bx, x, y, pts, Nr, Nphi)
    By_polar = cart_to_polar_mesh(By, x, y, pts, Nr, Nphi)
    Bz_polar_0 = cart_to_polar_mesh(Bz, x, y, pts, Nr, Nphi)

    Br_polar, Bp_polar, Bz_polar = get_polar_comps(Bx_polar, By_polar, Bz_polar_0, PHI, Xp, Yp)

    Br_m_polar = decompose(Br_polar, M=1, PHI=PHI)
    Bp_m_polar = decompose(Bp_polar, M=1, PHI=PHI)
    Bz_m_polar = decompose(Bz_polar, M=1, PHI=PHI)

    Br_rec = reconstruct(Br_m_polar, PHI, 1)
    Bp_rec = reconstruct(Bp_m_polar, PHI, 1)
    Bz_rec = reconstruct(Bz_m_polar, PHI, 1)

    Bx_rec, By_rec, Bz_rec = cartesian_rec(Br_rec, Bp_rec, Bz_rec, PHI)

    Bx_rec_cart = polar_field_to_cartesian(r, phi, Bx_rec, X, Y)
    By_rec_cart = polar_field_to_cartesian(r, phi, By_rec, X, Y)
    Bz_rec_cart = polar_field_to_cartesian(r, phi, Bz_rec, X, Y)

    return Bx_rec_cart, By_rec_cart, Bz_rec_cart, X, Y

def get_centroid(Bx, By, Bz, X, Y):
    w = Bx**2 + By**2 + Bz**2
    norm = np.sum(w)
    xc = np.sum(X * w) / norm
    yc = np.sum(Y * w) / norm

    return xc, yc


def plot(Bx, By, Bz, X, Y, savename, centroid=True):
    fig, ax = plt.subplots()
    bound = np.max(np.abs(Bz))
    pcm = ax.pcolormesh(X, Y, Bz, cmap='RdBu_r', vmax=bound, vmin=-bound)
    ax.streamplot(X, Y, Bx, By, color='black', density=0.8)
    if centroid:
        x_z, y_z = get_centroid(Bx, By, Bz, X, Y)
        ax.plot([x_z], [y_z], marker='x', color='green')
    fig.colorbar(pcm)
    fig.savefig(savename)

def get_centroids(Bvec, X, Y, it, nz):
    cents_x = np.zeros(nz)
    cents_y = np.zeros(nz)
    for iz in range(nz):
        Bx, By, Bz = get_comps(Bvec, it, iz)
        x_c, y_c = get_centroid(Bx, By, Bz, X, Y)
        cents_x[iz] = x_c
        cents_y[iz] = y_c
    return cents_x, cents_y

def animate(Bvec, iz, centroid=True):
    fig, ax = plt.subplots()

    # normalizes B_z contour to bounds in each z-slice
    all_Bz = [Bvec[it][iz]['DATA_Z'] for it, _ in enumerate(times)]

    Bx0, By0, Bz0 = get_comps(Bvec, 0, iz)
    bound = np.max(np.abs(all_Bz))
    pcm = ax.pcolormesh(X, Y, Bz0, shading="auto", vmin=-bound, vmax=bound, cmap='RdBu_r')
    strm = ax.streamplot(X, Y, Bx0, By0, color="black", density=0.8)
    
    if centroid:
        x_z, y_z = get_centroid(Bx, By, Bz, X, Y)
        (cent_artist,) = ax.plot([x_z], [y_z], marker='x', color='green')

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_aspect('equal')
    ax.set_title(f"z = {iz}")

    def update(frame):
        nonlocal strm
        strm.lines.remove()
        for patch in ax.patches:
            patch.remove()

        # new data
        Bx, By, Bz = get_comps(Bvec, frame, iz)

        pcm.set_array(Bz.ravel())
        pcm.set_clim(-bound, bound)
        strm = ax.streamplot(X, Y, Bx, By, color="black", density=0.8)
        if centroid:
            x_z, y_z = get_centroid(Bx, By, Bz, X, Y)
            cent_artist.set_data([x_z], [y_z])

    fig.colorbar(pcm, label=r'$B_z$ (slice norm)')
    anim = FuncAnimation(fig, update, frames=range(len(times)), interval=50, blit=False)
    anim.save(f'Bvec_z{iz}.mp4')

def plot_centroids(it, Bvec, X, Y, z, times):
    cents_x, cents_y = get_centroids(Bvec, X, Y, nz=len(z), it=it)
    plt.plot(z, cents_x, label=r'$x_c$')
    plt.plot(z, cents_y, label=r'$y_c$')
    plt.legend()
    plt.title(f't={times[it]}')
    plt.xlabel('z (cm)')
    plt.ylabel('distance from origin (cm)')
    plt.savefig(f'centroids_t{it}.png')

def plot_centroids_in_time(times, Bvec, X, Y, z):
    cents_x_t = np.zeros((len(times), len(z)))
    cents_y_t = np.zeros((len(times), len(z)))
    for it, time in enumerate(times):
        cents_x, cents_y = get_centroids(Bvec, X, Y, nz=len(z), it=it)
        cents_x_t[it] = cents_x
        cents_y_t[it] = cents_y
    T, Z = np.meshgrid(times, z, indexing='xy')
    fig, axs = plt.subplots(2,1)
    pcm_x = axs[0].pcolormesh(T, Z, cents_x_t.T, cmap='RdBu_r')
    pcm_y = axs[1].pcolormesh(T, Z, cents_y_t.T, cmap='RdBu_r')
    fig.colorbar(pcm_x, ax=axs[0])
    fig.colorbar(pcm_y, ax=axs[1])
    fig.savefig('centroids_t.png') 


def plot_centroids_polar(it, Bvec, X, Y, z, times, savename):
    cents_x, cents_y = get_centroids(Bvec, X, Y, nz=len(z), it=it)
    dx = np.diff(cents_x)
    dy = np.diff(cents_y)

    fig, ax = plt.subplots()
    ax.plot(cents_x, cents_y, '-k', alpha=0.5)
    ax.grid(True)
    ax.quiver(
        cents_x[:-1],
        cents_y[:-1],
        dx,
        dy,
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.003
    )

    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_title(f'Centroid trajectory with direction (t={times[it]})')
    plt.savefig(savename)

def main(args):
    # args.argparse()
    pass

if __name__ == '__main__':
    # main()
    # BASIC PARAMS
    working_dir = '/home/stsoukalas/Plasma_PINNS/mode_decomposition'
    filename = '/home/stsoukalas/shared/data/LAPD_Alfven_2025-02/b37-40.hdf5'

    # GETTING DATASETS
    f = h5py.File(filename, 'r')
    Bvec = f['B vectors/Bvec']
    times = f['B amplitudes/Timesteps']
    spatial_coords = f['Spatial grid']

    it=100
    iz=1
    x, y, z = get_LAPD_domain(spatial_coords)
    Bx, By, Bz = get_comps(Bvec, it, iz)
    Bx_rec_cart, By_rec_cart, Bz_rec_cart, X, Y = experimental_pipeline(Bx, By, Bz, spatial_coords)
    # cents_x, cents_y = get_centroids(Bvec, X, Y, nt=len(times), z=z)
    plot_centroids_polar(it, Bvec, X, Y, z, times, savename=f'{working_dir}/plots/experimental/centroids_polar_t{it}.png')
    # animate(Bvec, iz=iz)
    # plot(Bx_rec_cart-Bx, By_rec_cart-By, Bz_rec_cart-Bz, X, Y, savename=f'residual_t{it}z{iz}.png')
    # plot_centroids(it=it, Bvec=Bvec, X=X, Y=Y, z=z, times=times)
    # plot_centroids_in_time(times, Bvec, X, Y, z)
    # plot(Bx_rec_cart, By_rec_cart, Bz_rec_cart, X, Y, savename=f't{it}z{iz}.png')

