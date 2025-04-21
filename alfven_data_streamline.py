import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def read_hdf5_info(filename):
    """Print detailed information about an HDF5 file structure."""
    with h5py.File(filename, 'r') as f:
        print(f"\nHDF5 File: {filename}")
        print("=" * 50)
        print("Root level groups/datasets:")
        
        def print_item(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Group):
                print(f"{indent}Group: {name}")
                # Print attributes
                for key, val in obj.attrs.items():
                    print(f"{indent}  Attr: {key} = {val}")
            elif isinstance(obj, h5py.Dataset):
                shape_str = str(obj.shape)
                dtype_str = str(obj.dtype)
                print(f"{indent}Dataset: {name}, Shape: {shape_str}, Type: {dtype_str}")
                # Print attributes
                for key, val in obj.attrs.items():
                    print(f"{indent}  Attr: {key} = {val}")
        
        f.visititems(print_item)

def read_compound_dataset(filename, dataset_path):
    """
    Read a compound dataset from an HDF5 file correctly.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    dataset_path : str
        Path to the dataset within the HDF5 file
        
    Returns:
    --------
    data : dict
        Dictionary with each field of the compound dataset
    """
    with h5py.File(filename, 'r') as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset path {dataset_path} not found in {filename}")
        
        dataset = f[dataset_path]
        
        # Handle compound dataset
        if not hasattr(dataset, 'dtype') or not dataset.dtype.names:
            raise ValueError(f"Dataset {dataset_path} is not a compound dataset")
            
        # Get field names
        field_names = dataset.dtype.names
        
        # Create a dictionary to hold the data for each field
        data = {}
        
        # Read the dataset field by field
        for field in field_names:
            # This correctly extracts a single field from the compound dataset
            field_data = dataset[field][()]
            data[field] = field_data
            
        return data, dataset.shape

def visualize_field_components(data, field_names, shape, title=None, timestep=0):
    """
    Visualize components of vector field data.
    
    Parameters:
    -----------
    data : dict
        Dictionary with field data
    field_names : list
        List of field names to visualize
    shape : tuple
        Shape of the dataset
    title : str
        Title for the plot
    timestep : int
        Timestep to visualize
    """
    # Create figure
    fig = plt.figure(figsize=(15, len(field_names) * 5))
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # For each field component
    for i, field in enumerate(field_names):
        field_data = data[field]
        
        # Create a row of plots for each z position at the given timestep
        for z in range(min(n_z, 4)):  # Limit to 4 z positions
            ax = fig.add_subplot(len(field_names), min(n_z, 4), i * min(n_z, 4) + z + 1)
            
            # Extract the 2D slice at this timestep and z position
            slice_data = field_data[timestep, z, :, :]
            
            # Plot as image
            im = ax.imshow(slice_data, origin='lower', cmap='viridis', 
                         interpolation='none', aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(field)
            
            # Set labels
            ax.set_title(f"{field} (Z={z}, T={timestep})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for suptitle
    return fig

def create_vector_field_plot(data, field_names, shape, title=None, timestep=0, z_pos=0, skip=3):
    """
    Create a vector field plot from component data.
    
    Parameters:
    -----------
    data : dict
        Dictionary with field data
    field_names : list
        List of field names (should be 3 components)
    shape : tuple
        Shape of the dataset
    title : str
        Title for the plot
    timestep : int
        Timestep to visualize
    z_pos : int
        Z position to visualize
    skip : int
        Number of points to skip for clarity
    """
    if len(field_names) < 2:
        raise ValueError("Need at least 2 components for vector plot")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    
    if title:
        plt.title(title)
        
    # Extract the 2D slices for the vector components
    x_comp = data[field_names[0]][timestep, z_pos, ::skip, ::skip]
    y_comp = data[field_names[1]][timestep, z_pos, ::skip, ::skip]
    
    # Create a grid for the vectors
    ny, nx = x_comp.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Calculate vector magnitudes for color mapping
    magnitudes = np.sqrt(x_comp**2 + y_comp**2)
    
    # Normalize vectors for better visualization
    scale = np.max(magnitudes) * 1.5
    x_comp_norm = x_comp / scale
    y_comp_norm = y_comp / scale
    
    # Create vector plot
    q = ax.quiver(X, Y, x_comp_norm, y_comp_norm, magnitudes, 
                 cmap='viridis', pivot='mid', scale=1.0)
    
    cbar = plt.colorbar(q, ax=ax)
    cbar.set_label('Magnitude')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    return fig

def create_streamline_plot(data, field_names, shape, title=None, timestep=0, z_pos=0, skip=1, density=1, linewidth=1, cmap='viridis'):
    """
    Create a streamline plot from vector field components.
    
    Parameters:
    -----------
    data : dict
        Dictionary with field data
    field_names : list
        List of field names (should be 2 components)
    shape : tuple
        Shape of the dataset
    title : str
        Title for the plot
    timestep : int
        Timestep to visualize
    z_pos : int
        Z position to visualize
    skip : int
        Number of points to skip for downsampling
    density : float
        Density of streamlines
    linewidth : float
        Linewidth of streamlines
    cmap : str
        Colormap for magnitude coloring
    """
    if len(field_names) < 2:
        raise ValueError("Need at least 2 components for streamline plot")
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    
    if title:
        plt.title(title)
        
    # Get original dimensions from the dataset shape (timesteps, z, y, x)
    _, _, original_ny, original_nx = shape
    
    # Generate indices for downsampling
    y_indices = np.arange(0, original_ny, skip)
    x_indices = np.arange(0, original_nx, skip)
    
    # Extract the 2D slices with downsampling
    x_comp = data[field_names[0]][timestep, z_pos, ::skip, ::skip]
    y_comp = data[field_names[1]][timestep, z_pos, ::skip, ::skip]
    
    # Create grid using original coordinates
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(x_comp**2 + y_comp**2)
    
    # Create streamline plot
    strm = ax.streamplot(X, Y, x_comp, y_comp, color=magnitude,
                         density=density, linewidth=linewidth, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(strm.lines, ax=ax)
    cbar.set_label('Magnitude')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    
    return fig

def create_3d_vector_field(data, field_names, shape, title=None, timestep=0, skip=5):
    """
    Create a 3D vector field visualization.
    
    Parameters:
    -----------
    data : dict
        Dictionary with field data
    field_names : list
        List of field names (should be 3 components)
    shape : tuple
        Shape of the dataset
    title : str
        Title for the plot
    timestep : int
        Timestep to visualize
    skip : int
        Number of points to skip for clarity
    """
    if len(field_names) < 3:
        raise ValueError("Need 3 components for 3D vector plot")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if title:
        plt.title(title)
    
    # Get data dimensions
    n_timesteps, n_z, n_y, n_x = shape
    
    # Create coordinate grid (simplified - assumes uniform spacing)
    Z, Y, X = np.meshgrid(
        np.arange(n_z)[::skip],
        np.arange(n_y)[::skip],
        np.arange(n_x)[::skip],
        indexing='ij'
    )
    
    # Flatten coordinate arrays for quiver3d
    x_pos = X.flatten()
    y_pos = Y.flatten()
    z_pos = Z.flatten()
    
    # Extract vector components at this timestep and downsample
    u = data[field_names[0]][timestep, ::skip, ::skip, ::skip].flatten()
    v = data[field_names[1]][timestep, ::skip, ::skip, ::skip].flatten()
    w = data[field_names[2]][timestep, ::skip, ::skip, ::skip].flatten()
    
    # Calculate magnitudes for color mapping
    magnitudes = np.sqrt(u**2 + v**2 + w**2)
    
    # Normalize vectors for better visualization
    scale = np.max(magnitudes) * 2.0
    u_norm = u / scale
    v_norm = v / scale
    w_norm = w / scale
    
    # Create 3D vector plot
    q = ax.quiver(x_pos, y_pos, z_pos, u_norm, v_norm, w_norm, 
                 length=0.5, normalize=False, cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Improve 3D view
    ax.view_init(elev=30, azim=45)
    
    return fig

def visualize_time_evolution(data, field_name, shape, z_pos=0, n_times=5):
    """
    Visualize the time evolution of a field component.
    
    Parameters:
    -----------
    data : dict
        Dictionary with field data
    field_name : str
        Field name to visualize
    shape : tuple
        Shape of the dataset
    z_pos : int
        Z position to visualize
    n_times : int
        Number of timesteps to show
    """
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Get field data
    field_data = data[field_name]
    
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Select timesteps to visualize
    timesteps = np.linspace(0, n_timesteps-1, n_times, dtype=int)
    
    # Plot each timestep
    for i, t in enumerate(timesteps):
        ax = fig.add_subplot(2, (n_times+1)//2, i+1)
        
        # Extract the 2D slice at this timestep and z position
        slice_data = field_data[t, z_pos, :, :]
        
        # Plot as image
        im = ax.imshow(slice_data, origin='lower', cmap='viridis', 
                      interpolation='none', aspect='equal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_title(f"Time {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.tight_layout()
    plt.suptitle(f"Time Evolution of {field_name} (Z={z_pos})", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

def main():
    """Main function to read and visualize HDF5 data."""
    # File paths
    b_field_file = "/home/aleotz/datasets/b37-40.hdf5"
    e_field_file = "/home/aleotz/datasets/e37-40.hdf5"
    
    # Print detailed information about the files
    try:
        read_hdf5_info(b_field_file)
        read_hdf5_info(e_field_file)
    except Exception as e:
        print(f"Error reading HDF5 info: {e}")
    
    # Read magnetic field data correctly
    try:
        # Based on the error output, we know the path is "B vectors/Bvec"
        b_data, b_shape = read_compound_dataset(b_field_file, "B vectors/Bvec")
        print(f"Successfully read B field data with shape {b_shape}")
        print(f"Components: {list(b_data.keys())}")
        
        # Visualize magnetic field components
        fig = visualize_field_components(
            b_data, 
            ['DATA_X', 'DATA_Y', 'DATA_Z'], 
            b_shape,
            title="Magnetic Field Components (Gauss)",
            timestep=0  # First timestep
        )
        plt.savefig('b_field_components.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved magnetic field components plot to 'b_field_components.png'")
        
        # Create vector field plot for magnetic field
        fig = create_vector_field_plot(
            b_data,
            ['DATA_X', 'DATA_Y'],
            b_shape,
            title="Magnetic Field Vectors (XY-plane)",
            timestep=0,  # First timestep
            z_pos=0,     # First Z position
            skip=3       # Skip every 3 points for clarity
        )
        plt.savefig('b_field_vectors.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved magnetic field vector plot to 'b_field_vectors.png'")
        
        # Create streamline plot for magnetic field
        fig = create_streamline_plot(
            b_data,
            ['DATA_X', 'DATA_Y'],
            b_shape,
            title="Magnetic Field Streamlines (XY-plane)",
            timestep=0,
            z_pos=0,
            skip=3,
            density=1,
            linewidth=1,
            cmap='viridis'
        )
        plt.savefig('b_field_streamlines.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved magnetic field streamline plot to 'b_field_streamlines.png'")
        
        # Create 3D vector field visualization
        try:
            fig = create_3d_vector_field(
                b_data,
                ['DATA_X', 'DATA_Y', 'DATA_Z'],
                b_shape,
                title="3D Magnetic Field Vectors",
                timestep=0,  # First timestep
                skip=2       # Skip every 2 points for clarity
            )
            plt.savefig('b_field_3d.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("Saved 3D magnetic field visualization to 'b_field_3d.png'")
        except Exception as e:
            print(f"Error creating 3D vector field: {e}")
        
        # Visualize time evolution
        fig = visualize_time_evolution(
            b_data,
            'DATA_Z',  # Z-component
            b_shape,
            z_pos=0,   # First Z position
            n_times=6  # Show 6 timesteps
        )
        plt.savefig('b_field_time_evolution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved time evolution plot to 'b_field_time_evolution.png'")
        
    except Exception as e:
        print(f"Error processing magnetic field data: {e}")
        import traceback
        traceback.print_exc()
    
    # Read electric field data correctly
    try:
        # Path is "E vectors/Evec" based on the error output
        e_data, e_shape = read_compound_dataset(e_field_file, "E vectors/Evec")
        print(f"Successfully read E field data with shape {e_shape}")
        print(f"Components: {list(e_data.keys())}")
        
        # Visualize electric field components
        fig = visualize_field_components(
            e_data, 
            ['DATA_X', 'DATA_Y', 'DATA_Z'], 
            e_shape,
            title="Electric Field Components (V/cm)",
            timestep=0  # First timestep
        )
        plt.savefig('e_field_components.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved electric field components plot to 'e_field_components.png'")
        
        # Create vector field plot for electric field
        fig = create_vector_field_plot(
            e_data,
            ['DATA_X', 'DATA_Y'],
            e_shape,
            title="Electric Field Vectors (XY-plane)",
            timestep=0,  # First timestep
            z_pos=0,     # First Z position
            skip=3       # Skip every 3 points for clarity
        )
        plt.savefig('e_field_vectors.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved electric field vector plot to 'e_field_vectors.png'")
        
        # Create streamline plot for electric field
        fig = create_streamline_plot(
            e_data,
            ['DATA_X', 'DATA_Y'],
            e_shape,
            title="Electric Field Streamlines (XY-plane)",
            timestep=0,
            z_pos=0,
            skip=3,
            density=1,
            linewidth=1,
            cmap='viridis'
        )
        plt.savefig('e_field_streamlines.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved electric field streamline plot to 'e_field_streamlines.png'")
        
    except Exception as e:
        print(f"Error processing electric field data: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to calculate ExB drift
    try:
        # Since we have the data as separate components, we can calculate ExB drift
        timestep = 0  # Use first timestep
        z_pos = 0     # Use first Z position
        
        # Get field components at this timestep and z position
        ex = e_data['DATA_X'][timestep, z_pos]
        ey = e_data['DATA_Y'][timestep, z_pos]
        ez = e_data['DATA_Z'][timestep, z_pos]
        
        bx = b_data['DATA_X'][timestep, z_pos]
        by = b_data['DATA_Y'][timestep, z_pos]
        bz = b_data['DATA_Z'][timestep, z_pos]
        
        # Calculate B magnitude
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate ExB drift velocity components
        # vExB = (E×B)/B²
        exb_x = (ey * bz - ez * by) / (b_mag**2 + 1e-10)  # Add small value to avoid division by zero
        exb_y = (ez * bx - ex * bz) / (b_mag**2 + 1e-10)
        exb_z = (ex * by - ey * bx) / (b_mag**2 + 1e-10)
        
        # Create figure for ExB drift
        fig = plt.figure(figsize=(12, 10))
        
        # Plot ExB x-component
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(exb_x, origin='lower', cmap='RdBu_r', 
                        interpolation='none', aspect='equal')
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('ExB X-component')
        
        # Plot ExB y-component
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(exb_y, origin='lower', cmap='RdBu_r', 
                        interpolation='none', aspect='equal')
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('ExB Y-component')
        
        # Plot ExB z-component
        ax3 = fig.add_subplot(223)
        im3 = ax3.imshow(exb_z, origin='lower', cmap='RdBu_r', 
                        interpolation='none', aspect='equal')
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('ExB Z-component')
        
        # Plot ExB magnitude
        exb_mag = np.sqrt(exb_x**2 + exb_y**2 + exb_z**2)
        ax4 = fig.add_subplot(224)
        im4 = ax4.imshow(exb_mag, origin='lower', cmap='viridis', 
                        interpolation='none', aspect='equal')
        plt.colorbar(im4, ax=ax4)
        ax4.set_title('ExB Magnitude')
        
        plt.tight_layout()
        plt.suptitle('ExB Drift Velocity', fontsize=16)
        plt.subplots_adjust(top=0.9)
        
        plt.savefig('exb_drift.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved ExB drift plot to 'exb_drift.png'")
        
        # Create vector plot of ExB drift
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        
        # Create a grid for the vectors
        skip = 3  # Skip every 3 points for clarity
        ny, nx = exb_x.shape
        X, Y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
        
        # Downsample for clarity
        exb_x_ds = exb_x[::skip, ::skip]
        exb_y_ds = exb_y[::skip, ::skip]
        exb_mag_ds = exb_mag[::skip, ::skip]
        
        # Normalize vectors for better visualization
        scale = np.max(exb_mag_ds) * 1.5
        exb_x_norm = exb_x_ds / (scale + 1e-10)
        exb_y_norm = exb_y_ds / (scale + 1e-10)
        
        # Create vector plot
        q = ax.quiver(X, Y, exb_x_norm, exb_y_norm, exb_mag_ds, 
                     cmap='viridis', pivot='mid', scale=1.0)
        
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label('ExB Drift Magnitude')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('ExB Drift Velocity Vectors')
        ax.set_aspect('equal')
        
        plt.savefig('exb_drift_vectors.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved ExB drift vector plot to 'exb_drift_vectors.png'")
        
        # Create streamline plot for ExB drift
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        
        # Create grid using original coordinates
        X, Y = np.meshgrid(np.arange(0, nx, skip), np.arange(0, ny, skip))
        
        # Create streamline plot
        strm = ax.streamplot(X, Y, exb_x_ds, exb_y_ds, color=exb_mag_ds,
                             density=1, linewidth=1, cmap='viridis')
        
        cbar = plt.colorbar(strm.lines, ax=ax)
        cbar.set_label('ExB Drift Magnitude')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('ExB Drift Streamlines')
        ax.set_aspect('equal')
        
        plt.savefig('exb_drift_streamlines.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved ExB drift streamline plot to 'exb_drift_streamlines.png'")
        
    except Exception as e:
        print(f"Error calculating ExB drift: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()