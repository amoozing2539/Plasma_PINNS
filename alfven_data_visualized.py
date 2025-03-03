import h5py 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

##read the hd5 file####
def read_hdf5_maya_data(file_path, data_group_name, parameter_indices=None, time_index=0):
    """
    Read data from HDF5 file following the Maya visualization structure.
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
    data_group_name : str
        Name of the data group to read (e.g., 'Bdot vectors', 'density')
    parameter_indices : list or None
        List of parameter indices [p1_idx, p2_idx, p3_idx, p4_idx] or None
    time_index : int
        Index of the timestep to read
        
    Returns:
    --------
    data : ndarray
        3D volume data at specified parameter indices and time
    spatial_grid : ndarray
        Spatial grid coordinates (x, y, z) for each data point
    metadata : dict
        Dictionary containing metadata about the dataset
    """
    with h5py.File(file_path, 'r') as f:
        # Read spatial grid
        spatial_grid = f['Spatial grid'][:]
        
        # Get metadata
        metadata = {}
        
        # Check if data group exists
        if data_group_name not in f:
            raise ValueError(f"Data group '{data_group_name}' not found in file")
            
        # Get data group attributes
        group = f[data_group_name]
        for attr in ['Name', 'Symbol', 'Units', 'Type']:
            if attr in group.attrs:
                metadata[attr.lower()] = group.attrs[attr]
                
        # Get timesteps
        timesteps = group['Timesteps'][:]
        if time_index >= len(timesteps):
            raise ValueError(f"Time index {time_index} out of range (max: {len(timesteps)-1})")
        metadata['timestep_value'] = timesteps[time_index]
        
        # Determine parameter part of the dataset name
        param_part = ""
        if parameter_indices is not None:
            # Check for parameters
            param_names = []
            param_values = []
            
            for i, param_idx in enumerate(parameter_indices):
                param_name = f"Parameter{i+1}"
                if param_name in f and param_idx is not None:
                    param_dataset = f[param_name]
                    if param_idx >= len(param_dataset):
                        raise ValueError(f"Parameter index {param_idx} out of range for {param_name}")
                    param_names.append(param_name)
                    param_values.append(param_dataset[param_idx])
                    param_part += f".{param_idx}"
                    
            metadata['parameters'] = dict(zip(param_names, param_values))
        
        # Construct dataset name
        symbol = metadata.get('symbol', data_group_name)
        dataset_name = f"{symbol}{param_part}"
        
        # Access dataset
        if dataset_name not in group:
            raise ValueError(f"Dataset '{dataset_name}' not found in group '{data_group_name}'")
            
        dataset = group[dataset_name]
        
        # Extract data for the specific time index
        data = dataset[:, :, :, time_index]
        
    return data, spatial_grid, metadata

def visualize_scalar_field(data, spatial_grid, title=None, colormap='viridis'):
    """
    Visualize a 3D scalar field using slices.
    
    Parameters:
    -----------
    data : ndarray
        3D volume data (scalar field)
    spatial_grid : ndarray
        Spatial grid coordinates (x, y, z) for each data point
    title : str or None
        Plot title
    colormap : str
        Matplotlib colormap name
    """
    # Extract dimensions
    nx, ny, nz = data.shape
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # Plot slices through the middle of the volume
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2
    
    # XY slice
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(data[:, :, mid_z], cmap=colormap, origin='lower')
    ax1.set_title(f'XY Slice (Z={mid_z})')
    plt.colorbar(im1, ax=ax1)
    
    # XZ slice
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(data[:, mid_y, :], cmap=colormap, origin='lower')
    ax2.set_title(f'XZ Slice (Y={mid_y})')
    plt.colorbar(im2, ax=ax2)
    
    # YZ slice
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(data[mid_x, :, :], cmap=colormap, origin='lower')
    ax3.set_title(f'YZ Slice (X={mid_x})')
    plt.colorbar(im3, ax=ax3)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def visualize_vector_field(data, spatial_grid, title=None, sample_rate=5):
    """
    Visualize a 3D vector field using quiver plots at slices.
    
    Parameters:
    -----------
    data : ndarray
        Vector field data with shape (nx, ny, nz, 3)
    spatial_grid : ndarray
        Spatial grid coordinates (x, y, z) for each data point
    title : str or None
        Plot title
    sample_rate : int
        Sample every N points for clarity
    """
    # Extract dimensions
    nx, ny, nz = data.shape[:3]
    
    # Reshape spatial grid for easier indexing
    x = spatial_grid[:, :, :, 0].reshape(nx, ny, nz)
    y = spatial_grid[:, :, :, 1].reshape(nx, ny, nz)
    z = spatial_grid[:, :, :, 2].reshape(nx, ny, nz)
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # Mid-points
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2
    
    # XY slice
    ax1 = fig.add_subplot(131)
    X, Y = np.meshgrid(
        np.arange(0, nx, sample_rate),
        np.arange(0, ny, sample_rate)
    )
    u = data[::sample_rate, ::sample_rate, mid_z, 0]
    v = data[::sample_rate, ::sample_rate, mid_z, 1]
    ax1.quiver(X, Y, u, v)
    ax1.set_title(f'XY Slice (Z={mid_z})')
    ax1.set_aspect('equal')
    
    # XZ slice
    ax2 = fig.add_subplot(132)
    X, Z = np.meshgrid(
        np.arange(0, nx, sample_rate),
        np.arange(0, nz, sample_rate)
    )
    u = data[::sample_rate, mid_y, ::sample_rate, 0]
    w = data[::sample_rate, mid_y, ::sample_rate, 2]
    ax2.quiver(X, Z, u, w)
    ax2.set_title(f'XZ Slice (Y={mid_y})')
    ax2.set_aspect('equal')
    
    # YZ slice
    ax3 = fig.add_subplot(133)
    Y, Z = np.meshgrid(
        np.arange(0, ny, sample_rate),
        np.arange(0, nz, sample_rate)
    )
    v = data[mid_x, ::sample_rate, ::sample_rate, 1]
    w = data[mid_x, ::sample_rate, ::sample_rate, 2]
    ax3.quiver(Y, Z, v, w)
    ax3.set_title(f'YZ Slice (X={mid_x})')
    ax3.set_aspect('equal')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def plot_magnetic_field_and_potential(hdf5_file, time_index=0, parameter_indices=None):
    """
    Plot magnetic field vectors and potential scalar field from HDF5 file.
    
    Parameters:
    -----------
    hdf5_file : str
        Path to the HDF5 file
    time_index : int
        Index of the timestep to visualize
    parameter_indices : list or None
        List of parameter indices [p1_idx, p2_idx, p3_idx, p4_idx] or None
    """
    # Load magnetic field data (assuming "Bdot vectors" is the field group)
    try:
        bdot_data, spatial_grid, bdot_metadata = read_hdf5_maya_data(
            hdf5_file, 
            "Bdot vectors", 
            parameter_indices=parameter_indices, 
            time_index=time_index
        )
        
        # Create title with metadata
        time_value = bdot_metadata.get('timestep_value', time_index)
        title = f"Magnetic Field (Bdot) at t={time_value}"
        if 'units' in bdot_metadata:
            title += f" [{bdot_metadata['units']}]"
            
        # Visualize magnetic field
        visualize_vector_field(bdot_data, spatial_grid, title=title)
        plt.savefig('magnetic_field.png', dpi=300)
        plt.close()
        
        print(f"Magnetic field visualization saved as 'magnetic_field.png'")
    except Exception as e:
        print(f"Error plotting magnetic field: {e}")
    
    # Load potential data (assuming field name is "potential" - adjust if different)
    try:
        potential_data, spatial_grid, potential_metadata = read_hdf5_maya_data(
            hdf5_file, 
            "potential", 
            parameter_indices=parameter_indices, 
            time_index=time_index
        )
        
        # Create title with metadata
        time_value = potential_metadata.get('timestep_value', time_index)
        title = f"Scalar Potential at t={time_value}"
        if 'units' in potential_metadata:
            title += f" [{potential_metadata['units']}]"
            
        # Visualize potential field
        visualize_scalar_field(potential_data, spatial_grid, title=title)
        plt.savefig('potential_field.png', dpi=300)
        plt.close()
        
        print(f"Potential field visualization saved as 'potential_field.png'")
    except Exception as e:
        print(f"Error plotting potential field: {e}")

def compare_files(file1, file2, data_group, time_index=0, parameter_indices=None):
    """
    Compare the same data group from two different HDF5 files.
    
    Parameters:
    -----------
    file1, file2 : str
        Paths to the HDF5 files
    data_group : str
        Name of the data group to compare
    time_index : int
        Index of the timestep to visualize
    parameter_indices : list or None
        List of parameter indices [p1_idx, p2_idx, p3_idx, p4_idx] or None
    """
    # Load data from both files
    data1, grid1, meta1 = read_hdf5_maya_data(
        file1, data_group, parameter_indices=parameter_indices, time_index=time_index
    )
    
    data2, grid2, meta2 = read_hdf5_maya_data(
        file2, data_group, parameter_indices=parameter_indices, time_index=time_index
    )
    
    # Calculate difference
    diff = data2 - data1
    
    # Create figure
    fig = plt.figure(figsize=(15, 5))
    
    # Mid-point for visualization
    mid_z = data1.shape[2] // 2
    
    # First dataset
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(data1[:, :, mid_z], cmap='viridis', origin='lower')
    ax1.set_title(f'File 1: {data_group}')
    plt.colorbar(im1, ax=ax1)
    
    # Second dataset
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(data2[:, :, mid_z], cmap='viridis', origin='lower')
    ax2.set_title(f'File 2: {data_group}')
    plt.colorbar(im2, ax=ax2)
    
    # Difference
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(diff[:, :, mid_z], cmap='RdBu_r', origin='lower')
    ax3.set_title('Difference (File2 - File1)')
    plt.colorbar(im3, ax=ax3)
    
    # Overall title
    time1 = meta1.get('timestep_value', time_index)
    time2 = meta2.get('timestep_value', time_index)
    title = f"Comparison of {data_group} at t={time1} vs t={time2}"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{data_group}.png', dpi=300)
    plt.close()
    
    print(f"Comparison visualization saved as 'comparison_{data_group}.png'")
    
    return np.max(np.abs(diff)), np.mean(np.abs(diff))

def main(): 
    # %matplotlib inline
    b37 = "b37-40.hdf5"
    e37 = "e37-40.hdf5"
    
    # Plot magnetic field and potential for first file
    plot_magnetic_field_and_potential(
        b37,
        time_index=0,  # First timestep
        parameter_indices=[0, 0, 0, 0]  # First value of each parameter
    )
    
    # Compare magnetic field between two files
    max_diff, mean_diff = compare_files(
        b37,
        e37,
        "Bdot vectors",  # Compare magnetic fields
        time_index=0,
        parameter_indices=[0, 0, 0, 0]
    )
    
    print(f"Maximum difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")




if __name__ == '__main__':
  main()