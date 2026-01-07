import numpy as np
import h5py

class osiris_cartesian_data_object(object):
    """
    Create Osiris Data object from h5py file object made with cartesian/default simulation parameters
        - so x1 = x, x2 = y, etc.
      
    Attributes:
    
        General DATA attributes
            DIM (int):              number dimensions of the data object
            DATA (arr):             data from the file, either a 1D, 2D or 3D array
            DATA_UNITS (str):       units for the data to use in latex math mode
            DATA_PAR_NAME (str):    full label for the data (incuding particle type) to use in latex math mode
            DATA_NAME (str):        full label for the data to use in latex math mode
            TIME (flt):             output time from the simulation
            TIME_UNITS (str):       units for the time to use in latex math mode
            DT (flt):               dt between timesteps in simulation
        
        AXIS atributes (repeated for each dimension - example with AXIS1)
            AXIS1 (flt):            array with the endpoints of the axis
            AXIS1_UNITS (str):      units for the axis1 data
            AXIS1_NAME (str):       full label for axis1 to use in latex math mode
            NX (int):               number of data points in axis1 direction
            DX (flt):               spacing between subsequent datapoints
            X (1D arr):             array of axis1 values, midpoint of grid (from AXIS1[0] + DX/2 to AXIS1[1] + DX/2)
            X_MESH (1D arr):        array of axis1 values, w/ length=NX+1 (from AXIS1[0] to AXIS1[1], with spacing DX)
            
    Methods:
        __init__(self, h5_file)
            constructor for osiris_cartesian_data_object class
            
            Parameters:
            h5_file (h5py file object): File with the data, already opened using h5_file = h5py.File(filename,'r')
    """

    def __init__(self, h5_file, slice_arr, inner = False, zlen = None, no_data = False, dtype = 'f4'):
        
        keys = [key for key in h5_file.keys()]
        if keys[-1]=='SIMULATION': keys.pop() #want the data name to be the -1 element
        
        self.SHAPE = h5_file[keys[-1]].shape
        self.DIM = len(self.SHAPE)
        
        try:
            self.DATA_UNITS = h5_file[keys[-1]].attrs['UNITS'][0].decode("utf-8")
        except KeyError:
            try:
                self.DATA_UNITS = h5_file.attrs['UNITS'][0].decode("utf-8")
            except KeyError:
                pass
        
        try:
            self.DATA_NAME  = h5_file[keys[-1]].attrs['LONG_NAME'][0].decode("utf-8")
        except KeyError:
            try:
                self.DATA_NAME = h5_file.attrs['LONG_NAME'][0].decode("utf-8")
            except KeyError:
                pass

        if type(h5_file.attrs['NAME'][0]) == bytes:
            self.DATA_PAR_NAME     = h5_file.attrs['NAME'][0].decode("utf-8")
        else:
            self.DATA_PAR_NAME     = h5_file.attrs['NAME'][0]
        
        
        # self.DATA_PAR_NAME  = h5_file.attrs['NAME'][0].decode("utf-8")
        self.TIME_UNITS     = h5_file.attrs['TIME UNITS'][0].decode("utf-8")
        
        try:
            self.DT             = h5_file.attrs['DT'][0]
        except KeyError:
            self.DT = h5_file['SIMULATION'].attrs['DT']
            
        try:
            self.OFFSET_T             = h5_file.attrs['OFFSET_T']
        except KeyError:
            self.OFFSET_T = 0 #h5_file['SIMULATION'].attrs['OFFSET_T']
        
        try:
            self.OFFSET_X             = h5_file.attrs['OFFSET_X']
        except KeyError:
            self.OFFSET_X = [0, 0, 0] #h5_file['SIMULATION'].attrs['OFFSET_X']
        
        self.TIME           = h5_file.attrs['TIME'] + self.OFFSET_T*self.DT 
        
        
        self.AXIS1       = np.array(h5_file[keys[0]]['AXIS1'])
        self.AXIS1_UNITS = h5_file[keys[0]]['AXIS1'].attrs['UNITS'][0].decode("utf-8")
        if type(h5_file[keys[0]]['AXIS1'].attrs['LONG_NAME'][0]) == bytes:
            self.AXIS1_NAME  = h5_file[keys[0]]['AXIS1'].attrs['LONG_NAME'][0].decode("utf-8")
        else:
            self.AXIS1_NAME  = h5_file[keys[0]]['AXIS1'].attrs['LONG_NAME'][0]
            
        self.AXIS1_NAME  = h5_file[keys[0]]['AXIS1'].attrs['LONG_NAME'][0].decode("utf-8")
        self.NX          = h5_file[keys[-1]].shape[self.DIM-1]
        self.DX          = (self.AXIS1[1]-self.AXIS1[0])/(self.NX)
        self.X           = self.AXIS1[0] + (self.OFFSET_X[0]+np.arange(0, self.NX, dtype=dtype))*self.DX
        self.X_MESH      = np.linspace(self.AXIS1[0], self.AXIS1[-1], self.NX+1, endpoint = True, dtype=dtype)

        if self.DIM > 1:
            self.AXIS2       = np.array(h5_file[keys[0]]['AXIS2'], dtype=dtype)
            self.AXIS2_UNITS = h5_file[keys[0]]['AXIS2'].attrs['UNITS'][0].decode("utf-8")
            if type(h5_file[keys[0]]['AXIS2'].attrs['LONG_NAME'][0]) == bytes:
                self.AXIS2_NAME  = h5_file[keys[0]]['AXIS2'].attrs['LONG_NAME'][0].decode("utf-8")
            else:
                self.AXIS2_NAME  = h5_file[keys[0]]['AXIS2'].attrs['LONG_NAME'][0]
            self.NY          = h5_file[keys[-1]].shape[self.DIM-1 - 1]
            self.DY          = (self.AXIS2[1]-self.AXIS2[0])/(self.NY)
            self.Y           = self.AXIS2[0] + (self.OFFSET_X[1]+np.arange(0, self.NY, dtype=dtype))*self.DY
            self.Y_MESH      = np.linspace(self.AXIS2[0], self.AXIS2[-1], self.NY+1, endpoint = True, dtype=dtype)
        
        if self.DIM == 3:
            self.AXIS3       = np.array(h5_file[keys[0]]['AXIS3'])
            self.AXIS3_UNITS = h5_file[keys[0]]['AXIS3'].attrs['UNITS'][0].decode("utf-8")
            if type(h5_file[keys[0]]['AXIS3'].attrs['LONG_NAME'][0]) == bytes:
                self.AXIS3_NAME  = h5_file[keys[0]]['AXIS3'].attrs['LONG_NAME'][0].decode("utf-8")
            else:
                self.AXIS3_NAME  = h5_file[keys[0]]['AXIS3'].attrs['LONG_NAME'][0]
            self.NZ          = h5_file[keys[-1]].shape[self.DIM-1 - 2]
            self.DZ          = (self.AXIS3[1]-self.AXIS3[0])/(self.NZ)
            self.Z           = self.AXIS3[0] + (self.OFFSET_X[2]+np.arange(0, self.NZ, dtype=dtype))*self.DZ
            self.Z_MESH      = np.linspace(self.AXIS3[0], self.AXIS3[-1], self.NZ+1, endpoint = True, dtype=dtype)

        if inner:
            self.NX = max1 - min1
            self.NY = max2 - min2
            
            self.AXIS1 = np.array([self.X[min1], self.X[max1]], dtype=dtype)
            self.AXIS2 = np.array([self.Y[min2], self.Y[max2]], dtype=dtype)
            
            self.X = self.X[min1:max1]
            self.Y = self.Y[min2:max2]
        
        if not no_data:
            if any(slice_arr):
                slice_axis = [i for i, b in enumerate(slice_arr) if b][0] 
                N_slice_axis = h5_file[keys[-1]].shape[self.DIM -1 - slice_axis]
                ext = list(h5_file['AXIS/AXIS'+str(slice_axis+1)])

                if slice_arr[slice_axis] == 'half':
                    self.SLICE_IDX = N_slice_axis//2
                    
                elif isinstance(slice_arr[slice_axis], int):
                    self.SLICE_IDX = slice_arr[slice_axis]

                elif ext[0] <= slice_arr[slice_axis] <= ext[1]:
                    dx = (ext[1]-ext[0])/(N_slice_axis)
                    self.SLICE_IDX = int(np.round((slice_arr[slice_axis] - ext[0])/dx))
                
                slc = [slice(None)] * self.DIM 
                slc[self.DIM -1 - slice_axis] = slice(self.SLICE_IDX, self.SLICE_IDX+1)
                self.DATA = np.squeeze(h5_file[keys[-1]][tuple(slc)])


            else:
                if inner:
                    # 1/2 area
                    factor = 1. - np.sqrt(2./np.pi), 1. + np.sqrt(2./np.pi)

                    min1, max1 = int(self.SHAPE[self.DIM-1]//2 * factor[0]) - 3, int(self.SHAPE[self.DIM-1]//2 * factor[1]) + 4
                    min2, max2 = int(self.SHAPE[self.DIM-2]//2 * factor[0]) - 3, int(self.SHAPE[self.DIM-2]//2 * factor[1]) + 4
                    
                    if h5_file[keys[-1]].dtype != dtype:
                        self.DATA = h5_file[keys[-1]][:zlen, min2:max2, min1:max1].astype(dtype)
                    else:
                        self.DATA = h5_file[keys[-1]][:zlen, min2:max2, min1:max1]
#                     self.DATA = np.array(h5_file[keys[-1]][:zlen, min2:max2, min1:max1], dtype="float32")
                else:
                    if h5_file[keys[-1]].dtype != dtype:
                        self.DATA = h5_file[keys[-1]][:zlen, ...].astype(dtype)
                    else:
                        self.DATA = h5_file[keys[-1]][:zlen, ...]
#                     self.DATA = np.array(h5_file[keys[-1]][:zlen, ...], dtype="float32")

        
            
#             if len(self.Z_MESH) > self.NZ+1:
#                 self.Z_MESH = self.Z_MESH[:-1]
                

class osiris_cylindrical_data_object(object):
    """
    Create Osiris Data object from h5py file object made with Quasi-3D/azimuthally symmetric
    cylindrical simulation parameters (unclear if other newer files have same type so leaving 3d)
        - so x1 = z, x2 = r
      
    Attributes:
    
        General DATA attributes
            DIM (int):              number dimensions of the data object
            DATA (arr):             data from the file, either a 1D, 2D or 3D array
            DATA_UNITS (str):       units for the data to use in latex math mode
            DATA_LABEL (str):       full label for the data to use in latex math mode
            DATA_NAME (str):        label for the data from input deck
            TIME (flt):             output time from the simulation
            TIME_UNITS (str):       units for the time to use in latex math mode
            DT (flt):               dt between timesteps in simulation
        
        AXIS atributes (repeated for each dimension - example with AXIS1)
            AXIS1 (flt):            array with the endpoints of the axis
            AXIS1_UNITS (str):      units for the axis1 data
            AXIS1_NAME (str):       full label for axis1 to use in latex math mode
            NX (int):               number of data points in axis1 direction
            DX (flt):               spacing between subsequent datapoints
            X_MESH (1D arr):        array of axis1 values, w/ length=NX+1 (from AXIS1[0] to AXIS1[1], with spacing DX)
            X (1D arr):             array of axis1 values, midpoint of grid (from AXIS1[0] + DX/2 to AXIS1[1] + DX/2)
            
    Methods:
        __init__(self, h5_file)
            constructor for osiris_cartesian_data_object class
            
            Parameters:
            h5_file (h5py file object): File with the data, already opened using h5_file = h5py.File(filename,'r')
    """

    def __init__(self, h5_file, slice_z):
        keys = [key for key in h5_file.keys()]
        if keys[-1]=='SIMULATION': keys.pop() #want the data name to be the -1 element
        self.DIM            = len(h5_file[keys[-1]].shape)
        
        if not slice_z:
            self.DATA = h5_file[keys[-1]][...]
#             self.DATA = np.array(h5_file[keys[-1]])

        else:
            N_slice_axis = h5_file[keys[-1]].shape[self.DIM -1]
            ext = list(h5_file['AXIS/AXIS1'])

            if slice_z == 'half':
                self.SLICE_IDX = N_slice_axis//2

            elif ext[0] <= slice_z <= ext[1]:
                dx = (ext[1]-ext[0])/(N_slice_axis)
                self.SLICE_IDX = int(np.round((slice_z - ext[0])/dx))

            slc = [slice(None)] * self.DIM 
            slc[self.DIM - 1] = slice(self.SLICE_IDX, self.SLICE_IDX+1)
            self.DATA = np.squeeze(h5_file[keys[-1]][tuple(slc)])
        
        # stuff breaks when I don't do this, IDK why
        if type(h5_file.attrs['UNITS'][0]) == bytes:
            self.DATA_UNITS     = h5_file.attrs['UNITS'][0].decode("utf-8")
        else:
            self.DATA_UNITS     = h5_file.attrs['UNITS'][0]
        if type(h5_file.attrs['LABEL'][0]) == bytes:
            self.DATA_LABEL     = h5_file.attrs['LABEL'][0].decode("utf-8")
        else:
            self.DATA_LABEL     = h5_file.attrs['LABEL'][0].decode("utf-8")
        if type(h5_file.attrs['NAME'][0]) == bytes:
            self.DATA_NAME     = h5_file.attrs['NAME'][0].decode("utf-8")
        else:
            self.DATA_NAME     = h5_file.attrs['NAME'][0].decode("utf-8")

        if type(h5_file.attrs['TIME UNITS'][0]) == bytes:
            self.TIME_UNITS     = h5_file.attrs['TIME UNITS'][0].decode("utf-8")
        else:
            self.TIME_UNITS     = h5_file.attrs['TIME UNITS'][0].decode("utf-8")
        
        try:
            self.DT             = h5_file.attrs['DT'][0]
        except KeyError:
            self.DT = h5_file['SIMULATION'].attrs['DT']
            
        try:
            self.OFFSET_T             = h5_file.attrs['OFFSET_T']
        except KeyError:
            self.OFFSET_T = 0 #h5_file['SIMULATION'].attrs['OFFSET_T']
        
        try:
            self.OFFSET_X             = h5_file.attrs['OFFSET_X']
        except KeyError:
            self.OFFSET_X = [0, 0, 0] #h5_file['SIMULATION'].attrs['OFFSET_X']

        self.TIME           = h5_file.attrs['TIME'] + self.OFFSET_T*self.DT 
                
        self.AXIS1       = np.array(h5_file[keys[0]]['AXIS1'])
        self.AXIS1_UNITS = h5_file[keys[0]]['AXIS1'].attrs['UNITS'][0].decode("utf-8")
        self.AXIS1_NAME  = h5_file[keys[0]]['AXIS1'].attrs['LONG_NAME'][0].decode("utf-8")
        self.NX          = h5_file[keys[-1]].shape[self.DIM-1]
        self.DX          = (self.AXIS1[1]-self.AXIS1[0])/(self.NX)
        self.X           = self.AXIS1[0] + (self.OFFSET_X[0]+np.arange(0,self.NX))*self.DX
        self.X_MESH      = np.arange(self.AXIS1[0], self.AXIS1[-1]+self.DX, self.DX)
        if len(self.X_MESH) > self.NX+1:
            self.X_MESH = self.X_MESH[:-1]
            
        # slice z...
            

        if self.DIM > 1:
            self.AXIS2       = np.array(h5_file[keys[0]]['AXIS2'])
            self.AXIS2_UNITS = h5_file[keys[0]]['AXIS2'].attrs['UNITS'][0].decode("utf-8")
            self.AXIS2_NAME  = h5_file[keys[0]]['AXIS2'].attrs['LONG_NAME'][0].decode("utf-8")
            self.NY          = h5_file[keys[-1]].shape[self.DIM-1 - 1]
            self.DY          = (self.AXIS2[1]-self.AXIS2[0])/(self.NY)
            self.Y           = self.AXIS2[0] + (self.OFFSET_X[1]+np.arange(0,self.NY))*self.DY
            self.Y_MESH      = np.arange(self.AXIS2[0], self.AXIS2[-1]+self.DY, self.DY)
            if len(self.Y_MESH) > self.NY+1:
                self.Y_MESH = self.Y_MESH[:-1]

        if self.DIM == 3:
            self.AXIS3       = np.array(h5_file[keys[0]]['AXIS3'])
            self.AXIS3_UNITS = h5_file[keys[0]]['AXIS3'].attrs['UNITS'][0].decode("utf-8")
            self.AXIS3_NAME  = h5_file[keys[0]]['AXIS3'].attrs['LONG_NAME'][0].decode("utf-8")
            self.NZ          = h5_file[keys[-1]].shape[self.DIM-1 - 2]
            self.DZ          = (self.AXIS3[1]-self.AXIS3[0])/(self.NZ)
            self.Z           = self.AXIS3[0] + (self.OFFSET_X[2]+np.arange(0,self.NZ))*self.DZ
            self.Z_MESH      = np.arange(self.AXIS3[0], self.AXIS3[-1]+self.DZ, self.DZ)
            if len(self.Z_MESH) > self.NZ+1:
                self.Z_MESH = self.Z_MESH[:-1]

    # Open first hdf5 file, save its contents to vys_data object, and close file
def get_cartesian_data(filename, slice_x = False, slice_y = False, slice_z = False, inner = False, zlen = None, dtype = 'f4'):
    file = h5py.File(filename, 'r')
    data_object = osiris_cartesian_data_object(file, [slice_x, slice_y, slice_z], inner = inner, zlen = zlen, dtype = dtype)
    file.close()
    return(data_object)

def get_cartesian_everything_but_data(filename, inner = False, zlen = None):
    file = h5py.File(filename, 'r')
    axis_object = osiris_cartesian_data_object(file, [False, False, False], inner = inner, zlen = zlen, no_data = True)
    file.close()
    return(axis_object)

def get_cylindrical_data(filename, slice_z = False):
    file = h5py.File(filename, 'r')
    data_object = osiris_cylindrical_data_object(file, slice_z)
    file.close()
    return(data_object)



