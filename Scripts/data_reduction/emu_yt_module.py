import yt
import numpy as np
import math
import scipy.fft as fft
from yt import derived_field
import yt.units.dimensions as dimensions

class Dim3(object):
    def __init__(self, x=0, y=0, z=0):
        if type(x) == np.ndarray:
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
        else:
            self.x = x
            self.y = y
            self.z = z

    @staticmethod
    def dimensions(lo, hi):
        # return the number of elements (lo, hi)
        # contain along each dimension
        return [hi.x - lo.x + 1,
                hi.y - lo.y + 1,
                hi.z - lo.z + 1]

class FourierData(object):
    def __init__(self, time, kx, ky, kz, FT_magnitude, FT_phase):
        # Dataset time
        self.time = time

        # Wavenumbers
        self.kx = kx
        self.ky = ky
        self.kz = kz

        # Fourier transform magnitude & phase
        self.magnitude = FT_magnitude
        self.phase = FT_phase

class EmuDataset(object):
    def __init__(self, plotfile=None):
        # initialize from a supplied plotfile, otherwise just make an empty dataset object
        # and initialize it later with init_from_data()
        if plotfile:
            self.setup_dataset(yt.load(plotfile))
        else:
            self.ds = None

    def init_from_data(self, data, left_edge=None, right_edge=None, sim_time=0.0, dimensions=None,
                       length_unit=(1.0, "cm"), periodicity=(True, True, True), nprocs=1):

        assert(left_edge is not None and right_edge is not None)

        # initialize the dataset using a dictionary of numpy arrays in data, keyed by field name
        domain_bounds = np.array([[left_edge[0], right_edge[0]],
                                  [left_edge[1], right_edge[1]],
                                  [left_edge[2], right_edge[2]]])

        self.setup_dataset(yt.load_uniform_grid(data, dimensions, length_unit=length_unit, sim_time=sim_time,
                                                bbox=domain_bounds, periodicity=periodicity, nprocs=nprocs))

    def setup_dataset(self, yt_dataset):
        self.ds = yt_dataset
        self.construct_covering_grid()
        #self.add_emu_fields()

    def construct_covering_grid(self):
        self.cg = self.ds.covering_grid(level=0, left_edge=self.ds.domain_left_edge,
                                        dims=self.ds.domain_dimensions)

        # number of cells in each dimension
        self.Nx = self.ds.domain_dimensions[0]
        self.Ny = self.ds.domain_dimensions[1]
        self.Nz = self.ds.domain_dimensions[2]

        # find dx, dy, dz in each of X,Y,Z
        # this is the spacing between cell centers in the domain
        # it is the same as the spacing between cell edges
        self.dx = (self.ds.domain_right_edge[0] - self.ds.domain_left_edge[0])/self.ds.domain_dimensions[0]
        self.dy = (self.ds.domain_right_edge[1] - self.ds.domain_left_edge[1])/self.ds.domain_dimensions[1]
        self.dz = (self.ds.domain_right_edge[2] - self.ds.domain_left_edge[2])/self.ds.domain_dimensions[2]

        if self.Nx > 1:
            # low, high edge locations in x domain
            xlo = self.ds.domain_left_edge[0]
            xhi = self.ds.domain_right_edge[0]

            # the offset between the edges xlo, xhi and the interior cell centers
            x_cc_offset = 0.5 * self.dx

            self.X, DX = np.linspace(xlo + x_cc_offset, # first cell centered location in the interior of x domain
                                xhi - x_cc_offset, # last cell centered location in the interior of x domain
                                num=self.Nx,            # Nx evenly spaced samples
                                endpoint=True,     # include interval endpoint for the last cell-centered location in the domain
                                retstep=True)      # return the stepsize between cell centers to check consistency with dx

            # the spacing we calculated should be the same as what linspace finds between cell centers
            # using our edge-to-cell-center offset and the number of samples
            #print("dx, DX = ", dx, DX)
            assert math.isclose(self.dx,DX)

        if self.Ny > 1:
            # low, high edge locations in y domain
            ylo = self.ds.domain_left_edge[1]
            yhi = self.ds.domain_right_edge[1]

            # the offset between the edges ylo, yhi and the interior cell centers
            y_cc_offset = 0.5 * self.dy

            self.Y, DY = np.linspace(ylo + y_cc_offset, # first cell centered location in the interior of y domain
                                yhi - y_cc_offset, # last cell centered location in the interior of y domain
                                num=self.Ny,            # Ny evenly spaced samples
                                endpoint=True,     # include interval endpoint for the last cell-centered location in the domain
                                retstep=True)      # return the stepsize between cell centers to check consistency with dy

            # the spacing we calculated should be the same as what linspace finds between cell centers
            # using our edge-to-cell-center offset and the number of samples
            #print("dy, DY = ", dy, DY)
            assert math.isclose(self.dy,DY)


        if self.Nz > 1:
            # low, high edge locations in z domain
            zlo = self.ds.domain_left_edge[2]
            zhi = self.ds.domain_right_edge[2]

            # the offset between the edges zlo, zhi and the interior cell centers
            z_cc_offset = 0.5 * self.dz

            self.Z, DZ = np.linspace(zlo + z_cc_offset, # first cell centered location in the interior of z domain
                                zhi - z_cc_offset, # last cell centered location in the interior of z domain
                                num=self.Nz,            # Nz evenly spaced samples
                                endpoint=True,     # include interval endpoint for the last cell-centered location in the domain
                                retstep=True)      # return the stepsize between cell centers to check consistency with dz

            # the spacing we calculated should be the same as what linspace finds between cell centers
            # using our edge-to-cell-center offset and the number of samples
            #print("dz, DZ = ", dz, DZ)
            assert math.isclose(self.dz,DZ)

    def get_num_flavors(self):
        just_the_fields = [f for ftype, f in self.ds.field_list]
        if "N33_Re" in just_the_fields:
            raise NotImplementedError("Analysis script currently only supports 2 and 3 flavor simulations")
        elif "N22_Re" in just_the_fields:
            return 3
        else:
            return 2

    def get_num_dimensions(self):
        dim = self.ds.domain_dimensions
        return np.sum(dim > 1)

    #def add_emu_fields(self):
    #    # first, define the trace
    #    def _make_trace(ds):
    #        if self.get_num_flavors() == 3:
    #            def _trace(field, data):
    #                return data["N00_Re"] + data["N11_Re"] + data["N22_Re"]
    #            return _trace
    #        else:
    #            def _trace(field, data):
    #                return data["N00_Re"] + data["N11_Re"]
    #            return _trace
    #
    #    _trace = _make_trace(self.ds)
    #
    #    self.ds.add_field(("gas", "trace"), function=_trace, sampling_type="local", units="auto", dimensions=dimensions.dimensionless)
    #
    #    # now, define normalized fields
    #    for f in self.ds.field_list:
    #        if "_Re" in f[1] or "_Im" in f[1]:
    #            fname = f[1]
    #            fname_norm = "{}_norm_tr".format(fname)
    #
    #            def _make_derived_field(f):
    #                def _derived_field(field, data):
    #                    return data[f]/data[("gas", "trace")]
    #                return _derived_field
    #
    #            _norm_derived_f = _make_derived_field(f)
    #            self.ds.add_field(("gas", fname_norm), function=_norm_derived_f, sampling_type="local", units="auto", dimensions=dimensions.dimensionless)

    def fourier(self, field_Re, field_Im=None, nproc=None):
        if field_Im:
            FT = np.squeeze(self.cg[field_Re][:,:,:].d + 1j * self.cg[field_Im][:,:,:].d)
        else:
            FT = np.squeeze(self.cg[field_Re][:,:,:].d)

        # use fftn to do an N-dimensional FFT on an N-dimensional numpy array
        FT = fft.fftn(FT,workers=nproc)

        # we're shifting the sampling frequencies next, so we have to shift the FFT values
        FT = fft.fftshift(FT)

        # get the absolute value of the fft
        FT_mag = np.abs(FT)

        # get the phase of the fft
        FT_phi = np.angle(FT)

        if self.Nx > 1:
            # find the sampling frequencies in X & shift them
            kx = fft.fftfreq(self.Nx, self.dx)
            kx = fft.fftshift(kx)
        else:
            kx = None

        if self.Ny > 1:
            # find the sampling frequencies in Y & shift them
            ky = fft.fftfreq(self.Ny, self.dy)
            ky = fft.fftshift(ky)
        else:
            ky = None

        if self.Nz > 1:
            # find the sampling frequencies in Z & shift them
            kz = fft.fftfreq(self.Nz, self.dz)
            kz = fft.fftshift(kz)
        else:
            kz = None

        return FourierData(self.ds.current_time, kx, ky, kz, FT_mag, FT_phi)

    def get_rectangle(self, left_edge, right_edge):
        # returns an EmuDataset containing only the rectangular region
        # defined by [left_edge, right_edge] in the current dataset

        data, data_dimensions = self.get_data(bounds=(left_edge, right_edge))

        # return a new EmuDataset object
        new_dataset = EmuDataset()
        new_dataset.init_from_data(data, left_edge=left_edge, right_edge=right_edge, sim_time=self.ds.current_time.in_units("s"),
                                   dimensions=data_dimensions, length_unit=self.ds.length_unit,
                                   periodicity=(False, False, False), nprocs=1)
        return new_dataset

    def get_data(self, bounds=None):
        # gets a dictionary of numpy arrays with the raw data in this dataset, keyed by field
        # if bounds = None, returns the entire domain, otherwise interpret bounds = (left_edge, right_edge)
        # where left_edge and right_edge are each numpy arrays with the physical positions of the selection edges
        # and return the subdomain inside those bounds at the same resolution as the original dataset
        cg = self.ds.covering_grid(left_edge=self.ds.domain_left_edge, dims=self.ds.domain_dimensions, level=0)

        ddim = self.ds.domain_dimensions

        # lo, hi Dim3's are defined in the AMReX style where they are inclusive indices
        # defined for Fortran-style array slicing (not Python)
        lo = Dim3(0,0,0)
        hi = Dim3(ddim - 1)

        if bounds:
            left_edge, right_edge = bounds
            dleft = self.ds.domain_left_edge
            dright = self.ds.domain_right_edge
            delta = [self.dx, self.dy, self.dz]

            # initialize lo to the first indices in the domain
            lo = np.array([0,0,0])
            # initialize hi to the last indices in the domain
            hi = ddim - 1

            for i in range(3):
                if ddim[i] > 1:
                    # set lo[i] to the first cell-centered index to the right of left_edge
                    lo[i] = round((left_edge[i].d - dleft[i].d) / delta[i].d)
                    # set hi[i] to the last cell-centered index to the left of right_edge
                    hi[i] = hi[i] - round((dright[i].d - right_edge[i].d) / delta[i].d)

            lo = Dim3(lo)
            hi = Dim3(hi)

        data = {}
        data_dimensions = Dim3.dimensions(lo, hi)

        # Note: we have to throw away the field type, e.g. 'boxlib' because YT's uniform data loader
        # gives its fields a 'stream' type. If we were to keep the field type here, we would be unable
        # to call to_3D() on a dataset returned by to_2D() since the field types would not match.
        for ftype, f in self.ds.field_list:
            data[f] = (cg[f][lo.x:hi.x+1, lo.y:hi.y+1, lo.z:hi.z+1].d, "")

        return data, data_dimensions

    def to_2D(self, extend_dims=None):
        # transform this 1D dataset into a 2D EmuDataset object

        # first, assert this dataset is 1D along z
        assert(self.ds.domain_dimensions[0] == 1 and self.ds.domain_dimensions[1] == 1 and self.ds.domain_dimensions[2] > 1)

        # get the 1D data
        data, _ = self.get_data()

        data_2D = {}

        # by default extend y to match z unless extend_dims is passed
        length_z = self.ds.domain_dimensions[2]
        length_y = length_z
        if extend_dims:
            length_y = extend_dims

        for f in data.keys():
            df, df_units = data[f]
            data_2D[f] = (np.tile(df, (1, length_y, 1)), df_units)

        left_edge = self.ds.domain_left_edge
        left_edge[1] = left_edge[2]
        right_edge = self.ds.domain_right_edge
        right_edge[1] = right_edge[2]
        dimensions = self.ds.domain_dimensions
        dimensions[1] = length_y

        # return a new EmuDataset object
        new_dataset = EmuDataset()
        new_dataset.init_from_data(data_2D, left_edge=left_edge, right_edge=right_edge, sim_time=self.ds.current_time.in_units("s"),
                                   dimensions=dimensions, length_unit=self.ds.length_unit,
                                   periodicity=self.ds.periodicity, nprocs=1)
        return new_dataset

    def to_3D(self, extend_dims=None):
        # transform this 1D or 2D dataset into a 3D EmuDataset object

        # check if this dataset is 1D or 2D and extended along z
        assert(self.ds.domain_dimensions[0] == 1 and self.ds.domain_dimensions[2] > 1)

        # get the current data
        data, _ = self.get_data()

        data_3D = {}
        left_edge = None
        right_edge = None
        dimensions = None

        if self.ds.domain_dimensions[1] == 1:
            # we are 1D, extended along z
            # by default extend x, y to match z unless extend_dims is passed
            length_z = self.ds.domain_dimensions[2]
            length_y = length_z
            length_x = length_z
            if extend_dims:
                length_x, length_y = extend_dims

            for f in data.keys():
                df, df_units = data[f]
                data_3D[f] = (np.tile(df, (length_x, length_y, 1)), df_units)

            left_edge = self.ds.domain_left_edge
            left_edge[0] = left_edge[2]
            left_edge[1] = left_edge[2]
            right_edge = self.ds.domain_right_edge
            right_edge[0] = right_edge[2]
            right_edge[1] = right_edge[2]
            dimensions = self.ds.domain_dimensions
            dimensions[0] = length_x
            dimensions[1] = length_y

        else:
            # we are 2D, extended along y and z
            # by default extend x to match y unless extend_dims is passed
            length_y = self.ds.domain_dimensions[1]
            length_x = length_y
            if extend_dims:
                length_x = extend_dims

            for f in data.keys():
                df, df_units = data[f]
                data_3D[f] = (np.tile(df, (length_x, 1, 1)), df_units)

            left_edge = self.ds.domain_left_edge
            left_edge[0] = left_edge[1]
            right_edge = self.ds.domain_right_edge
            right_edge[0] = right_edge[1]
            dimensions = self.ds.domain_dimensions
            dimensions[0] = length_x

        # return a new EmuDataset object
        new_dataset = EmuDataset()
        new_dataset.init_from_data(data_3D, left_edge=left_edge, right_edge=right_edge, sim_time=self.ds.current_time.in_units("s"),
                                   dimensions=dimensions, length_unit=self.ds.length_unit,
                                   periodicity=self.ds.periodicity, nprocs=1)
        return new_dataset

