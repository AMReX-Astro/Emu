
'''
EMU algorithm to compute the shape function, translated to Python using AI (ChatGPT)
Note: Be careful, this portion of the code might not be correct
'''

class ParticleInterpolator:
    def __init__(self, max_spline_order, delta, order):
        """
        Initialize the ParticleInterpolator with given delta and order.
        The maximum spline order is defined at the creation of the object.
        """
        self.max_spline_order = max_spline_order
        self.shape_functions = [0.0] * (max_spline_order + 1)

        if max_spline_order >= 2 and order > 1:
            self.compute_order_2(delta)
        elif max_spline_order >= 1 and order > 0:
            self.compute_order_1(delta)
        else:
            self.compute_order_0(delta)

    def __call__(self, i):
        """
        Access shape functions with bounds checking.
        """
        assert self.first() <= i <= self.last(), "Index out of bounds!"
        return self.shape_functions[i - self.first()]

    def first(self):
        """
        Get the first cell index.
        """
        return self.first_cell

    def last(self):
        """
        Get the last cell index.
        """
        return self.last_cell

    def compute_order_0(self, delta):
        """
        Perform order-0 interpolation using the nearest cell center.
        """
        self.first_cell = self.nearest_cell_center_index(delta)
        self.last_cell = self.first_cell
        self.shape_functions[0] = 1.0

    def compute_order_1(self, delta):
        """
        Perform order-1 spline interpolation.
        """
        nearest = self.nearest_cell_center_index(delta)
        offset = self.nearest_cell_center_offset(delta)

        if offset >= 0:
            self.first_cell = nearest
            self.shape_functions[0] = 1.0 - offset
            self.shape_functions[1] = offset
        else:
            self.first_cell = nearest - 1
            self.shape_functions[0] = -offset
            self.shape_functions[1] = 1.0 + offset

        self.last_cell = self.first_cell + 1

    def compute_order_2(self, delta):
        """
        Perform order-2 spline interpolation.
        """
        self.first_cell = self.nearest_cell_center_index(delta) - 1
        self.last_cell = self.first_cell + 2

        offset = self.nearest_cell_center_offset(delta)

        self.shape_functions[0] = 0.5 * (0.5 - offset) * (0.5 - offset)
        self.shape_functions[1] = 0.75 - offset * offset
        self.shape_functions[2] = 0.5 * (0.5 + offset) * (0.5 + offset)

    def nearest_cell_center_index(self, delta):
        """
        Compute the index of the nearest cell center.
        """
        dstar = delta - 1.0 if delta - 0.5 < 0.0 else delta
        return int(dstar)
    
    def nearest_cell_center_offset(self, delta):
        """
        Compute the offset from the nearest cell center.
        """
        fraction = delta - int(delta)
        offset = fraction + 0.5 if delta <= 0 else fraction - 0.5
        return offset
