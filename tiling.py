import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
import matplotlib.pyplot as plt

from utilities import equi_to_cart, cart_to_equi
import parameters as params

def _rotate(rotation_angle):
    """
    Defines the matrix to rotate a point about the origin (RA=0, DEC=0) counter-clockwise
    in degrees.
    """
    theta = np.radians(rotation_angle)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return Rx


def _move(ra, dec):
    """
    Defines the matrix to move the center point to the desired equatorial coordinates.
    """
    alpha = np.radians(ra)
    delta = np.radians(dec)
    
    # Move to DEC coordinate
    Ry = np.array([
        [np.cos(delta), 0, -np.sin(delta)],
        [0, 1, 0],
        [np.sin(delta), 0, np.cos(delta)]
    ])
    # Move to RA coordinate
    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry


def _FPA_coords(l, w, g):
    """
    Defines the corner points of each detector in a 2x2 focal-plane array (FPA), 
    centered at RA=0 and DEC=0.
    
    The function returns a list of 4 sublists, each corresponding to a detector. 
    Each sublist contains 4 NumPy arrays representing the Cartesian coordinates 
    of the detector's corners, ordered as follows: lower-right, upper-right, 
    upper-left, and lower-left.
    """
    ra_coords = [
        g / 2 + l, g / 2 + l, g / 2, g / 2,
        g / 2 + l, g / 2 + l, g / 2, g / 2,
        -g / 2, -g / 2, -(g / 2 + l), -(g / 2 + l),
        -g / 2, -g / 2, -(g / 2 + l), -(g / 2 + l)
    ]

    dec_coords = [
        -(g / 2 + w), -g / 2, -g / 2, -(g / 2 + w),
        g / 2, g / 2 + w, g / 2 + w, g / 2,
        g / 2, g / 2 + w, g / 2 + w, g / 2,
        -(g / 2 + w), -g / 2, -g / 2, -(g / 2 + w)
    ]

    FPA = [[] for _ in range(4)]
    for i in range(16):
        FPA[i // 4].append(equi_to_cart((ra_coords[i], dec_coords[i])))
    return FPA


def create_tile(ra_center, dec_center, rotation_angle, l, w, g, footprint=None):
    """
    Creates a `tile`, which is a MultiPolygon object representing a 2x2 focal-plane array (FPA).
    
    If the `footprint` is provided, the function also returns the `intersection_ratio`, 
    indicating the overlap between the `tile` and the `footprint`.
    
    Parameters
    ----------
    ra_center : float
        Right Ascension of the tile center in degrees.
    
    dec_center : float
        Declination of the tile center in degrees.
    
    rotation_angle : float
        Tile rotation angle in degrees (counter-clockwise).
    
    l : float
        Length of each detector in degrees.
    
    w : float
        Width of each detector in degrees.
    
    g : float
        Gap between detectors in degrees.
    
    footprint : shapely.geometry.Polygon, optional
        The survey footprint polygon. If not provided, the `intersection_ratio` is not calculated.
    
    Returns
    -------
    tile : shapely.geometry.MultiPolygon
        A MultiPolygon object representing the 2x2 focal-plane array.
    
    intersection_ratio : float, optional
        The ratio indicating how much of the tile's area overlaps with the footprint. 
        Returned only if `footprint` is provided.
    """
    detectors = []
    intersect_area = 0

    # Define the coordinates of the focal-plane array centered at the origin (RA=0, DEC=0)
    FPA = _FPA_coords(l, w, g)

    # Define the rotation and movement matrices
    rotation_matrix = _rotate(rotation_angle)
    movement_matrix = _move(ra_center, dec_center)

    # For each detector in the focal-plane array
    for detector in FPA:
        # stores the equitorial vertices of the detector
        ras = [] 
        decs = []
        # For the four corners of the detector
        for corner_vec in detector:
            # Rotate the corner about the origin
            rotated_vec = rotation_matrix @ np.array(corner_vec)
            # Move the corner to the correct RA and Dec
            moved_vec = movement_matrix @ rotated_vec
            # Convert to equatorial coordinates
            ra, dec = cart_to_equi(moved_vec)
            ras.append(ra)
            decs.append(dec)
        # Repeat the first vertex to close the polygon
        ras.append(ras[0])
        decs.append(decs[0])
        # Create a detector Polygon object
        vertices = [(x, y) for x, y in zip(ras, decs)]
        polygon = Polygon(vertices)

        if footprint:
            # Find the detector's intersection with the footprint
            detector_with_footprint = polygon.intersection(footprint)
            intersect_area += detector_with_footprint.area
        detectors.append(polygon)

    # A tile is a MultiPolygon object consisting of the detector polygons
    tile = MultiPolygon(detectors)
    if footprint:
        tile_area = tile.area
        intersection_ratio = intersect_area / tile_area
        return tile, intersection_ratio
    else:
        return tile


def create_course_tile(ra_center, dec_center, rotation_angle, l, w, g):
    """
    Create a coarse tile Polygon object (which does not account for gaps in the detector array).
    The course version of create_tile(). 
    """
    # Define the vertices of a tile centered at the origin
    tile_vertex = [
        equi_to_cart((x * (g / 2 + l), y * (g / 2 + w)))
        for x, y in [(1, -1), (1, 1), (-1, 1), (-1, -1)]
    ]
    # Define the rotation and movement matrices
    rotation_matrix = _rotate(rotation_angle)
    movement_matrix = _move(ra_center, dec_center)

    # Store the equatorial vertices of the FPA
    RAs = []
    DECs = []

    for corner_vec in tile_vertex:
        # Rotate the corner about the origin
        rotated_vec = rotation_matrix @ np.array(corner_vec)
        # Move the corner to the correct RA and Dec
        moved_vec = movement_matrix @ rotated_vec
        # Convert to equatorial coordinates
        ra, dec = cart_to_equi(moved_vec)
        RAs.append(ra)
        DECs.append(dec)

    # Repeat the first vertex to close the polygon
    RAs.append(RAs[0])
    DECs.append(DECs[0])

    # Create a tile Polygon object
    vertices = [(x, y) for x, y in zip(RAs, DECs)]
    polygon = Polygon(vertices)

    return polygon


def _rotate2d(x_array, y_array, rotation_angle):
    """
    Performs a 2D rotation (treating as a Cartesian plane) on a footprint boundary coordinates set.
    """
    x_center = np.mean(x_array)
    y_center = np.mean(y_array)
    x_translated = x_array - x_center
    y_translated = y_array - y_center

    # Define the 2D rotation matrix
    theta = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Apply rotation
    rotated = rotation_matrix @ np.vstack((x_translated, y_translated))
    x_rotated = rotated[0, :] + x_center
    y_rotated = rotated[1, :] + y_center
    return x_rotated, y_rotated


def _rotate_move_grid(ra_grid_flat, dec_grid_flat, rotation_angle, ra_center, dec_center):
    """
    Rotates and moves a grid of tile center coordinates.
    """
    new_ra_grid_flat = []
    new_dec_grid_flat = []

    # Iterate over each RA and DEC in the grid
    for RA, DEC in zip(ra_grid_flat, dec_grid_flat):
        # Convert equatorial coordinates to Cartesian coordinates
        vector = equi_to_cart((RA, DEC))
        # Apply rotation about the origin
        vector = np.matmul(_rotate(rotation_angle), vector)
        # Translate the vector to the specified RA and DEC center
        vector = np.matmul(_move(ra_center, dec_center), vector)
        # Convert back to equatorial coordinates
        newRA, newDEC = cart_to_equi(vector)
        # Append the transformed coordinates
        new_ra_grid_flat.append(newRA)
        new_dec_grid_flat.append(newDEC)

    return np.array(new_ra_grid_flat), np.array(new_dec_grid_flat)


def snake_scan_tiling(boundary_coords, rotation_angle, min_intersect_ratio, tile_overlap, l, w, g):
    """
    Generates a list of tile center coordinates and tile MultiPolygon objects 
    in the order of the vertical snake scan tiling sequence.

    Parameters
    ----------
    boundary_coords : numpy.ndarray
        An array of footprint boundary coordinates as (ra, dec) in degrees.
        Note: RA values are within the [-180, 180] range.

    rotation_angle : float
        Tile rotation angle in degrees counter-clockwise
    
    min_intersect_ratio : float
        Minimum required intersection ratio between a tile and the footprint to be included.
        Must be between 0 and 1.
    
    tile_overlap : float
        Overlapping length between adjacent tiles in degrees.
    
    l : float
        Length of each detector in degrees.
    
    w : float
        Width of each detector in degrees.
    
    g : float
        Gap between detectors in degrees.

    Returns
    -------
    tile_centers : list of tuples
        List of (RA, DEC) tile center coordinates in degrees, ordered in a vertical snake scan 
        sequence.
    
    tile_shapes : list of shapely.geometry.MultiPolygon
        The corresponding MultiPolygon objects for each tile.
    """
    # Define footprint
    footprint = Polygon(boundary_coords)
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
        if not footprint.is_valid:
            raise ValueError("Provided `boundary_coords` do not form a valid polygon.")

    # Create a Cartesian grid boundary
    xmin, ymin, xmax, ymax = footprint.bounds

    # Adjust grid boundary to account for projection warp near polar coordinates
    if ymin < 30:
        ymin -= 5
    if ymax > 30:
        ymax += 5

    # Define the corner points of the bounding box
    xbound = np.array([xmin, xmax, xmax, xmin])
    ybound = np.array([ymin, ymin, ymax, ymax])

    # Rotate the boundary
    xbound_rotated, ybound_rotated = _rotate2d(xbound, ybound, rotation_angle)

    # Updated grid boundary (now treated as equatorial coordinates)
    ra_min, ra_max = np.min(xbound_rotated), np.max(xbound_rotated)
    dec_min, dec_max = np.min(ybound_rotated), np.max(ybound_rotated)

    # Avoid computing tiles with RA at the [-180, 180] boundary
    ra_min = max(ra_min, -179.9)
    ra_max = min(ra_max, 179.9)

    # Calculate the slew distance (in degrees) between adjacent tiles (not rotated)
    slew_length = (2 * l + g) - tile_overlap
    slew_width = (2 * w + g) - tile_overlap

    # Define a grid of tile centers, centered at the origin
    ra_center_grid = (ra_min + ra_max) / 2
    dec_center_grid = (dec_min + dec_max) / 2
    ra_vals = np.arange(ra_min + l, ra_max + l, step=slew_length) - ra_center_grid
    dec_vals = np.arange(dec_min + w, dec_max + w, step=slew_width) - dec_center_grid
    dec_grid, ra_grid = np.meshgrid(dec_vals, ra_vals)

    # Vertical snake tiling: alternate between tiling upwards and downwards for each RA value
    for i in range(dec_grid.shape[0]):
        if i % 2 == 1:
            dec_grid[i] = dec_grid[i][::-1]

    # Flatten the grid into a vertical snake tiling order, still centered at the origin
    ra_grid_flat = ra_grid.flatten()
    dec_grid_flat = dec_grid.flatten()

    # Apply rotation and move the grid center to the desired coordinates
    ra_list, dec_list = _rotate_move_grid(
        ra_grid_flat, dec_grid_flat, rotation_angle, ra_center_grid, dec_center_grid
    )

    # Select tiles with sufficient intersection with the footprint
    tile_centers = []
    tile_shapes = []
    for ra, dec in zip(ra_list, dec_list):
        tile, intersection_ratio = create_tile(ra, dec, rotation_angle, l, w, g, footprint)
        if intersection_ratio >= min_intersect_ratio:
            tile_centers.append((ra, dec))
            tile_shapes.append(tile)

    return tile_centers, tile_shapes


def cut_footprint(boundary_coords, dec_cutoff, cut_below=True):
    """
    Trims a footprint to exclude regions above or below a specified declination cutoff.

    Parameters
    ----------
    boundary_coords : np.ndarray
        An array of footprint boundary coordinates as (ra, dec) in degrees.
        
    dec_cutoff : float
        The declination cutoff in degrees.
        
    cut_below : bool, optional, default=True
        If True, removes the portion of the footprint below the `dec_cutoff`. 
        If False, removes the portion above `dec_cutoff`.
        
    Returns
    -------
    np.ndarray
        A 2D array of coordinates defining the boundary of the trimmed footprint. 
    """
    if dec_cutoff is None:
        return boundary_coords
    else:
        footprint = Polygon(boundary_coords)
        xmin, ymin, xmax, ymax = footprint.bounds
        if cut_below is True:
            limit_box = box(xmin, dec_cutoff, xmax, 90)
        else:
            limit_box = box(xmin, -90, xmax, dec_cutoff)
        trimmed_footprint = footprint.intersection(limit_box)
        return np.array(trimmed_footprint.exterior.coords)


def bin_tiles(bin_size, tile_centers):
    """
    Bins the `tile_centers` based on their Right Ascension (RA) coordinates and sorts 
    them within each bin by Declination (DEC) in descending order.

    Parameters
    ----------
    bin_size : float
        The size of each bin in degrees for grouping the tile centers based on their RA values.
        
    tile_centers : list of tuples
        A list of tuples where each tuple contains the RA and DEC coordinates (in degrees) of a tile
        center.
        
    Returns
    -------
    bins : list of lists
        A list of bins where each bin is a list of tuples representing tile centers (RA, DEC).
        The bins are ordered by ascending RA, and within each bin, the tiles are sorted by 
        descending DEC.
    """
    # Determine number of bins
    tile_RAs = [coord[0] for coord in tile_centers]
    min_RA, max_RA = min(tile_RAs), max(tile_RAs)
    num_bins = int(np.ceil((max_RA - min_RA) / bin_size))

    # Create a list of empty bins
    bins = [[] for _ in range(num_bins + 1)]

    # Determine which RA bin the coordinate belongs to
    for coord in tile_centers:
        RA, DEC = coord
        # use floor division to assign the bin index
        bin_index = int((RA - min_RA) // bin_size) 
        bins[bin_index].append((RA, DEC))

    # Delete the last bin if it is empty
    if len(bins[-1]) == 0:
        del bins[-1]

    # Sort coordinates in each bin from largest to smallest DEC values
    for i in range(len(bins)):
        bins[i] = sorted(bins[i], key=lambda coord: -coord[1])
    return bins
    

class Tiling():
    def __init__(self, boundary_coords, rotation_angle, tile_sequence, tile_shapes, tile_bins=None):
        """
        Create a Tiling instance.
        All parameters are stored as attributes.
        
        Parameters
        ----------
        boundary_coords : np.ndarray
            An array of footprint boundary coordinates as (ra, dec) in degrees.

        rotation_angle : float
            Tile rotation angle in degrees counter-clockwise.
            
        tile_sequence : list of tuples or list of lists
            A list of (RA, DEC) tile center coordinates in degrees, ordered in the snake scan tiling
            sequence.

        tile_shapes : list of shapely.geometry.MultiPolygon; optional
            List of MultiPolygon objects for each tile ordered in the snake scan tiling sequence.
            Note that surveys with large number of tiles will take longer to plot with these tiles. 

        tile_bins : list of lists, optional
            Only applicable to WideSurvey.
            A list of bins where each bin is a list of tuples representing tile centers (RA, DEC).
            The bins are ordered by ascending RA, and within each bin, the tiles are sorted by 
            descending DEC.
        
        Attributes
        ----------
        survey : str
            The survey type can either be "small" or "wide". 

        num_tiles : int
            Total number of tiles in the survey footprint boundary.

        footprint_area : float
            The total area (in square degrees) covered by the tiles.

        num_bins : int, optional
            Only applicable to WideSurvey.
            Number of bins the tiles are grouped in.

        Returns
        -------
        `Tiling` instance
        """
        self.boundary_coords = boundary_coords
        self.rotation_angle = rotation_angle
        self.tile_sequence = tile_sequence
        self.tile_shapes = tile_shapes
        self.tile_bins = tile_bins

        # Define attributes 
        self.num_tiles = len(self.tile_sequence)
        tile_area = (params.DETECTOR_LENGTH * params.DETECTOR_WIDTH) * 4
        self.footprint_area = round(tile_area * self.num_tiles, 2)
        
        # base on survey type
        if self.tile_bins:
            self.survey = "wide"
            self.num_bins = len(self.tile_bins)
        else:
            self.survey = "small"
            
    
    def __repr__(self):
        if self.survey == "small":
            return (
                f"<Tiling object: survey='{self.survey}', "
                f"footprint_area={self.footprint_area} deg², "
                f"num_tiles={self.num_tiles}>"
            )
        if self.survey == "wide":
            return (
                f"<Tiling object: survey='{self.survey}', "
                f"footprint_area={self.footprint_area} deg², "
                f"num_tiles={self.num_tiles}, "
                f"num_bins={self.num_bins}>"
            )

    
    def plot_snake_scan(self, tile_color='tab:blue', projection_type=None,
                        show_sequence=False, show_tile_centers=False, save_figure=False):
        """
        Plots the tiles on the survey footprint in snake scan tiling sequence.
    
        Parameters
        ----------
        tile_color : str, default='tab:blue'
            Color for the survey tiles.
    
        projection_type : None or 'mollweide'
            If None, uses equirectangular projection.
    
        show_sequence : bool, optional
            If True, plots the snake scan tiling sequence in red.
    
        show_tile_centers : bool, optional
            If True, plots the centers of the tiles.
    
        save_figure : bool, optional
            If `True`, saves the plot as 'plot_snake_scan.pdf'.
        """
        # Create plot
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection=projection_type)
    
        # Plot each tile multipolygon
        for tile_object in self.tile_shapes:
            for geom in tile_object.geoms:
                xs, ys = geom.exterior.xy
                if projection_type == 'mollweide':
                    ax.fill(np.radians(np.array(xs)), np.radians(np.array(ys)),
                            alpha=0.5, color=tile_color, ec=None)
                else:
                    ax.fill(xs, ys, alpha=0.5, color=tile_color, ec=None)
    
        if projection_type == 'mollweide':
            ax.grid(True)
            # Plot footprint boundary
            ax.plot(np.radians(self.boundary_coords[:, 0]), np.radians(self.boundary_coords[:, 1]),
                    linewidth=0.8, color='black')
    
            if show_tile_centers or show_sequence:
                ras, decs = zip(*self.tile_sequence)  # Unpack tile centers into separate lists
                if show_tile_centers:
                    ax.scatter(np.radians(ras), np.radians(decs), s=5, color='black')
                if show_sequence:
                    # Plot snake scan tiling sequence
                    ax.plot(np.radians(ras), np.radians(decs), linewidth=0.2, color='tab:red')
        else:
            ax.set_aspect(1)
            # Plot footprint boundary
            ax.plot(self.boundary_coords[:, 0], self.boundary_coords[:, 1],
                    linewidth=0.8, color='black')
    
            if show_tile_centers or show_sequence:
                ras, decs = zip(*self.tile_sequence)  # Unpack tile centers into separate lists
                if show_tile_centers:
                    ax.scatter(ras, decs, s=5, color=tile_color)
                if show_sequence:
                    # Plot snake scan tiling sequence
                    ax.plot(ras, decs, linewidth=0.6, color='tab:red')
    
        # Set labels and tick parameters
        ax.set_xlabel('R.A. (deg)', fontsize=14)
        ax.set_ylabel('Dec. (deg)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
        # Save or display the figure
        if save_figure:
            plt.savefig('plot_tiles.pdf')
        else:
            plt.show()
        
        
    def animate_snake_scan(
        self, tile_color, tiles_per_frame, frames_per_second=30, save_animation=False
    ):
        """
        Animates the snake scan tiling sequence.
    
        Parameters
        ----------
        tile_color : str
            Color for the survey tiles.
            
        tiles_per_frame : int
            Number of tiles plotted per frame.
            
        frames_per_second : int, optional, default=30
            Frame rate of the animation.
            
        save_animation : bool, optional
            If True, saves the animation as 'tiling_animation.gif'.
    
        Returns
        -------
        HTML or None
            Returns an HTML object to display the animation inline when
            'save_animation' is False. Otherwise, saves the animation as a gif.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from IPython.display import HTML
        
        # Create the plot
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111)

        # Plot footprint boundary
        ax.plot(self.boundary_coords[:, 0], self.boundary_coords[:, 1], linewidth=0.8, color='black')

        # Set plot parameters and labels
        footprint = Polygon(self.boundary_coords)
        xmin, ymin, xmax, ymax = footprint.bounds
        ax.axis([xmin - 1, xmax + 1, ymin - 1, ymax + 1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("R.A. (deg)", fontsize=12)
        ax.set_ylabel("Dec. (deg)", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_title("Snake Scan Tiling Strategy")
    
        # Define the number of frames
        num_frames = -(-self.num_tiles // tiles_per_frame)  # Ceiling division
    
        def animate(i):
            if i == 0:
                return []
    
            # Determine batch indices
            start_index = (i - 1) * tiles_per_frame
            end_index = min(start_index + tiles_per_frame, self.num_tiles)
    
            # Plot tiles for the current frame
            if self.num_tiles <= 200:
                # Small number of tiles: Plot fine tiles
                for tile_index in range(start_index, end_index):
                    tile_object = self.tile_shapes[tile_index]
                    for geom in tile_object.geoms:  # Iterate through MultiPolygon geometries
                        xs, ys = geom.exterior.xy
                        ax.fill(xs, ys, alpha=0.3, color=tile_color)
            else:
                # Large number of tiles: Plot coarse tiles
                for tile_center in self.tile_sequence[start_index:end_index]:
                    coarse_tile = create_course_tile(
                        tile_center[0],
                        tile_center[1],
                        self.rotation_angle,
                        l=params.DETECTOR_LENGTH,
                        w=params.DETECTOR_WIDTH,
                        g=params.DETECTOR_ARRAY_GAP,
                    )
                    x, y = coarse_tile.exterior.xy
                    ax.fill(x, y, alpha=0.3, color=tile_color)
    
            return ax.patches
    
        # Set up the animation
        ani = FuncAnimation(fig, animate, frames=num_frames + 1, interval=1000 / frames_per_second)
    
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams["animation.embed_limit"] = 2**128  # For large animations
    
        # Save or display the animation
        if save_animation:
            writer = PillowWriter(fps=frames_per_second)
            ani.save("tiling_animation.gif", writer=writer, dpi=150)
        else:
            return HTML(ani.to_jshtml())
            

    def plot_bins(self, tile_color='tab:blue', save_figure=False):
        """
        Plots WideSurvey tiles in bins.

        Parameters
        ----------
        tile_color : str, default='tab:blue'
            Color for the survey tiles.

        save_figure : bool, optional
            If `True`, saves the plot as 'plot_bins.pdf'.
        """
        if not self.survey == "wide":
            raise TypeError(
                "This function is designed specifically for WideSurvey tiling."
                "Use 'plot_snake_scan' function to plot SmallSurvey tiles."
            )
        # Create plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        
        # Plot footprint boundary
        ax.plot(self.boundary_coords[:, 0], self.boundary_coords[:, 1], linewidth=0.8, color='black')
        
        switch = True
        for bin in self.tile_bins:
            alpha_val = 0.3 if switch else 0.5
            for tile_center in bin:
                # Plot the (coarse) tile polygon
                coarse_tile = create_course_tile(
                    tile_center[0], 
                    tile_center[1], 
                    self.rotation_angle, 
                    l=params.DETECTOR_LENGTH, 
                    w=params.DETECTOR_WIDTH,
                    g=params.DETECTOR_ARRAY_GAP
                )
                x, y = coarse_tile.exterior.xy
                ax.fill(x, y, alpha=alpha_val, color=tile_color)
            switch = not switch 
        
        # Set labels and tick parameters
        ax.set_xlabel('R.A. (deg)', fontsize=14)
        ax.set_ylabel('Dec. (deg)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
        # Save or display the figure
        if save_figure:
            plt.savefig('plot_tiles.pdf')
        else:
            plt.show()