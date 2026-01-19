import numpy as np
import pygmsh
from scipy.special import hankel1
from numba import njit


# ------------------------------------------------------------
# JIT-compiled inner assembly loop (no SciPy inside)
# ------------------------------------------------------------
@njit
def assemble_inner(triangles, areas, H):
    """
    Parameters
    ----------
    triangles : (M, 3) int64 array
        Triangle connectivity.
    areas : (M,) float64 array
        Triangle areas.
    H : (N_b, N) complex128 array
        H[i, n] = H0^{(1)}(k * r(i, n)), i.e. Hankel(0, k * distance).

    Returns
    -------
    F : (N_b, N) complex128 array
    """
    N_b, N = H.shape
    M = triangles.shape[0]

    F = np.zeros((N_b, N), dtype=np.complex128)

    c = -0.25j  # -1i/4
    for i in range(N_b):
        for j in range(M):
            i1 = triangles[j, 0]
            i2 = triangles[j, 1]
            i3 = triangles[j, 2]

            w = areas[j] / 3.0  # area/3

            F[i, i1] += w * c * H[i, i1]
            F[i, i2] += w * c * H[i, i2]
            F[i, i3] += w * c * H[i, i3]

    return F


class HelmholtzForwardOperator2D:
    """
    Builds a triangular mesh of a disk, and assembles the forward matrix F
    corresponding to the 2D Helmholtz single-layer potential integral.
    """

    def __init__(self):
        self.points = None      # (N, 2)
        self.triangles = None   # (M, 3)
        self.N = None
        self.M = None
        self.triangle_area = None  # (M,)

    # ------------------------------------------------------------
    # Mesh generation (disk of radius R0)
    # ------------------------------------------------------------
    def create_mesh(self, R0, h=0.05):
        with pygmsh.geo.Geometry() as geom:
            circle = geom.add_circle([0.0, 0.0, 0.0], R0, mesh_size=h)
            surf = geom.add_plane_surface(circle.curve_loop)

            mesh = geom.generate_mesh()

        pts = np.array(mesh.points[:, :2])
        tri = np.array(mesh.cells_dict["triangle"])

        self.points = pts
        self.triangles = tri
        self.N = pts.shape[0]
        self.M = tri.shape[0]
        self.triangle_area = self._compute_triangle_areas()

        return pts, tri

    # ------------------------------------------------------------
    # Boundary point generator (matches MATLAB boundary_points.m)
    # ------------------------------------------------------------
    def boundary_points(self, N_b, R, theta):
        """
        Generate N_b boundary points on circle of radius R,
        between angle 0 and theta (radians).
        """
        angles = np.linspace(0.0, theta, N_b)
        return np.column_stack((R*np.cos(angles), R*np.sin(angles)))

    # ------------------------------------------------------------
    # Precompute triangle areas
    # ------------------------------------------------------------
    def _compute_triangle_areas(self):
        A = np.zeros(self.M)
        pts = self.points
        tri = self.triangles

        for j in range(self.M):
            i1, i2, i3 = tri[j]
            x1, y1 = pts[i1]
            x2, y2 = pts[i2]
            x3, y3 = pts[i3]
            det = abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            A[j] = det / 2.0
        return A

    # ------------------------------------------------------------
    # Operator assembly (Numba-accelerated)
    # ------------------------------------------------------------
    def assemble_F(self, boundary_pts, k):
        """
        Assemble the forward operator matrix F.

        Parameters
        ----------
        boundary_pts : (N_b, 2) ndarray
        k : float
            Wavenumber.

        Returns
        -------
        F : complex ndarray of shape (N_b, N)
        """

        if self.points is None or self.triangles is None:
            raise RuntimeError("Mesh not created. Call create_mesh(...) first.")

        pts = self.points
        tri = self.triangles
        areas = self.triangle_area

        # 1) Compute all distances r(i, n) between boundary points and mesh nodes
        #    shape: (N_b, N)
        bp = boundary_pts
        # broadcast: (N_b, 1, 2) - (1, N, 2) -> (N_b, N, 2)
        diff = bp[:, None, :] - pts[None, :, :]
        R = np.sqrt(diff[..., 0]**2 + diff[..., 1]**2)

        # 2) Compute Hankel on full (N_b, N) grid (SciPy, outside Numba)
        H = hankel1(0, k * R)  # complex128 array

        # 3) Use Numba to do the triangle-based accumulation
        F = assemble_inner(tri, areas, H)

        return F
