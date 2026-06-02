"""
Module containing utility functions for the ShadowPy package.
"""

import numpy as np

def rotation_matrix(axis, angle):
    """
    Generate a rotation matrix for a given axis and angle.
    
    Parameters:
        axis (str): The axis of rotation ('x', 'y', or 'z').
        angle (float): The angle of rotation in degrees.
    
    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    angle = np.radians(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")
    

class ReferenceFrame:
    """
    Right-handed orthonormal reference frame.

    Convention:
        v_lab = R @ v_local

    Columns of R are the local basis vectors expressed
    in lab coordinates.
    """

    def __init__(self, origin=None, orientation=None, name=None):

        self.name = name or "frame"

        self.origin = np.array(
            origin if origin is not None else [0., 0., 0.],
            dtype=float
        )

        self.R = np.array(
            orientation if orientation is not None else np.eye(3),
            dtype=float
        )

        self._validate_rotation_matrix()

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    def _validate_rotation_matrix(self, atol=1e-10):

        should_be_identity = self.R.T @ self.R

        if not np.allclose(should_be_identity, np.eye(3), atol=atol):
            raise ValueError("Orientation matrix is not orthogonal")

        det = np.linalg.det(self.R)

        if not np.isclose(det, 1.0, atol=atol):
            raise ValueError("Orientation matrix must have determinant +1")

    # -------------------------------------------------
    # Vector transforms
    # -------------------------------------------------

    def vector_to_lab(self, v_local):
        v_local = np.asarray(v_local, dtype=float)
        return self.R @ v_local

    def vector_from_lab(self, v_lab):
        v_lab = np.asarray(v_lab, dtype=float)
        return self.R.T @ v_lab

    # -------------------------------------------------
    # Point transforms
    # -------------------------------------------------

    def point_to_lab(self, p_local):
        p_local = np.asarray(p_local, dtype=float)
        return self.origin + self.R @ p_local

    def point_from_lab(self, p_lab):
        p_lab = np.asarray(p_lab, dtype=float)
        return self.R.T @ (p_lab - self.origin)

    # -------------------------------------------------
    # Relative frame construction
    # -------------------------------------------------

    def child_frame(self, relative_origin, relative_rotation, name=None):
        """
        Create a new frame defined relative to THIS frame.
        """

        relative_origin = np.asarray(relative_origin, dtype=float)
        relative_rotation = np.asarray(relative_rotation, dtype=float)

        new_origin = self.point_to_lab(relative_origin)
        new_rotation = self.R @ relative_rotation

        return ReferenceFrame(
            origin=new_origin,
            orientation=new_rotation,
            name=name
        )

    # -------------------------------------------------
    # Relations between frames
    # -------------------------------------------------

    def rotation_to(self, other):
        """
        Rotation matrix mapping vectors from this frame
        into another frame.
        """

        return other.R.T @ self.R

    def __repr__(self):
        return (
            f"ReferenceFrame(name={self.name}, "
            f"origin={self.origin})"
        )