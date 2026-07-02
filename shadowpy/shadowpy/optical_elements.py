"""
Module containing the optical elements that can be used in ShadowPy. 

Each optical element is represented as a class that inherits from the base 
class `OpticalElement`. Each class has its own specific parameters and 
methods for calculating the effect of the optical element on the beam.
"""

import Shadow
import numpy as np

class OpticalElement:
    """
    A wrapper class for a Shadow optical element, 
    providing methods to load specifications and interact with the element.
    """

    OFFSET_ATTRIBUTES = ['OFFX', 'OFFY', 'OFFZ']
    TILT_ATTRIBUTES = ['X_ROT', 'Y_ROT', 'Z_ROT']

    PERSISTENT_ATTRIBUTES = OFFSET_ATTRIBUTES + TILT_ATTRIBUTES

    def __init__(self, name: str, specification_file: str):
        self.name = name
        self.shadow_oe = Shadow.OE()
        self.specification_file = specification_file
        self.load_specification()
        # Original properties to reset after simulation
        self.specification = self.shadow_oe.to_dictionary()
        # To be set when the element is added to a beamline
        self.frame = None  
        self.beamline = None
        # Image
        self.image = None
        # Persistent attributes that should not be reset after simulation
        self.persistent_attributes = self.PERSISTENT_ATTRIBUTES
        
    # General attributes    
        
    @property
    def orientation(self):
        """
        Optical element orientation angle (degrees) relative to the previous 
        element. Positive values indicate counter-clockwise rotation.
        """
        return self.shadow_oe.ALPHA
    

    @property
    def source_distance(self):
        """
        Distance from the previous element to the source point (user units).
        """
        return self.shadow_oe.T_SOURCE
    
    @property 
    def image_distance(self):
        """
        Distance from the previous element to the image point (user units).
        """
        return self.shadow_oe.T_IMAGE 

    @property
    def units(self):
        """
        User units for distances.
        """

        return self.shadow_oe.unit()
    
    @property
    def offset(self):
        """
        Get the current offset of the mirror's position in the global frame.
        
        Returns:
            np.ndarray: A 3D vector representing the offset in the global frame.
        """
        if self.frame is None:
            raise ValueError("Mirror frame is not defined. Please add the mirror to a beamline first.")
        
        offset_local = np.array([
            self.shadow_oe.OFFX,
            self.shadow_oe.OFFY,
            self.shadow_oe.OFFZ
        ])
        
        return self.frame.vector_to_lab(offset_local)
    
    @offset.setter
    def offset(self, offset_vector: np.ndarray):
        """
        Apply an offset to the mirror's position.
        
        Parameters:
            offset_vector (np.ndarray): A 3D vector representing the offset in the mirror's local frame.
        """
        if self.frame is None:
            raise ValueError("Mirror frame is not defined. Please add the mirror to a beamline first.")
        
        # Convert the offset vector from the mirror's local frame to the lab frame
        offset_local = self.frame.vector_from_lab(offset_vector)
        
        self.shadow_oe.OFFX = offset_local[0]
        self.shadow_oe.OFFY = offset_local[1]
        self.shadow_oe.OFFZ = offset_local[2]

        self.update()  # Update the element after changing the offset

    @property
    def tilt(self):
        """
        Get the current tilt angles of the mirror's orientation in the global frame.
        
        Returns:
            np.ndarray: A 3D vector representing the tilt angles (in degrees) around the lab frame axes.
        """
        if self.frame is None:
            raise ValueError("Mirror frame is not defined. Please add the mirror to a beamline first.")
        
        tilt_local = np.array([
            self.shadow_oe.X_ROT,
            self.shadow_oe.Y_ROT,
            self.shadow_oe.Z_ROT
        ])
        
        return self.frame.vector_to_lab(tilt_local)

    @tilt.setter
    def tilt(self, tilt_angles: np.ndarray):
        """
        Apply tilts to the mirror's orientation.
        
        Parameters:
            tilt_angles (np.ndarray): A 3D vector representing the tilt angles 
            (in degrees) around the lab frame axes.
        """
        if self.frame is None:
            raise ValueError("Mirror frame is not defined. Please add the mirror to a beamline first.")
        
        # Convert the tilt angles from the mirror's local frame to the lab frame
        self.tilt_global = tilt_angles
        self.tilt_local = self.frame.vector_from_lab(tilt_angles)
        
        self.shadow_oe.X_ROT = self.tilt_local[0]
        self.shadow_oe.Y_ROT = self.tilt_local[1]
        self.shadow_oe.Z_ROT = self.tilt_local[2]

        # Update the element after changing the tilt angles
        self.update()

    @property
    def tx(self):
        """
        Global X offset
        """
        return self.offset[0]
    
    @tx.setter
    def tx(self, value):
        offset = self.offset
        offset[0] = value
        self.offset = offset

    @property
    def ty(self):
        """
        Global Y offset
        """
        return self.offset[1]
    
    @ty.setter
    def ty(self, value):
        offset = self.offset
        offset[1] = value
        self.offset = offset

    @property
    def tz(self):
        """
        Global Z offset
        """
        return self.offset[2]
    
    @tz.setter
    def tz(self, value):
        offset = self.offset
        offset[2] = value
        self.offset = offset

    @property
    def rx(self):
        """
        Global X tilt angle (degrees)
        """
        return self.tilt[0]

    @rx.setter
    def rx(self, value):
        tilt = self.tilt
        tilt[0] = value
        self.tilt = tilt

    @property
    def ry(self):
        """
        Global Y tilt angle (degrees)
        """
        return self.tilt[1]

    @ry.setter
    def ry(self, value):
        tilt = self.tilt
        tilt[1] = value
        self.tilt = tilt

    @property
    def rz(self):
        """
        Global Z tilt angle (degrees)
        """
        return self.tilt[2]

    @rz.setter
    def rz(self, value):
        tilt = self.tilt
        tilt[2] = value
        self.tilt = tilt

    def update(self):
        """
        Update the optical element after changing its parameters. This method 
        should be called after modifying any parameters to ensure the changes 
        are applied to the beamline.
        """
        # Reset the element to apply the new tilt angles
        self.reset()  
        # Clear the image since the element has changed
        self.image = None  
        # Retrace the beamline to update the beam with the new element 
        if self.beamline is not None:
            self.beamline.trace()  

    def load_specification(self, specification_file: str = None):
        """
        Load the optical element specification from a file.
        
        Parameters:
            specification_file (str): The path to the specification file. 
            If None, uses the default specification file.
        """

        if specification_file is None:
            specification_file = self.specification_file
        self.shadow_oe.load(specification_file)
        # self.specification = self.shadow_oe.to_dictionary()

    def reset(self):
        """
        Reset the optical element to its original specification.
        """
        for key, value in self.specification.items():
            if key not in self.persistent_attributes:
                setattr(self.shadow_oe, key, value)

# ========================================================================
#                        OPTICAL ELEMENTS
# ========================================================================

class ToroidalMirror(OpticalElement):
    """
    A wrapper class for a Shadow toroidal mirror optical element.
    """

    def __init__(self, name: str, specification_file: str):
        super().__init__(name, specification_file)
        
        if self.shadow_oe.FMIRR != 3:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for a toroidal mirror: "
                             f"FMIRR = {self.shadow_oe.FMIRR}")


class Screen(OpticalElement):
    """
    A wrapper class for a Shadow screen optical element.
    """

    def __init__(self, name: str, specification_file: str, 
                 screen_dimensions: tuple = (5., 5.)):
        super().__init__(name, specification_file)
        
        if self.shadow_oe.F_SCREEN != 1:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for a screen: "
                             f"F_SCREEN = {self.shadow_oe.F_SCREEN}")

class Slits(Screen):
    """
    A wrapper class for a Shadow slits optical element.
    """

    def __init__(self, name: str, specification_file: str, 
                 slit_positions: tuple = (0., 0.)):
        super().__init__(name, specification_file)
        
        if any(self.shadow_oe.I_SLIT) != 1:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for slits: "
                             f"I_SLIT = {self.shadow_oe.I_SLIT}")
