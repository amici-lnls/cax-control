"""
Module conaining wrapper classes around Shadow's Source class.
"""

import Shadow
import numpy as np

class Source():
    """
    A wrapper class for a Shadow source optical element.
    """

    def __init__(self, name: str, specification_file: str):
        self.name = name
        self.shadow_oe = Shadow.Source()
        self.specification_file = specification_file
        self.load_specification()
        self.pixel_size = None
        self.frame = None  # To be set when the element is added to a beamline
    
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

class BendingMagnet(Source):
    """
    A wrapper class for a Shadow bending magnet optical element.
    """

    def __init__(self, name: str, specification_file: str):
        super().__init__(name, specification_file)
        
        if self.shadow_oe.FDISTR != 4:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for a synchrotron source.")
