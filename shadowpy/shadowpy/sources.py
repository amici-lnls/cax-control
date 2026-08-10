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
        self.analyzer = None  # To be set when the element is added to a beamline
        self.frame = None  # To be set when the element is added to a beamline
        self.up_to_date = False  # Flag to indicate if the element's image is up-to-date
    
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

    @property
    def image(self):
        """
        Analyzer instance used to characterize the element's image
        """

        if self.beamline is None:
            raise Warning(f"Element {self.name} has not been added to a beamline."
                          f"Add this element to a BeamLine instance to trace it.")

        if not self.up_to_date: 
            # Retrace the beamline to update the beam with the new element 
            self.beamline.trace()

        return self.analyzer

class BendingMagnet(Source):
    """
    A wrapper class for a Shadow bending magnet optical element.
    """

    def __init__(self, name: str, specification_file: str):
        super().__init__(name, specification_file)
        
        if self.shadow_oe.FDISTR != 4:
            raise ValueError(f"File {specification_file} is not"
                             f" configured for a synchrotron source.")
