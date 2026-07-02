from shadowpy.beamline import BeamLine
from shadowpy.sources import BendingMagnet
from shadowpy.optical_elements import ToroidalMirror, Screen, Slits
from shadowpy.utils import rotation_matrix, ReferenceFrame
import Shadow
import numpy as np
import os
import sys

# Utilize the cax-scripts repository for additional functionality
# sys.path.append(os.path.expanduser("~/repos/cax-scripts"))


class CAXSim(BeamLine):
    """
    Class to simulate the CARCARÁ-X (CAX) beamline using ShadowPy. 
    
    This class inherits from the Beamline class and provides methods to set 
    up and run simulations for the CAX beamline, perform scans and save 
    results.
    """

    sirius_frame = ReferenceFrame(name="sirius")

    R1 = rotation_matrix('x', 90)
    R2 = rotation_matrix('y', 180)
    shadow_R = R1 @ R2

    shadow_frame = sirius_frame.child_frame(relative_origin=[0, 0, 0], 
                                            relative_rotation=shadow_R, 
                                            name="shadow")

    def __init__(self, total_rays: int = 100000):
        # Initialize the elements
        source = BendingMagnet("B1", "../../specs/source.txt")
        
        self.mirror = ToroidalMirror(name="M1", 
                                     specification_file="specs/mirror.txt")
        self.dvf_A1 = Slits(name="DVF_A1", 
                            specification_file="specs/slits.txt")
        self.dvf_B1 = Screen(name="DVF_B1", 
                            specification_file="specs/screen.txt")
        beam = Shadow.Beam()
        
        source.shadow_oe.NPOINT = total_rays

        super().__init__(name="CAX", beam=beam, source=source, 
                 optical_elements=[self.mirror, self.dvf_A1, self.dvf_B1], 
                 beamline_frame=self.shadow_frame, 
                 lab_frame=self.sirius_frame)
        
        self.trace()