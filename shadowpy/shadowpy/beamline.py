"""
Module containing the Beamline class, which is a wrapper for a beamline. 
It also contains some helper functions for working with the beamline.
"""

import Shadow
import numpy as np
from .optical_elements import OpticalElement
from.sources import Source
from .utils import rotation_matrix, ReferenceFrame
from caxscripts.image_statistics import Histogram2DAnalyzer
import os
import sys
import contextlib

import os
import sys
import contextlib

@contextlib.contextmanager
def silence_c_libs():
    # 1. Flush Python's buffers to make sure order stays clean
    sys.stdout.flush()
    sys.stderr.flush()

    # 2. Save a duplicate copy of the original OS file descriptors (1 and 2)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    # 3. Open the null device
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        # 4. Force System FD 1 and 2 to point to /dev/null
        # This completely kills all C/Fortran printf outputs
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        # 5. Point Python's high-level sys.stdout to the SAVED stream copy.
        # This allows YOUR explicit python print() statements to keep working!
        original_stdout_stream = os.fdopen(saved_stdout_fd, 'w')
        sys.stdout = original_stdout_stream

        yield original_stdout_stream
    finally:
        # 6. Restore system behavior back to normal when exiting the block
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)

        # 7. Reset Python streams back to default system references
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Clean up stray handles
        os.close(devnull)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


class BeamLine:
    """
    A class representing a beamline, which is a sequence of optical elements.
    """

    def __init__(self, name: str, beam: Shadow.Beam, source: Source, 
                 optical_elements: list = None, beamline_frame=None, 
                 lab_frame=None):
        self.name = name
        self.beam = beam
        self.source = source
        self.elements = optical_elements if optical_elements is not None else []
        self.beamline_frame = beamline_frame
        self.lab_frame = lab_frame
        self.initialize_elements()

        # Initialize the source and save the initial image

        with silence_c_libs():
            self.beam.genSource(self.source.shadow_oe)

        self.save_image(self.source, self.beam)

        # List to hold beams at each stage of the beamline
        self.beams = list([None for _ in range(len(self.elements)+1)])  
        self.beams[0] = self.beam.duplicate() 

    def initialize_elements(self):
        """
        Initialize the beamline elements by constructiong their reference 
        frames and flagging them as part of the beamline. 
        
        The source reference frame is defined relative to the beamline frame, 
        and each subsequent elements are defined relative to the previous element's frame.
        """
        
        # Frame definitions
        if self.beamline_frame is None:
            raise ValueError("Beamline frame is not defined.")
        
        self.source.frame = self.beamline_frame.child_frame(relative_origin=[0, 0, 0], 
                                                            relative_rotation=np.eye(3),
                                                            name=f"{self.name}_{self.source.name}_frame")
        current_frame = self.source.frame

        # Iterate through the optical elements and construct their frames
        # Orientation is relative to the previous element, using the left-hand 
        # rule for positive angles

        for element in self.elements:
            relative_origin = [0, element.source_distance, 0]
            relative_orientation = rotation_matrix('y', -element.orientation)
            element.frame = current_frame.child_frame(relative_origin, 
                                                      relative_orientation,
                                                      name=f"{element.name}_frame")
            current_frame = element.frame

            # Add the optical elements to the beamline.
            element.beamline = self


    def save_image(self, element, beam: Shadow.Beam, nbins=200):
        """
        Save the image of the beam after passing through an optical element.
        """
        lab_x = np.array([1, 0, 0])  #Lab x direction
        lab_y = np.array([0, 1, 0])  #Lab y direction

        local_x = element.frame.vector_from_lab(lab_x)
        local_y = element.frame.vector_from_lab(lab_y)

        # 1 = x, 2 = y, 3 = z
        x_index = np.argmax(np.abs(local_x))+1 
        y_index = np.argmax(np.abs(local_y))+1

        image_ticket = beam.histo2(col_h=x_index, col_v=y_index, 
                                   nbins=nbins, nolost=1)
        
        histogram = image_ticket["histogram"]
        bin_h_edges = image_ticket["bin_h_edges"]
        bin_v_edges = image_ticket["bin_v_edges"]

        # We instantiate the analyzer to compute the image statistics, 
        # used for fitting the beam profile and computing the beam size.
        ana = Histogram2DAnalyzer(img=histogram,
                                  x_bin_edges=bin_h_edges,
                                  y_bin_edges=bin_v_edges)
        ana.compute_momenta()
        ana.fit(hprm=ana.hprm_momenta, useroi=True)
        # Set element image to the analyzer, used for fitting and plotting
        element.image = ana

    def full_trace(self):
        """
        Trace the beam through the whole beamline.
        """
        
        for i, element in enumerate(self.elements):
            self.beam.traceOE(element.shadow_oe, i+1)
            
            # Store the beam after each element
            self.beams[i+1] = self.beam.duplicate()

            # Save the image after each element
            self.save_image(element, self.beams[i+1])

    def trace(self):
        """
        Trace the beam through the beamline, starting from whichever element 
        was modified (and hence triggered this method).
        """

        # To check which element was modified, we check if its image is None, 
        # which means it has been wiped since the last trace. We then trace 
        # from the first modified element.
        start_index = 0
        for i, element in enumerate(self.elements):
            if element.image is None:
                start_index = i
                break
        
        for i in range(start_index, len(self.elements)):
            element = self.elements[i]
            
            # Get the current beam before tracing through the element
            current_beam = self.beams[i].duplicate()
            element.reset()  # Reset the element to apply any new parameters
  
            # Trace the beam through the current element
            with silence_c_libs():
                current_beam.traceOE(element.shadow_oe, i+1)
            print(f"Traced through element {i+1}: {element.name}")

            # Save the image after each element
            self.save_image(element, current_beam)
            
            # Update the beam after each element
            self.beams[i+1] = current_beam.duplicate()
            
            # Free memory of the current beam since we have saved it in the list
            del current_beam  
