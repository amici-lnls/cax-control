#%% Imports

from optcore import Undulator, CRL, VerboseInhibitor, get_input_path, clean_shadow_files, plot_wrapper

import os
from dataclasses import dataclass
from typing import Union, List

import Shadow
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOpticalElement

from optlnls.hybrid import run_hybrid
from optlnls.crystal import calc_Bragg_angle

import numpy as np

from tqdm import trange

#%%

class ARI_SHADOW3:
    
    def __init__(self,
                 workspace = os.getcwd() + '/',
                 dcm: bool = False, optics: str = None
                 ):
        
        self.name = 'ARI'
        self.workspace = workspace
        self.source = Undulator()
        self.setup = self.Setup(dcm = dcm, optics = optics)
        self.Set_Beamline()
    
    @dataclass
    class Setup:
        dcm: bool = False
        optics: str = None # Can be "KB", "CRL", or None
        
        def __post_init__(self):
            allowed_optics = {'KB', 'CRL', None}
            if (not self.dcm and self.optics is not None) or (self.dcm and self.optics not in allowed_optics):
                print('Warning: Invalid combination (dcm={}, optics={}).'.format(self.dcm,
                                                                                 self.optics))
                print('Setting optics to None.')
                self.optics = None
    
    def Set_CRL(self,
                n: int = 100,
                radius: float = 50e-6,
                aperture: float = 660e-6,
                apexes_space: float = 30e-06,
                element: str = 'Be',
                density: float = None,
                energy: float = 20,
                p = 33):
        
        crl = CRL(n = n, radius = radius, aperture = aperture,
                  apexes_space = apexes_space, element = element,
                  density = density, energy = energy, p = p)
        crl.Create_Shadow3_OE()
        
        self.crl = crl
    
    def Set_Beamline(self,
                     nrays: int = 1_000_000, seed: int = 5_676_561,
                     energy: float = 20000, BW: float = 0.001,
                     
                     WB_Slit_z: float = 28073.145,
                     acceptance: Union[List[float], np.ndarray] = [66e-6, 66e-6],
                     
                     DCM_z: float = 29726.495, crystal: str = 'Si',
                     crystal_deformation: str = None,
                     crystal_deformation_file: str = '',
                     
                     VFM_z: float = 31749.720, HFM_z: float = 32049.720,
                     VFM_L: float = 170, HFM_L: float = 300,
                     VFM_angle: float = 2.0e-3, HFM_angle: float = 2.0e-3,
                     KB_coating: str = get_input_path('ari',
                                                      'reflectivities',
                                                      'Rh.dat'),
                     KB_multilayer: bool = False,
                     VFM_error: str = get_input_path('ari',
                                                     'surfaces',
                                                     'VFM--ESRF_Report_Shape_Error--Unbent.dat'),
                     HFM_error: str = get_input_path('ari',
                                                     'surfaces',
                                                     'HFM--ESRF_Report_Shape_Error--Unbent.dat'),
                     
                     crl = CRL(),
                     crl_prerefl_file_path: str = None,
                     
                     # Focus_z = 43143.099, # Reflectometer
                     # Focus_z = 50330.670, # Diffractometer 1
                     Focus_z: float = 53330.670, # Optical table center
                     # Focus_z = 56525.596, # Diffractometer 2

                     Horizontal_focus: float = None,
                     Vertical_focus: float = None,

                     Image_z: float = None,
                     
                     # Misalignment not implemented yet
                     moving_elements = None, movement_types = [''],
                     movements = [1]
                     ):
        
        self.source.Set_Harmonic_from_Energy(energy)
        self.source.Set_Photon_Beam()
        

        if self.setup.optics == 'KB':
            self.acceptance = np.array([HFM_L * HFM_angle / HFM_z,
                                        VFM_L * VFM_angle / VFM_z])
        
        elif self.setup.optics == 'CRL':
            
            if crl_prerefl_file_path is None:
                crl.Create_Shadow3_OE()
            else:
                crl.Create_Shadow3_OE(prerefl_file_path = crl_prerefl_file_path)
            
            self.crl = crl
            
            self.acceptance = [2 * self.crl.radius0 / (self.crl.source_to_first_interface + self.crl.previous_optical_element_z),
                               2 * self.crl.radius0 / (self.crl.source_to_first_interface + self.crl.previous_optical_element_z)]
        
        else:
            self.acceptance = acceptance
    
        if Image_z is None:
            Image_z = Focus_z
        if (Image_z - HFM_z) < 0:
            print('Warning: Image position behind KB. Setting Image_z to Optical Table.')
            Image_z = 53330.670
        if Horizontal_focus is None:
            Horizontal_focus = Focus_z
        if Vertical_focus is None:
            Vertical_focus = Focus_z

        self.energy = energy
        
        source = Shadow.Source()
        wb_slit = Shadow.OE()
        dcm_first_crystal = Shadow.OE()
        dcm_second_crystal = Shadow.OE()
        vfm = Shadow.OE()
        hfm = Shadow.OE()
        
        final_retrace = Shadow.OE()
        
        ###########################################################################
        # Geometrical Source
        source.FDISTR = 3
        source.F_COLOR = 3
        source.F_PHOT = 0
        source.HDIV1 = self.acceptance[0] / 2 #Acceptances [rad]
        source.HDIV2 = self.acceptance[0] / 2
        source.VDIV1 = self.acceptance[1] / 2
        source.VDIV2 = self.acceptance[1] / 2
        source.IDO_VX = 0
        source.IDO_VZ = 0
        source.IDO_X_S = 0
        source.IDO_Y_S = 0
        source.IDO_Z_S = 0
        source.ISTAR1 = seed # seed for Monte Carlo method
        source.NPOINT = nrays # Number of rays
        source.SIGMAX = self.source.photon_beam_size[0] * 1e-3 # Horizontal source size [mm]
        source.SIGMAZ = self.source.photon_beam_size[1] * 1e-3 # Vertical source size [mm]
        source.SIGDIX = self.source.photon_beam_divergence[0] * 1e-6 #Horizontal source divergence [rad]
        source.SIGDIZ = self.source.photon_beam_divergence[1] * 1e-6 #Vertical source divergence [rad]
        source.PH1 = self.energy - self.energy * BW / 2 # Minimum energy
        source.PH2 = self.energy + self.energy * BW / 2 # Maximum energy
        
        ###########################################################################
        # Defining aperture slit
        wb_slit.DUMMY = 0.1
        wb_slit.FWRITE = 0
        wb_slit.F_REFRAC = 2
        wb_slit.F_SCREEN = 1
        wb_slit.I_SLIT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        wb_slit.N_SCREEN = 1
        wb_slit.RX_SLIT = np.array([self.acceptance[0] * WB_Slit_z,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        wb_slit.RZ_SLIT = np.array([self.acceptance[1] * WB_Slit_z,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        wb_slit.T_IMAGE = 0.0
        wb_slit.T_INCIDENCE = 0.0
        wb_slit.T_REFLECTION = 180.0
        wb_slit.T_SOURCE = WB_Slit_z
        
        ###########################################################################
        # DCM
        if crystal == 'Si':
            crystal_path = get_input_path('ari', 'crystals', 'Si111.dat')
        ang = 90 - calc_Bragg_angle(crystal = crystal, energy = self.energy/1000,
                                    h = 1, k = 1, l = 1, 
                                    corrected=True)*180/np.pi
        
        # DCM First Crystal
        dcm_first_crystal.DUMMY = 0.1
        dcm_first_crystal.FILE_REFL = crystal_path.encode('UTF-8')
        dcm_first_crystal.FWRITE = 0
        dcm_first_crystal.F_CENTRAL = 1
        dcm_first_crystal.F_CRYSTAL = 1
        dcm_first_crystal.PHOT_CENT = self.energy
        dcm_first_crystal.R_LAMBDA = self.energy
        dcm_first_crystal.T_IMAGE = 50.0
        dcm_first_crystal.T_INCIDENCE = ang
        dcm_first_crystal.T_REFLECTION = ang
        dcm_first_crystal.T_SOURCE = DCM_z - WB_Slit_z - 50 # DCM position minus WB Slit - half crystal separation
        if crystal_deformation is not None:
            dcm_first_crystal.FHIT_C = 0
            dcm_first_crystal.F_G_S = 2
            dcm_first_crystal.F_RIPPLE = 1
            dcm_first_crystal.FILE_RIP = (self.workspace + crystal_deformation_file).encode('UTF-8')
        
        # DCM Second Crystal
        dcm_second_crystal.ALPHA = 180.0
        dcm_second_crystal.DUMMY = 0.1
        dcm_second_crystal.FILE_REFL = crystal_path.encode('UTF-8')
        dcm_second_crystal.FWRITE = 1
        dcm_second_crystal.F_CENTRAL = 1
        dcm_second_crystal.F_CRYSTAL = 1
        dcm_second_crystal.PHOT_CENT = self.energy
        dcm_second_crystal.R_LAMBDA = self.energy
        dcm_second_crystal.T_IMAGE = -50.0
        dcm_second_crystal.T_INCIDENCE = ang
        dcm_second_crystal.T_REFLECTION = ang
        dcm_second_crystal.T_SOURCE = 50.0
        
        ###########################################################################
        # KB VFM
        if KB_multilayer:
            vfm.F_REFL = 2
        vfm.ALPHA = 180.0
        vfm.DUMMY = 0.1
        vfm.FCYL = 1
        vfm.FHIT_C = 1
        vfm.FILE_REFL = KB_coating.encode('UTF-8')
        vfm.FMIRR = 2
        vfm.FWRITE = 0
        vfm.F_DEFAULT = 0
        vfm.F_REFLEC = 1
        vfm.RLEN1 = VFM_L / 2
        vfm.RLEN2 = VFM_L / 2
        vfm.RWIDX1 = 15.0
        vfm.RWIDX2 = 15.0
        vfm.T_SOURCE = VFM_z - DCM_z
        vfm.SSOUR = VFM_z
        vfm.T_IMAGE = Image_z - VFM_z
        vfm.SIMAG = Vertical_focus - VFM_z
        vfm.T_INCIDENCE = 90 - np.rad2deg(VFM_angle)
        vfm.T_REFLECTION = 90 - np.rad2deg(VFM_angle)
        vfm.THETA = 90 - np.rad2deg(VFM_angle)
        
        ###########################################################################
        # KB HFM
        if KB_multilayer:
            hfm.F_REFL = 2
        hfm.ALPHA = 270.0
        hfm.DUMMY = 0.1
        hfm.FCYL = 1
        hfm.FHIT_C = 1
        hfm.FILE_REFL = KB_coating.encode('UTF-8')
        hfm.FMIRR = 2
        hfm.FWRITE = 0
        hfm.F_DEFAULT = 0
        hfm.F_REFLEC = 1
        hfm.RLEN1 = HFM_L / 2
        hfm.RLEN2 = HFM_L / 2
        hfm.RWIDX1 = 20.0
        hfm.RWIDX2 = 20.0
        hfm.T_SOURCE = -(Image_z - HFM_z)
        hfm.SSOUR = HFM_z
        hfm.T_IMAGE = Image_z - HFM_z
        hfm.SIMAG = Horizontal_focus - HFM_z
        hfm.THETA = 90 - np.rad2deg(HFM_angle)
        hfm.T_INCIDENCE = 90 - np.rad2deg(HFM_angle)
        hfm.T_REFLECTION = 90 - np.rad2deg(HFM_angle)
        
        self.Hybrid_calcType = np.ones(2)
        
        if VFM_error is not None:
            print('VFM error applied')
            self.Hybrid_calcType[0] = 2
            vfm.F_G_S = 2
            vfm.F_RIPPLE = 1
            vfm.FILE_RIP = VFM_error.encode('UTF-8')
            
        if HFM_error is not None:
            print('HFM error applied')
            self.Hybrid_calcType[1] = 2
            hfm.F_G_S = 2
            hfm.F_RIPPLE = 1
            hfm.FILE_RIP = HFM_error.encode('UTF-8')
        
        ###################### Implemennt misalignment
        # if moving_elements is not None:
        #     for i, moving_element in enumerate(moving_elements):
                
        #         ind = ind_dict[moving_element]
        #         self.oe[ind].F_MOVE = 1
                
        #         print('{} {} {:.3e}'.format(moving_element,
        #                                     movement_types[i],
        #                                     movements[i]))
                
        #         if movement_types[i] == 'Tx':
        #             self.oe[ind].OFFX = movements[i]
        #         elif movement_types[i] == 'Ty':
        #             self.oe[ind].OFFY = movements[i]
        #         elif movement_types[i] == 'Tz':
        #             self.oe[ind].OFFZ = movements[i]
        #         elif movement_types[i] == 'Rx':
        #             self.oe[ind].X_ROT = movements[i]
        #         elif movement_types[i] == 'Ry':
        #             self.oe[ind].Y_ROT = movements[i]
        #         elif movement_types[i] == 'Rz':
        #             self.oe[ind].Z_ROT = movements[i]
        
        ###########################################################################
        final_retrace.DUMMY = 0.1
        final_retrace.FWRITE = 3
        final_retrace.F_REFRAC = 2
        final_retrace.T_INCIDENCE = 0.0
        final_retrace.T_REFLECTION = 180.0
        final_retrace.T_SOURCE = 0.0
        
        # White Beam Condition
        self.oe_list = [source, wb_slit]
        final_retrace.ALPHA = 0.0
        if (Image_z - WB_Slit_z) < 0:
            print('Warning: Image position behind WB Slit. Setting Image_z to Optical Table.')
            Image_z = 53330.670
        final_retrace.T_IMAGE = Image_z - WB_Slit_z
        
        # DCM Condition
        if self.setup.dcm:
            self.oe_list.append(dcm_first_crystal)
            self.oe_list.append(dcm_second_crystal)
            final_retrace.ALPHA = 180.0
            if (Image_z - DCM_z) < 0:
                print('Warning: Image position behind DCM. Setting Image_z to Optical Table.')
                Image_z = 53330.670
            final_retrace.T_IMAGE = Image_z - DCM_z
            
            # KB Condition
            if self.setup.optics == 'KB':
                self.oe_list.append(vfm)
                self.oe_list.append(hfm)
                final_retrace.ALPHA = 90.0
                final_retrace.T_IMAGE = 0.0
            
            # CRL Condition
            elif self.setup.optics == 'CRL':
                self.oe_list = self.oe_list + self.crl.shadow_crl
                final_retrace.ALPHA = 0.0
                final_retrace.T_IMAGE = Image_z - (self.crl.p + self.crl.l/2 + self.crl.principal_plane) * self.crl.meters_to_shadow_units
        
        self.oe_list.append(final_retrace)
    
    def Open_Shutter(self, run_till: str = 'Sample',
                     last: int = None, save_beam_name: str = None,
                     use_hybrid: bool = False, hybrid_diff_plane: int = 3,
                     hybrid_near_field: int = 0,
                     analize_geometry_to_avoid_unuseful_calculation: int = 1,
                     verbose: bool = True, use_tqdm: bool = True,
                     clean: bool = True):
        if verbose:
            print('\nRunning beamline')
        else:
            use_tqdm = False
        
        silence_shadow = VerboseInhibitor()
        silence_shadow.start()
        
        if last is None:
            last = len(self.oe_list)
            
        if (self.setup.optics == 'KB'):
            beam = ShadowBeam()
            beam = beam.traceFromSource(ShadowSource(src = self.oe_list[0]))
            
            for i in trange(1, last) if use_tqdm else range(1, last):
                
                if not use_tqdm:
                    silence_shadow.print(f"    Running optical element: {i} / {last - 1}")
                
                if (i in [4,5]):
                    beam = beam.traceFromOE(beam, ShadowOpticalElement(self.oe_list[i]),
                                            widget_class_name = 'Mirror')
                    if use_hybrid:
                        print('Using hybrid')
                        beam = run_hybrid(beam, calcType = self.Hybrid_calcType[i-4],
                                          diff_plane = hybrid_diff_plane,
                                          nf = hybrid_near_field,
                                          automatic = analize_geometry_to_avoid_unuseful_calculation
                                          )
                else:
                    beam = beam.traceFromOE(beam, ShadowOpticalElement(self.oe_list[i]))
            
            self.beam = beam._beam
        
        else:
            self.beam = Shadow.Beam()
            self.beam.genSource(self.oe_list[0])
            
            for i in trange(1, last) if use_tqdm else range(1, last):
                
                if not use_tqdm:
                    silence_shadow.print(f"    Running optical element: {i} / {last - 1}")
                
                self.beam.traceOE(self.oe_list[i], i)
        
        if clean: clean_shadow_files(directory = self.workspace)
        
        if save_beam_name is not None: self.Save_Beam(save_beam_name = save_beam_name)
    
    def Save_Beam(self, save_beam_name: str = None):
        if hasattr(self, 'beam'):
            if save_beam_name is not None:
                self.beam.write(save_beam_name)
            else: print('No name to save beam!')
        else: print('No beam defined!')
# %%
