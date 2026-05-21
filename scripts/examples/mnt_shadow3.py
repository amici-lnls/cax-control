#%% Imports

from typing import List, Union

from optcore import Bending_Magnet, VerboseInhibitor, clean_shadow_files

import os
import xraylib

import Shadow
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOpticalElement

from optlnls.hybrid import ShadowPhysics

import numpy as np

#%%

class MNT_SHADOW3:
    
    def __init__(self,
                 workspace = os.getcwd() + '/',
                 ):
        
        self.name = 'MNT'
        self.workspace = workspace
        self.source = Bending_Magnet()
        self.Set_Beamline()
    
    
    def Set_Beamline(self,
                     n_rays=500000, energy=52397.0, delta_E=800, 
                     acceptance: Union[List[float], np.ndarray] = [1.5e-3, 0.66e-3],
                     alpha_deg=34.3, radius= 191242.5, thickness=2.0, convexo=True, size_x=0.0193, size_z=0.0019,
                     p_distance=27000.0, at_focus: bool = False, 
                     offx: float = 0.0, offy: float = 0.0, offz: float = 0.0, rotx: float = 0.0, roty: float = 0.0, rotz: float = 0.0,
                     moving_elements=None, movement_types=None, movements=None
                     # Misalignment not implemented yet
                     ):
        

        self.energy = energy
        self.acceptance = acceptance
        source = Shadow.Source()
        crystal = Shadow.OE()
        final_retrace = Shadow.OE()

        self.source.magnetic_field=6.6 #SWLS
        
        if energy == 52397.0:
            propag_dist = 42000.0
            if at_focus==True:
                propag_dist = 13460.0

        elif energy == 85000.0:
            propag_dist = 37000.0
            if at_focus==True:
                propag_dist = 15100.0


        ###########################################################################
        # Source
        source.BENER = self.source.ring_energy
        source.EPSI_X = 2.475247525e-07
        source.EPSI_Z = 2.475247525e-09
        source.FDISTR = 4
        source.FILE_BOUND = b'C:/Users/fernanda.labio/Documents/BEAMLINES/MANATI/SUSSUARANA_optimize_source_1p5x0p66mrad2.txt'
        source.FSOURCE_DEPTH = 4
        source.F_BOUND_SOUR = 2
        source.F_COLOR = 3
        source.F_PHOT = 0
        source.HDIV1 = self.acceptance[0] / 2 #accept_hor_rad/2 #0.00075
        source.HDIV2 = self.acceptance[0] / 2 #accept_hor_rad/2 #0.00075
        source.ISTAR1 = 5676561
        source.NCOL = 0
        source.NPOINT = n_rays
        source.N_COLOR = 0
        source.PH1 = energy - delta_E/2 #51897.0
        source.PH2 = energy + delta_E/2 #52897.0
        source.POL_DEG = 0.0
        source.R_ALADDIN = 1516.20043271887
        source.R_MAGNET = 1.51620043271887
        source.SIGDIX = 0.0
        source.SIGDIZ = 0.0
        source.SIGMAX = size_x #0.0193   0.0000193
        source.SIGMAY = 0.0
        source.SIGMAZ = size_z #0.0019
        source.VDIV1 = self.acceptance[1] / 2 #accept_ver_rad/2 #0.00033
        source.VDIV2 = self.acceptance[1] / 2 #accept_ver_rad/2 #0.00033
        source.WXSOU = 0.0
        source.WYSOU = 0.0
        source.WZSOU = 0.0
        

        # print('Source beam energy:', source.BENER, 'GeV')
        # print('Source ring current:', self.source.ring_current, 'mA')
        # print('Source magnetic field:', self.source.magnetic_field, 'T')
        # print('Source electron beam divergence:', self.source.electron_beam_divergence[1], 'rad')
        # print('Source horizontal acceptance:', self.acceptance[0], 'rad')
        # print('Source vertical acceptance:', self.acceptance[1], 'rad')

        # print('Emitancia horizontal [rad.mm]:', source.EPSI_X)
        # print('Emitancia vertical [rad.mm]:', source.EPSI_Z)
        # print('Acceptance horizontal [rad]:', self.acceptance[0])
        # print('Acceptance vertical [rad]:', self.acceptance[1])
        # print('Magnetic Radius [mm]:', source.R_MAGNET * 1e3)
        # print('Source size horizontal [mm]:', source.SIGMAX)
        # print('Source size vertical [mm]:', source.SIGMAZ)
        # ###########################################################################
    
        #CRISTAL Laue
        crystal.ALPHA = 270.0
        crystal.A_BRAGG = alpha_deg # alpha_deg
        crystal.DUMMY = 0.1 #define em mm
        crystal.FCYL = 1 #cilindrico
        crystal.FILE_REFL = b'C:/Users/fernanda.labio/Documents/BEAMLINES/MANATI/Si333.dat'
        crystal.FMIRR = 1 #spherical
        #crystal.F_ANGLE = 1
        crystal.FWRITE = 2 #TUDO
        crystal.F_BRAGG_A = 1 #cristal assimetrico
        crystal.F_CENTRAL = 1 #autotuning
        crystal.F_CONVEX = convexo # 0-concavo, 1-convexo,
        crystal.F_CRYSTAL = 1 #é cristal
        crystal.F_EXT = 1 #user defined parameters
        crystal.F_MOVE=1
        crystal.F_REFRAC = 1 #reflector (0). refractor (1), empty element (2).
        crystal.PHOT_CENT = energy #energia selecionada
        crystal.RMIRR = radius 
        crystal.R_LAMBDA = 5000.0
        crystal.THICKNESS = thickness #mm
        crystal.RLEN1 = 5.0  #size crystal
        crystal.RLEN2 = 5.0
        crystal.RWIDX1 = 100.0 #size crystal
        crystal.RWIDX2 = 100.0
        crystal.T_IMAGE = 0.0 #distance from mirror to image plane in mm
        if energy == 52397.0 and alpha_deg==34.3:
            crystal.T_INCIDENCE = 49.2002483665
            crystal.T_REFLECTION = 117.7999122197
            print('Energy: 52397 eV')
        elif energy == 85000.0 and alpha_deg==34.3:
            crystal.T_INCIDENCE = 51.6983443533
            crystal.T_REFLECTION = 120.2985811643
            print('Energy: 85000 eV')
        else:
            raise ValueError("Incidence/reflection angles not defined for this energy")
        crystal.T_SOURCE = p_distance #distancia para fonte
        
        # crystal.OFFX = offx #Tx
        # crystal.OFFY = offy #Ty
        # crystal.OFFZ = offz #Tz
        # crystal.X_ROT = rotx #Rx
        # crystal.Y_ROT = roty #Ry
        # crystal.Z_ROT = rotz #Rz
        # if crystal.OFFX != 0.0 or crystal.OFFY != 0.0 or crystal.OFFZ != 0.0:
        #     print(f"Applying crystal translations: Tx={crystal.OFFX} mm, Ty={crystal.OFFY} mm, Tz={crystal.OFFZ} mm")
        # if crystal.X_ROT != 0.0 or crystal.Y_ROT != 0.0 or crystal.Z_ROT != 0.0:
        #     print(f"Applying crystal rotations: Rx={crystal.X_ROT} deg, Ry={crystal.Y_ROT} deg, Rz={crystal.Z_ROT} deg")

        ###########################################################################
        
        ###################### Implemennt misalignment
        if moving_elements is not None:
            for i, moving_element in enumerate(moving_elements):
                
                print('{} {} {:.3e}'.format(moving_element,
                                            movement_types[i],
                                            movements[i]))
                
                if movement_types[i] == 'Tx':
                    crystal.OFFX = movements[i]
                elif movement_types[i] == 'Ty':
                    crystal.OFFY = movements[i]
                elif movement_types[i] == 'Tz':
                    crystal.OFFZ = movements[i]
                elif movement_types[i] == 'Rx':
                    crystal.X_ROT = movements[i]
                elif movement_types[i] == 'Ry':
                    crystal.Y_ROT = movements[i]
                elif movement_types[i] == 'Rz':
                    crystal.Z_ROT = movements[i]
        ###########################################################################
        
        final_retrace.ALPHA =90.0
        final_retrace.T_IMAGE = propag_dist
        final_retrace.DUMMY = 0.1
        final_retrace.FWRITE = 3
        final_retrace.F_REFRAC = 2
        final_retrace.T_INCIDENCE = 90.0
        final_retrace.T_REFLECTION = 90.0
        final_retrace.T_SOURCE = 0.0
        

        self.oe_list = [source]
        self.oe_list.append(crystal)
        self.oe_list.append(final_retrace)
    
    def Open_Shutter(self,
                     last: int = None, save_beam_name: str = None,
                     verbose: bool = True,
                     energy: float = 52397.0, crystal: str ='Si', hkl: list = [3, 3, 3], alpha_deg: float=34.3,
                     filename_txt: str = "MNT_difraction_profile.txt", oe_number: int = 1):
        
        
        if verbose:
            print('\nRunning beamline')
        else:
            silence_shadow = VerboseInhibitor()
            silence_shadow.start()
        
        if last is None:
            last = len(self.oe_list)
            
        beam_before=ShadowBeam()
        beam_after=ShadowBeam()
        beam_before = beam_before.traceFromSource(ShadowSource(src = self.oe_list[0]))



#beam afterrrr

        beam_after = beam_before.duplicate()
        for i in range(1, last):
            if verbose:
                print(f"    Running optical element: {i} / {last - 1}")
            beam_after=beam_after.traceFromOE(beam_after, ShadowOpticalElement(self.oe_list[i]))

        
#contasssss
        #o fluxo ta saindo errado vei
        if energy==52397.0:
            angle_arq = "/252angle."
        elif energy==85000.0:
            angle_arq = "/285angle."
        values = np.loadtxt(os.path.abspath(os.path.curdir + angle_arq +
                                                ("0" + str(oe_number) if (oe_number < 10) else
                                                    str(oe_number))))

        beam_incident_angles = (values[:, 1] - 90) #degrees #em relação à sup

        beam_wavelengths     = ShadowPhysics.getWavelengthFromShadowK(beam_after._beam.rays[:, 10]) 

        d_spacing            = xraylib.Crystal_dSpacing(xraylib.Crystal_GetCrystal(crystal), hkl[0], hkl[1], hkl[2])

        bragg_angles         = np.degrees(np.arcsin(0.5*beam_wavelengths/d_spacing)) #angulos de bragg para cada energia que chega no cristal  #graus
        
        perfect_incidence   =  (alpha_deg + bragg_angles)  #incidencia para reflec máxima para cada energia que chega no cristal #below  #graus #em relação à sup
        
        delta_thetas         = beam_incident_angles-perfect_incidence

        values = np.loadtxt(os.path.abspath(filename_txt) if filename_txt.startswith('/') else
                            os.path.abspath(os.path.curdir + "/" + filename_txt))
        
        crystal_delta_thetas     = np.degrees(values[:, 0]) #eixo x da RC

        crystal_reflectivities_s = values[:, 1] #eixo y da RC para polarizaçao s

        interpolated_weight_s = np.sqrt(np.interp(delta_thetas,
                                                        crystal_delta_thetas,
                                                        crystal_reflectivities_s,
                                                        left=crystal_reflectivities_s[0],
                                                        right=crystal_reflectivities_s[-1]))
            

        crystal_reflectivities_p = values[:, 2]
        interpolated_weight_p = np.sqrt(np.interp(delta_thetas,
                                                            crystal_delta_thetas,
                                                            crystal_reflectivities_p,
                                                            left=crystal_reflectivities_p[0],
                                                            right=crystal_reflectivities_p[-1]))

        output_beam = beam_after.duplicate()


        output_beam._beam.rays[:, 6]  = beam_before._beam.rays[:, 6]  * interpolated_weight_s
        output_beam._beam.rays[:, 7]  = beam_before._beam.rays[:, 7]  * interpolated_weight_s
        output_beam._beam.rays[:, 8]  = beam_before._beam.rays[:, 8]  * interpolated_weight_s
        output_beam._beam.rays[:, 15] = beam_before._beam.rays[:, 15] * interpolated_weight_p
        output_beam._beam.rays[:, 16] = beam_before._beam.rays[:, 16] * interpolated_weight_p
        output_beam._beam.rays[:, 17] = beam_before._beam.rays[:, 17] * interpolated_weight_p

        self.beam = output_beam._beam        
        
        if save_beam_name is not None: self.Save_Beam(save_beam_name = save_beam_name)
    
    def Save_Beam(self, save_beam_name: str = None):
        if hasattr(self, 'beam'):
            if save_beam_name is not None:
                self.beam.write(save_beam_name)
            else: print('No name to save beam!')
        else: print('No beam defined!')


