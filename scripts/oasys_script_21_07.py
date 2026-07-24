#
# Python script to run shadow3. Created automatically with ShadowTools.make_python_script_from_list().
#
import Shadow
import numpy as np
from optlnls.plot import plot_beam

# write (1) or not (0) SHADOW files start.xx end.xx star.xx
iwrite = 0

#
# initialize shadow3 source (oe0) and beam
#
beam = Shadow.Beam()
B1 = Shadow.Source()
M1 = Shadow.OE()
DVF_A1 = Shadow.OE()
DVF_B1 = Shadow.OE()

#
# Define variables. See meaning of variables in: 
#  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml 
#  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
#

B1.BENER = 3.0
B1.EPSI_X = 2.5e-07
B1.EPSI_Z = 2.5e-09
B1.FDISTR = 4
B1.FSOURCE_DEPTH = 4
B1.F_COLOR = 3
B1.F_PHOT = 0
B1.HDIV1 = 0.0002
B1.HDIV2 = 0.0002
B1.ISTAR1 = 1
B1.NCOL = 0
B1.NPOINT = 500000
B1.N_COLOR = 0
B1.PH1 = 10999.0
B1.PH2 = 11001.0
B1.POL_DEG = 0.0
B1.R_ALADDIN = 17736.481488735484
B1.R_MAGNET = 17.736481488735485
B1.SIGDIX = 0.0
B1.SIGDIZ = 0.0
B1.SIGMAX = 0.0199
B1.SIGMAY = 0.0
B1.SIGMAZ = 0.0115
B1.VDIV1 = 1.0
B1.VDIV2 = 1.0
B1.WXSOU = 0.0
B1.WYSOU = 0.0
B1.WZSOU = 0.0

M1.ALPHA = 90.0
M1.DUMMY = 0.1
M1.FHIT_C = 1
M1.FMIRR = 3
M1.FWRITE = 1
M1.F_EXT = 1
M1.F_MOVE = 1
M1.OFFZ = 2.0
M1.RLEN1 = 110.0
M1.RLEN2 = 110.0
M1.RWIDX1 = 4.0
M1.RWIDX2 = 4.0
M1.R_MAJ = 901300.0
M1.R_MIN = 318.4
M1.T_IMAGE = 0.0
M1.T_INCIDENCE = 88.924
M1.T_REFLECTION = 88.924
M1.T_SOURCE = 17000.0

DVF_A1.DUMMY = 0.1
DVF_A1.FWRITE = 3
DVF_A1.F_REFRAC = 2
DVF_A1.F_SCREEN = 1
DVF_A1.I_SLIT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
DVF_A1.N_SCREEN = 1
DVF_A1.RX_SLIT = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DVF_A1.RZ_SLIT = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DVF_A1.T_IMAGE = 0.0
DVF_A1.T_INCIDENCE = 0.0
DVF_A1.T_REFLECTION = 180.0
DVF_A1.T_SOURCE = 4359.0

DVF_B1.DUMMY = 0.1
DVF_B1.FWRITE = 3
DVF_B1.F_REFRAC = 2
DVF_B1.F_SCREEN = 1
DVF_B1.N_SCREEN = 1
DVF_B1.T_IMAGE = 0.0
DVF_B1.T_INCIDENCE = 0.0
DVF_B1.T_REFLECTION = 180.0
DVF_B1.T_SOURCE = 12641.0

#Run SHADOW to create the source

if iwrite:
    B1.write("start.00")

beam.genSource(B1)

if iwrite:
    B1.write("end.00")
    beam.write("begin.dat")


#
#run optical element 1
#
print("    Running optical element: %d"%(1))
if iwrite:
    M1.write("start.01")

beam.traceOE(M1,1)

if iwrite:
    M1.write("end.01")
    beam.write("star.01")


#
#run optical element 2
#
print("    Running optical element: %d"%(2))
if iwrite:
    DVF_A1.write("start.02")

beam.traceOE(DVF_A1,2)

if iwrite:
    DVF_A1.write("end.02")
    beam.write("star.02")


#
#run optical element 3
#
print("    Running optical element: %d"%(3))
if iwrite:
    DVF_B1.write("start.03")

beam.traceOE(DVF_B1,3)

if iwrite:
    DVF_B1.write("end.03")
    beam.write("star.03")



Shadow.ShadowTools.plotxy(beam,3,1,nbins=201,nolost=1,title="Real space")
# Shadow.ShadowTools.plotxy(beam,1,4,nbins=101,nolost=1,title="Phase space X")
# Shadow.ShadowTools.plotxy(beam,3,6,nbins=101,nolost=1,title="Phase space Z")


focus = beam
image_ticket = beam.histo2(3, 1, 201, nolost=1)

x = image_ticket['bin_h_center']
y = image_ticket['bin_v_center']
img = image_ticket['histogram']

beam2D = np.zeros((len(y)+1, len(x)+1))
beam2D[0,1:] = x
beam2D[1:,0] = y
beam2D[1:,1:] = img.T


plot_beam(beam2D, cut=0, textA=1, textB=10, textC=9, textD=0, fitType=1,
          plot_title='CAX M1 nominal', show_colorbar=1,
            x_range=1, y_range=1, x_range_min=-0.12, x_range_max=0.12, y_range_min=-0.12, y_range_max=0.12,
          zero_pad_x=1, zero_pad_y=1, cmap='viridis', integral=1e11,
          units=2)

