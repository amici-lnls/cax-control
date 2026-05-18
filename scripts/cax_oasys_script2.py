#
# Python script to run shadow3. Created automatically with ShadowTools.make_python_script_from_list().
#
import Shadow
import numpy

# write (1) or not (0) SHADOW files start.xx end.xx star.xx
iwrite = 0

#
# initialize shadow3 source (oe0) and beam
#
beam = Shadow.Beam()
oe0 = Shadow.Source()
oe1 = Shadow.OE()
oe2 = Shadow.OE()
oe3 = Shadow.OE()
oe4 = Shadow.OE()

#
# Define variables. See meaning of variables in: 
#  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml 
#  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
#

oe0.BENER = 3.0
oe0.EPSI_X = 2.5e-07
oe0.EPSI_Z = 2.5e-09
oe0.FDISTR = 4
oe0.FSOURCE_DEPTH = 4
oe0.F_COLOR = 3
oe0.F_PHOT = 0
oe0.HDIV1 = 0.0002
oe0.HDIV2 = 0.0002
oe0.ISTAR1 = 1
oe0.NCOL = 0
oe0.NPOINT = 1000000
oe0.N_COLOR = 0
oe0.PH1 = 10999.0
oe0.PH2 = 11001.0
oe0.POL_DEG = 0.0
oe0.R_ALADDIN = 17736.481488735484
oe0.R_MAGNET = 17.736481488735485
oe0.SIGDIX = 0.0
oe0.SIGDIZ = 0.0
oe0.SIGMAX = 0.0199
oe0.SIGMAY = 0.0
oe0.SIGMAZ = 0.0115
oe0.VDIV1 = 1.0
oe0.VDIV2 = 1.0
oe0.WXSOU = 0.0
oe0.WYSOU = 0.0
oe0.WZSOU = 0.0

oe1.ALPHA = 90.0
oe1.DUMMY = 0.1
oe1.FHIT_C = 1
oe1.FMIRR = 3
oe1.FWRITE = 1
oe1.F_EXT = 1
oe1.F_MOVE = 1
oe1.RLEN1 = 110.0
oe1.RLEN2 = 110.0
oe1.RWIDX1 = 4.0
oe1.RWIDX2 = 4.0
oe1.R_MAJ = 901300.0
oe1.R_MIN = 318.4
oe1.T_IMAGE = 0.0
oe1.T_INCIDENCE = 88.924
oe1.T_REFLECTION = 88.924
oe1.T_SOURCE = 17000.0

oe2.DUMMY = 0.1
oe2.FWRITE = 3
oe2.F_REFRAC = 2
oe2.T_IMAGE = 0.0
oe2.T_INCIDENCE = 90.0
oe2.T_REFLECTION = 90.0
oe2.T_SOURCE = 4359.0

oe3.DUMMY = 0.1
oe3.FWRITE = 3
oe3.F_REFRAC = 2
oe3.F_SCREEN = 1
oe3.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
oe3.N_SCREEN = 1
oe3.RX_SLIT = numpy.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
oe3.RZ_SLIT = numpy.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
oe3.T_IMAGE = 0.0
oe3.T_INCIDENCE = 0.0
oe3.T_REFLECTION = 180.0
oe3.T_SOURCE = 0.0

oe4.ALPHA = 90.0
oe4.DUMMY = 0.1
oe4.FWRITE = 3
oe4.F_REFRAC = 2
oe4.T_IMAGE = 0.0
oe4.T_INCIDENCE = 90.0
oe4.T_REFLECTION = 90.0
oe4.T_SOURCE = 12641.0



#Run SHADOW to create the source

if iwrite:
    oe0.write("start.00")

beam.genSource(oe0)

if iwrite:
    oe0.write("end.00")
    beam.write("begin.dat")


#
#run optical element 1
#
print("    Running optical element: %d"%(1))
if iwrite:
    oe1.write("start.01")

beam.traceOE(oe1,1)

if iwrite:
    oe1.write("end.01")
    beam.write("star.01")


#
#run optical element 2
#
print("    Running optical element: %d"%(2))
if iwrite:
    oe2.write("start.02")

beam.traceOE(oe2,2)

if iwrite:
    oe2.write("end.02")
    beam.write("star.02")


#
#run optical element 3
#
print("    Running optical element: %d"%(3))
if iwrite:
    oe3.write("start.03")

beam.traceOE(oe3,3)

if iwrite:
    oe3.write("end.03")
    beam.write("star.03")


#
#run optical element 4
#
print("    Running optical element: %d"%(4))
if iwrite:
    oe4.write("start.04")

beam.traceOE(oe4,4)

if iwrite:
    oe4.write("end.04")
    beam.write("star.04")


Shadow.ShadowTools.plotxy(beam,1,3,nbins=101,nolost=1,title="Real space")
# Shadow.ShadowTools.plotxy(beam,1,4,nbins=101,nolost=1,title="Phase space X")
# Shadow.ShadowTools.plotxy(beam,3,6,nbins=101,nolost=1,title="Phase space Z")
    