#header
pdbSet Grid Adaptive 1
pdbSet Grid SnMesh UseLines 1
pdbSet Grid AdaptiveField Refine.Rel.Error 0.75
mgoals on
pdbSet Diffuse minT 600
AdvancedCalibration

#Calibration_Set


math coord.ucs
math numThreads=50
pdbSet MCImplant Smoothing 1 
#endheader

#---------------------Mesh----------------------#
fset InitialSpace 10      
fset oxbotspace 0.05
fset PolySpace 0.01
fset MinNormalSize 0.005
fset Xj 0.05

grid set.min.normal.size= 0.01 set.normal.growth.ratio.2d= 1.618 set.min.edge= 5e-5
mgoals accuracy= 2e-5 
refinebox interface.materials= {Silicon Oxide PolySilicon}

#---------------------Read Mask----------------------#
#if [string match @Domain@ "HBT"]

set xmin 174984.8
set ymin 1224597.8
set xmax 174984.8
set ymax 1233862.5

#endif

icwb gds.file= @pwd@/012_ADC_BBM_HBT_LOTA.gds cell= TOP_BBM_HBT \
layer.numbers= {14:1 20:1 31:1 31:2 32:1 35:1 36:1 37:1 38:1 57:1 49:1 58:1 59:1 92:1 100:1 121:1 23:1 208:3 40:1} \
layer.names= {DTI MAA NBL NBL2 BCT SPC LOC HG DG EMSA EMW EMP BSP PPL MCT M1 PWCO BORDER MGT}\
stretches= {HBT = {173500.0 1230000.0 176500.0 1230000.0} } \
sim2d= "$xmin $ymin $xmax $ymax" scale= 1e-3


icwb stretch name= "@stretch@" value= @stretchlength@

icwb.create.all.masks


#rem # Query utility: Returns the bounding box of the simulation
fset Ymin [icwb bbox left ]
fset Ymax [icwb bbox right]
fset Zmin [icwb bbox back ]
fset Zmax [icwb bbox front]
LogFile "icwb: Centered Bounding Box -> $Ymin $Ymax $Zmin $Zmax"


fset LXmin [icwb bbox xmin]
fset LXmax [icwb bbox xmax]
fset LYmin [icwb bbox ymin]
fset LYmax [icwb bbox ymax]
LogFile "icwb: Layout Bounding Box -> $LXmin $LXmax $LYmin $LYmax"


fset DIM [icwb dimension]

line x location= 0.0 spacing= 100.0 tag= top
line x location= 5.0 spacing= 100.0 tag= bot

set Ydim ""
set Zdim ""

if {$DIM == 2} {
  line y location= $Ymin   spacing= 100.0 tag= left
  line y location= $Ymax   spacing= 100.0 tag= right
  set Ydim "ylo= left yhi= right"
  set Zdim ""
}  

if {$DIM == 3} {
  line y location= $Ymin   spacing= 100.0 tag= left
  line y location= $Ymax   spacing= 100.0 tag= right
  line z location= $Zmin   spacing= 100.0 tag= front
  line z location= $Zmax   spacing= 100.0 tag= back
  set Ydim "ylo= left yhi= right"
  set Zdim "zlo= front zhi= back"
}



eval region silicon xlo=top xhi= bot $Ydim $Zdim
init silicon field= Boron concentration= 1e15  wafer.orient = { 0 0 1 } flat.orient = { 1 1 0 }


refinebox name= BG \
      min = " -5.0 $Ymin $Zmin" max = "10.0 $Ymax $Zmax" \
      def.max.dose.error= 1.0E8 \
      refine.min.edge= "0.15 0.15 0.15" \
      refine.max.edge= {0.75 0.75 0.75} \
      Adaptive add



#split @NBL@ 
#--- Burried Implants

deposit Photoresist thickness= 15.0
etch Photoresist mask= NBL2_n thickness= 15.5 anisotropic
implant Arsenic  dose= 8e15<cm-2> energy= 50<keV> tilt= 0 rotation= 0
implant Arsenic  dose= 7e15<cm-2> energy= 50<keV> tilt= 0 rotation= 0
implant Arsenic  dose= 5e15<cm-2> energy= 50<keV> tilt= 0 rotation= 0
strip Photoresist

#-----  change to diffuse  -----
#rem # Drive-in #------diffuse stage NBL2-ANN
deposit material= Oxide type= isotropic rate= 1.0 time= 0.2
temp_ramp  name=T_F0030DRV1100C time= 46.15384  temp= 600  t.final= 940 
temp_ramp  name=T_F0030DRV1100C time= 64.61538  temp= 940  t.final= 1100
temp_ramp  name=T_F0030DRV1100C time= 182.76923  temp= 1100  t.final= 1100
temp_ramp  name=T_F0030DRV1100C time= 212.76923  temp= 1100  t.final= 600
diffuse temp_ramp=T_F0030DRV1100C 
strip Oxide


#split @SI_EPI@ 
#----- Epi Layer

deposit Silicon thickness= 0.8
#------diffuse stage EPI-DEP
temp_ramp  name=SI950U0D8V01 time= 0.12833  temp= 645.50000  ramprate= 0.00000 
temp_ramp  name=SI950U0D8V01 time= 1.00500  temp= 645.50000  ramprate= 2.91045 
temp_ramp  name=SI950U0D8V01 time= 0.82167  temp= 821.00000  ramprate= 4.88235 
temp_ramp  name=SI950U0D8V01 time= 1.94333  temp= 1061.70000  ramprate= -0.02487 
temp_ramp  name=SI950U0D8V01 time= 1.42333  temp= 1058.80000  ramprate= -1.26932 
temp_ramp  name=SI950U0D8V01 time= 4.13833  temp= 950.40000  ramprate= -0.00121 
temp_ramp  name=SI950U0D8V01 time= 1.52500  temp= 950.10000  ramprate= -3.39672 
diffuse temp_ramp=SI950U0D8V01 


#split @STI@ 

#----  STI -----

deposit material = Oxide type = isotropic  rate = 1.0 time=0.011
#------diffuse stage MAA-BKOX
temp_ramp  name=006EPOX398C time= 65.00000  temp= 700.30000  ramprate= 0.00000 
temp_ramp  name=006EPOX398C time= 19.00000  temp= 700.30000  ramprate= 0.18982 
temp_ramp  name=006EPOX398C time= 39.00000  temp= 916.70000  ramprate= 0.00073 
temp_ramp  name=006EPOX398C time= 80.00000  temp= 918.40000  ramprate= -0.04527 
temp_ramp  name=006EPOX398C time= 45.00000  temp= 701.10000  ramprate= 0.00000 
diffuse temp_ramp=006EPOX398C 

deposit material = Nitride type = isotropic rate = 1.0 time=0.078

deposit Photoresist thickness= 15.0
etch Photoresist mask= MAA_p thickness= 16.0 anisotropic
etch Nitride thickness= 0.08<um> type= trapezoidal angle= 90.0
etch Oxide  thickness= 0.015<um> type= trapezoidal angle= 90.0
etch Silicon thickness= 0.36<um> type= trapezoidal angle= 80.0
strip Photoresist

#----- pu Back  -----
etch material = {Nitride} type=isotropic rate = {0.01}  time=1 

#-----  change to diffuse  -----
deposit material = Oxide type = isotropic selective.materials=Silicon rate = 1.0 time=0.0128
#------diffuse stage STI-LOX
temp_ramp  name=RTO1100A125 time= 0.21667  temp= 600.70000  ramprate= 38.40000 
temp_ramp  name=RTO1100A125 time= 1.35000  temp= 1099.90000  ramprate= 0.01605 
temp_ramp  name=RTO1100A125 time= 0.45000  temp= 1101.20000  ramprate= -18.93333 
diffuse temp_ramp=RTO1100A125 

#------diffuse stage STI-LOX
temp_ramp  name=0078ANN3B6C time= 17.99451  temp= 700.50000  ramprate= 0.18200 
temp_ramp  name=0078ANN3B6C time= 18.59712  temp= 897.00000  ramprate= 0.04633 
temp_ramp  name=0078ANN3B6C time= 125.00000  temp= 948.70000  ramprate= 0.00011 
temp_ramp  name=0078ANN3B6C time= 80.06472  temp= 949.50000  ramprate= -0.05150 
diffuse temp_ramp=0078ANN3B6C 


deposit material = Oxide type = isotropic  rate = 1.0 time=0.74
#------diffuse stage MD-FUR-HTP
temp_ramp  name=001EANN41AC time= 16.21000  temp= 699.80000  ramprate= 0.20687 
temp_ramp  name=001EANN41AC time= 3.00000  temp= 901.00000  ramprate= 0.00000 
temp_ramp  name=001EANN41AC time= 13.00000  temp= 901.00000  ramprate= 0.12513 
temp_ramp  name=001EANN41AC time= 12.00000  temp= 998.60000  ramprate= 0.06944 
temp_ramp  name=001EANN41AC time= 46.00000  temp= 1048.60000  ramprate= 0.00033 
temp_ramp  name=001EANN41AC time= 185.74000  temp= 1049.50000  ramprate= -0.03133 
diffuse temp_ramp=001EANN41AC 

struct tdr= n@node@_stiDEP
etch material = Oxide type= cmp coord= -(0.8+0.011+0.068)<um> 

#split @DTI@ 
deposit Photoresist thickness= 15.0
etch Photoresist mask= DTI_n thickness= 16.0 anisotropic

etch Oxide  thickness= 0.5<um> type= trapezoidal angle= 90.0
etch Silicon thickness= 4.0<um> type= trapezoidal angle= 88.0
implant Boron dose= 5e12<cm-2> energy= 15.0<keV> tilt= 0 rotation= 0
strip Photoresist

deposit material = Oxide type = isotropic selective.materials=Silicon rate = 1.0 time=0.07

deposit material = Poly type = isotropic  rate = 1.0 time=0.84

etch material = {Poly} type=anisotropic rate = {1.2} time=1

deposit material = Oxide type = isotropic  rate = 1.0 time=0.8
temp_ramp  name=001EANN41AC time= 16.21000  temp= 699.80000  ramprate= 0.20687 
temp_ramp  name=001EANN41AC time= 3.00000  temp= 901.00000  ramprate= 0.00000 
temp_ramp  name=001EANN41AC time= 13.00000  temp= 901.00000  ramprate= 0.12513 
temp_ramp  name=001EANN41AC time= 12.00000  temp= 998.60000  ramprate= 0.06944 
temp_ramp  name=001EANN41AC time= 46.00000  temp= 1048.60000  ramprate= 0.00033 
temp_ramp  name=001EANN41AC time= 185.74000  temp= 1049.50000  ramprate= -0.03133 
diffuse temp_ramp=001EANN41AC 


etch material = Oxide type= cmp coord= -(0.8+0.011+0.068)<um> 

#----- STI OX wet ETCH  -----
etch material = {Oxide} type=isotropic rate = 1.0  time=0.008

etch material = {Nitride} type=isotropic rate = 1.0  time=0.08
# Reflect
# -------

#split @BCT@

refinebox name= HBT min= "-0.8 3.35" max= "0 5.8" \
           xrefine= "0.04" yrefine= "0.08" add all
grid    remesh

deposit Photoresist thickness= 15.0
etch Photoresist mask= BCT_n thickness= 16.0 anisotropic
implant Phosphorus dose= 8e15<cm-2> energy= 40<keV> tilt= 0 rotation= 0
strip Photoresist

deposit Photoresist thickness= 15.0
etch Photoresist mask= PWCO_n thickness= 16.0 anisotropic
implant Boron dose= 1e13<cm-2> energy= 280.0<keV> tilt= 0 rotation= 22
implant Boron dose= 6.5e12<cm-2> energy= 180.0<keV> tilt= 0 rotation= 22
implant Boron dose= 6e12<cm-2> energy= 100.0<keV> tilt= 0 rotation= 22
implant Boron dose= 7e12<cm-2> energy= 20.0<keV> tilt= 0 rotation= 22
strip Photoresist

temp_ramp  name=RTA1050S030 time= 0.28333  temp= 499.90000  ramprate= 32.24706 
temp_ramp  name=RTA1050S030 time= 0.41667  temp= 1048.10000  ramprate= 0.07600 
temp_ramp  name=RTA1050S030 time= 0.16667  temp= 1050.00000  ramprate= -25.01000 
diffuse temp_ramp=RTA1050S030 


etch material = {Oxide} type=isotropic rate = 1.0  time=0.012

#split @GOX@
gas_flow name=GF_GOX1_SO2 flowO2=0.2 flowN2=15
gas_flow name=GF_GOX1_Main flowO2=2 flowN2=16 flowH2=2

temp_ramp  name=TR_GOX1_1  time= 5  temp= 750  ramprate= 0
diffuse temp_ramp=TR_GOX1_1
temp_ramp  name=TR_GOX1_2  time= 10  temp= 750  ramprate= 0.0833333333333333

temp_ramp  name=TR_GOX1_2  time= 10  temp= 800  ramprate= 0
diffuse temp_ramp=TR_GOX1_2 gas_flow=GF_GOX1_SO2

temp_ramp  name=TR_GOX1_3  time= 10  temp= 800  ramprate= 0
diffuse temp_ramp=TR_GOX1_3
temp_ramp  name=TR_GOX1_4  time= 27.0333333333333  temp= 800  ramprate= 0
diffuse temp_ramp=TR_GOX1_4 gas_flow= GF_GOX1_Main
temp_ramp  name=TR_GOX1_5  time= 5  temp= 800  ramprate= 0
temp_ramp  name=TR_GOX1_5  time= 10  temp= 800  ramprate= 0
temp_ramp  name=TR_GOX1_5  time= 25  temp= 800  ramprate= -0.0333333333333333
temp_ramp  name=TR_GOX1_5  time= 30  temp= 750  ramprate= 0
diffuse temp_ramp=TR_GOX1_5

temp_ramp  name=GSP1080A028T time= 0.36667  temp= 475.79000  ramprate= 27.27773 
temp_ramp  name=GSP1080A028T time= 0.93333  temp= 1075.90000  ramprate= 0.03214 
temp_ramp  name=GSP1080A028T time= 1.22000  temp= 1077.70000  ramprate= -8.81967 
temp_ramp  name=GSP1080A028T time= 0.16667  temp= 432.10000  ramprate= 16.79000 
temp_ramp  name=GSP1080A028T time= 0.30000  temp= 600.00000  ramprate= 27.70556 
temp_ramp  name=GSP1080A028T time= 0.90000  temp= 1098.70000  ramprate= -0.00370 
temp_ramp  name=GSP1080A028T time= 0.63333  temp= 1098.50000  ramprate= -15.83158 
diffuse temp_ramp=GSP1080A028T 

#split @LDD_SPC_LOC@

refinebox name= HBTCollector min= "-0.8 5.1" max= "-0.3 5.72" \
           xrefine= "0.02" yrefine= "0.04" add all
grid    remesh

deposit material = Oxide type = isotropic  rate = 1.0 time=0.02
deposit material = Nitride type = isotropic rate = 1.0 time=0.06

deposit Photoresist thickness= 15.0
etch Photoresist mask= SPC_n thickness= 16.0 anisotropic
etch Nitride thickness= 0.08<um> type= trapezoidal angle= 90.0
strip Photoresist


deposit Photoresist thickness= 15.0
etch Photoresist mask= LOC_n thickness= 16.0 anisotropic
implant Arsenic  dose= 7e12<cm-2> energy= 14<keV> tilt= 0 rotation= 0
implant Arsenic  dose= 2e13<cm-2> energy= 80<keV> tilt= 0 rotation= 0
strip Photoresist

temp_ramp  name=SPK0950R180 time= 0.04820  temp= 500.10000  ramprate= 138.52006 
temp_ramp  name=SPK0950R180 time= 0.01030  temp= 900.70000  ramprate= 82.20065 
temp_ramp  name=SPK0950R180 time= 0.01820  temp= 951.50000  ramprate= -46.24542 
temp_ramp  name=SPK0950R180 time= 0.16730  temp= 901.00000  ramprate= -39.86850 
diffuse temp_ramp=SPK0950R180 

etch material = {Nitride} type=isotropic rate = 1.0  time=0.015
etch material = {Oxide} type=isotropic rate = 1.0  time=0.075

#split @SiGe_EPI@

#----------------------------------------------------------------------#
refinebox name= Base min= "-0.9 5.0" max= "-0.75 5.8" \
           xrefine= "0.005" yrefine= "0.025" add all
refinebox name= GeLayer min= "-0.99 5.1" max= "-0.8 5.71" \
           xrefine= "0.00325" yrefine= "0.025" add all

strain_profile species= Germanium silicon ratio= {0 1} strain= {0 0.0425}

pdbSetBoolean Silicon Mechanics UpdateStrain 1
#----------------------------------------------------------------------#
grid     remesh

doping name= strainB1 field= Boron depths= {0 0.02925} values= {1e12 5e19}
doping name= strainC1 field= Carbon depths= {0 0.02925} values= {1e11 3.4e19} 


doping name= strainGe field= Germanium depths= {0 0.01841} values= {1e22 2.8e21}
doping name= strainB field= Boron depths= {0 0.01841} values= {5e19 5e19}
doping name= strainC field= Carbon depths= {0 0.01841} values= {3.4e19 3.4e19} 

deposit material= silicon doping= {strainB1 strainC1}  type= isotropic thickness= 0.02925

deposit material= silicon doping= {strainB strainGe strainC} type= isotropic thickness= 0.01841

deposit material= silicon  type= isotropic thickness= 0.00975

diffuse temperature= 850<C> time= 10.0<s> 

#split @SAP@

#----------------------------------------------------------------------#
deposit material = Oxide type = isotropic  rate = 1.0 time=0.0078
deposit material = Poly  type = isotropic  rate = 1.0 time=0.12

deposit Photoresist thickness= 15.0
etch Photoresist mask= EMSA_p thickness= 16.0 anisotropic
etch poly thickness= 0.3<um> type= trapezoidal angle= 90.0
strip Photoresist

struct tdr= n@node@_SAPoly

deposit material = Oxide type = isotropic  rate = 1.0 time=0.05

etch material = Oxide type = anisotropic rate = 1.0 time=0.05

struct tdr= n@node@_SAPoly_OX1

deposit material = Oxide type = isotropic  rate = 1.0 time=0.1

etch material = Oxide type = anisotropic rate = 1.0 time=0.09

implant Boron dose= 5e15<cm-2> energy= @SAP@<keV> tilt= 0 rotation= 0

struct tdr= n@node@_SAPoly_OX2

deposit material = Oxide type = isotropic  rate = 1.0 time=0.05

struct tdr= n@node@_SAPoly_OX3

etch material = Oxide type = anisotropic rate = 1.0 time=0.02
deposit Photoresist thickness= 5.0
etch Photoresist mask= EMW_n thickness= 6.0 anisotropic
etch Oxide thickness= 0.1<um> type= trapezoidal angle= 90.0
struct tdr= n@node@_SAPoly_OXETCH
etch Poly thickness= 0.15<um> type= trapezoidal angle= 90.0
strip Photoresist

etch material = {Oxide} type=isotropic rate = 1.0  time=0.0085

#split @EMP@
doping name= EMPAs field= Arsenic depths= {0 0.080} values= {4e19 8e19}
deposit  material= {Silicon} doping= {EMPAs} type= isotropic rate= {1.0} time= 0.080

deposit Photoresist thickness= 15.0
etch Photoresist mask= EMP_p thickness= 16.0 anisotropic
etch Silicon thickness= 0.13<um> type= trapezoidal angle= 90.0
etch Oxide thickness= 0.03<um> type= trapezoidal angle= 90.0
strip Photoresist

etch material = {Oxide} type=isotropic rate = 1.0  time=0.04


#split @BSP@

deposit Photoresist thickness= 15.0
etch Photoresist mask= BSP_p thickness= 16.0 anisotropic
etch Silicon thickness= 0.3<um> type= trapezoidal angle= 90.0
etch material = {Nitride} type=anisotropic rate = {1} time=0.09
etch material = {oxide} type=anisotropic rate = {1} time=0.03
strip Photoresist

temp_ramp  name=SPK1055R075 time= 0.08050  temp= 650.70000  ramprate= 74.05797 
temp_ramp  name=SPK1055R075 time= 0.01383  temp= 1008.40000  ramprate= 60.00000 
temp_ramp  name=SPK1055R075 time= 0.01447  temp= 1058.20000  ramprate= -57.37327 
temp_ramp  name=SPK1055R075 time= 0.08237  temp= 1008.40000  ramprate= -59.93525 
diffuse temp_ramp=SPK1055R075 



struct tdr= n@node@_HBTprocess ; # Final

# save final structure:
#  - 1D cross sections


SetPlxList   {AsTotal BTotal GeTotal CTotal NetActive}
WritePlx n@node@_HBT_2.plx y=5.4 Silicon

exit


