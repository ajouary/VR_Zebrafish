from IPython.html.widgets import interact

#import mpld3
#mpld3.enable_notebook()
from cmath import *
import scipy.io
import numpy as np
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt
#from pylab import *

import matplotlib.cm as cmx
import matplotlib.colors as colors
import seaborn as sns
sns.set(style="ticks")

import pickle



values = range(100)
RdYlBu = cm = plt.get_cmap('RdYlBu') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=RdYlBu)

Bl = scalarMap.to_rgba(values[-10])
Re = scalarMap.to_rgba(values[10])

rc={'font.size': 15, 'axes.labelsize': 15, 'legend.fontsize': 15.0, 
    'axes.titlesize': 15, 'xtick.labelsize':15, 'ytick.labelsize': 15}
plt.rcParams.update(**rc)
sns.set(rc=rc)

sns.set_style("whitegrid")
sns.set_style("whitegrid", {"legend.frameon": True})

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class LarvaSwimBout:
    def __init__(self,p_TailCurv,p_AvgCurv,p_HeadAngle,p_HeadPos,p_HeadVect,p_NumFrames,p_NumFilm,p_LarvaLen):#,p_DeltaSin,p_DeltaCos):
        self.TailCurv=p_TailCurv
        self.AvgCurv=p_AvgCurv
        self.HeadAngle=p_HeadAngle
        self.HeadPos=p_HeadPos
        self.HeadVect=p_HeadVect
        VMvt=np.hstack([np.zeros([2,1]),np.diff(self.HeadPos)])
        self.AxialMvt=np.zeros([VMvt.shape[1]])
        self.LateralMvt=np.zeros([VMvt.shape[1]])
        for i in range(0,VMvt.shape[1]):
            self.AxialMvt[i]=self.HeadVect[0,i]*VMvt[0,i]+self.HeadVect[1,i]*VMvt[1,i]
            self.LateralMvt[i]=self.HeadVect[0,i]*VMvt[1,i]-self.HeadVect[1,i]*VMvt[0,i]
        self.NumFrames=p_NumFrames
        self.NumFilm=p_NumFilm
        self.LarvaLen=p_LarvaLen
        
class Model:
    def __init__(self,p_ParaHeadAngleDiff,p_nxHeadAngleDiff,p_nyHeadAngleDiff,p_ParaLateralMvt,p_nxLateralMvt,p_nyLateralMvt,p_ParaAxialMvt,p_nxAxialMvt,p_nyAxialMvt):
        self.ParaHeadAngleDiff=p_ParaHeadAngleDiff
        self.nxHeadAngleDiff=p_nxHeadAngleDiff
        self.nyHeadAngleDiff=p_nyHeadAngleDiff
        self.ParaLateralMvt=p_ParaLateralMvt
        self.nxLateralMvt=p_nxLateralMvt
        self.nyLateralMvt=p_nyLateralMvt
        self.ParaAxialMvt=p_ParaAxialMvt
        self.nxAxialMvt=p_nxAxialMvt
        self.nyAxialMvt=p_nyAxialMvt



# 1 Parameter:

def ARXMat(x,nx,y,ny):
    K=len(x)
    if len(x)!=len(y):
        print('pb size Input Output missmatch')
    M=np.zeros([len(x),nx+ny])
    for i in range(0,len(x)):
        xside=np.array([])
        yside=np.array([])
        for k in range(0,0+nx):
            if i-k>=0:
                xside=np.insert(xside,len(xside),x[i-k]) 
            else:
                xside=np.insert(xside,len(xside),0) 
        for k in range(1,1+ny):
            if i-k>=0:
                yside=np.insert(yside,len(yside),-y[i-k]) 
            else:
                yside=np.insert(yside,len(yside),0) 
        M[i,:]=np.hstack([yside,xside])
    return M


def RegressARXWeight(M,y,W):
    a=np.linalg.inv(np.dot(np.transpose(M),np.dot(W,M)))
    b=np.dot(np.transpose(M),np.dot(W,y))
    Para=np.dot(a,b)
    return Para


def RegressARX(M,y):
    a=np.linalg.inv(np.dot(np.transpose(M),M))
    b=np.dot(np.transpose(M),y)
    Para=np.dot(a,b)
    return Para


# 1 Parameter

def Sim(x,Para,nx,ny,init):
    ySim=init*np.ones(x.shape)
    for i in range(0,len(x)):
        xside=np.array([])
        yside=np.array([])
        for k in range(0,0+nx):
            if i-k>=0:
                xside=np.insert(xside,len(xside),x[i-k]) 
            else:
                xside=np.insert(xside,len(xside),0) 
        for k in range(1,1+ny):
            if i-k>=0:
                yside=np.insert(yside,len(yside),-ySim[i-k]) 
            else:
                yside=np.insert(yside,len(yside),0) 
        ySim[i]=np.dot(Para,np.hstack([yside,xside]))
    return ySim

def GoodnessOfFit(x,y,Para,nx,ny):
    yrec=Sim(x,Para,nx,ny,0)
    SStot=np.sum((y-np.mean(y))**2)
    SSres=np.sum((y-yrec)**2)

    r2=1-SSres/SStot
    return r2

def rot(P,Angle,Center):
    #RotMat=np.array([[cos(Angle),-sin(Angle)],[sin(Angle),cos(Angle)]]).real
    #P=P-Center
    #return dot(RotMat,P)+Center
    PRot=np.zeros(P.shape)
    for i in range(0,P.shape[1]):
                PRot[0,i]=np.cos(Angle)*(P[0,i]-Center[0])-np.sin(Angle)*(P[1,i]-Center[1])+Center[0]
                PRot[1,i]=np.sin(Angle)*(P[0,i]-Center[0])+np.cos(Angle)*(P[1,i]-Center[1])+Center[1]
    return PRot
