import numpy as np
import matplotlib.pyplot as plt
import os

#in order to use my commons
import sys
sys.path.insert(0, 'C:\\Users\\skoci\\Documents\dust\\000_commons')

#neat plotting settings
import matplotlib as mpl
import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi']= 600

Csc = 3.55e-10                  #SolO capacitance [F]
G = 0.35                        #SolO gain factor [dimensionless]
q = 1.61                        #underestimation by Rackovic 2021 [dimensionless]
electron = 1.602e-19            #elementary charge [C]
mass = [5e-17]      #different representative dust grain masses [kg]
important = [1]             #just to identify the most likely mass


material = ["Al$^1$",
            "W", 
            "Al$^2$",
            "Au$^2$",
            "PCB", 
            "BeCu$^3$",
            "Kapton + Al",
            "Polyamide", 
            "Ag$^3$",
            "BeCu", 
            "Kapton + Ge", 
            "Sol. c.",
            "MLI$^3$",
            "TPS$^4$"]

A = np.array([7e-1,
              5.1e-1,
              1.4e-3,
              6.3e-4,
              4.7e-3,
              5e-2,
              1e-2,
              1.2e-1,
              8.9e-3,
              1.2e-2,
              2.5e-3,
              4.7e-3,
              1.7e-3,
              4.3e-2])

a = np.array([1.02]+13*[1])

b = np.array([3.48,
              3.5,
              4.8,
              5.6,
              4.1,
              3.9,
              4.6,
              3.3,
              3.9,
              3.8,
              4.5,
              4.2,
              4.7,
              3.46])

interval = np.array(([2,40],
                    [2,40],
                    [8,46],
                    [9,51],
                    [3,36],
                    [3,40],
                    [3,40],
                    [3,45],
                    [2,40],
                    [2,30],
                    [2,40],
                    [2,40],
                    [2,40],
                    [16,40]))

importance = np.array([1,
                      0,
                      1,
                      1,
                      0,
                      1,
                      0,
                      0,
                      1,
                      0,
                      0,
                      1,
                      1,
                      2])

color = ["lightslategray",
         "cyan",
         "dimgray",
         "gold",
         "lightcoral",
         "red",
         "pink",
         "paleturquoise",
         "lightsalmon",
         "darkseagreen",
         "deepskyblue",
         "mediumblue",
         "green",
         "black"]



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for j in range(len(mass)):
    m = mass[j]
    for i in range(len(material)):
        v = np.linspace(interval[i,0],interval[i,1],100)
        y = A[i] * (m)**a[i] * v**b[i] / electron   #elementary charge, constant, 1-micrometer, km/s
        if importance[i] and important[j]:
            linewidth = 1*importance[i]
            extrapolated_linewidth = 1*importance[i]
            lbl=material[i]
        else:
            linewidth=0.5
            extrapolated_linewidth = 0.5
            lbl='_nolegend_'
        ax1.loglog(v,y,lw=linewidth,label=lbl,c=color[i])
        if importance[i]:
            v = np.linspace(5,100,100)
            y = A[i] * (m)**a[i] * v**b[i] / electron   #elementary charge, constant, 1-micrometer, km/s
            ax1.loglog(v,y,lw=extrapolated_linewidth,label='_nolegend_',c=color[i],ls="dotted")
ax1.legend(fontsize="x-small",ncol=2)
ax2.set_yscale("log")
ax1.set_xlim([9,90])
ax1.set_xticks(np.append([10,15,30,50],np.arange(20, 81, step=20)))
ax1.set_xticklabels(np.append([10,15,30,50],np.arange(20, 81, step=20)))
ax1.set_ylim([5e4,2e9])
ax2.set_ylim(np.array(ax1.get_ylim())*1e3*electron*G/Csc/q)
ax1.set_xlabel("Velocity $[km/s]$")
ax1.set_ylabel("Yeild $[e]$")
ax2.set_ylabel(r"$\approx$ SolO Voltage $[mV]$")
ax1.set_title(r"$5 \cdot 10^{-17} \, kg$ Fe dust, Accelerator Results")
#hline1 = ax1.axhline(y=2.6e7,color="gold")
#hline2 = ax1.axhline(y=7e8,color="gold")
#vline1 = ax1.axvline(x=50,color="gold",zorder=-1)
#vline2 = ax1.axvline(x=100,color="gold",zorder=-1)
fig.show()





show = np.array([0,2,3,5,12,13])
styles = ["dotted","dotted","dashed","dashed","solid","solid"]
colors = ["grey","k","k","grey","grey","k"]
widths = [1,1,1,1,1,2]

fig, ax1 = plt.subplots()
for j in range(len(mass)):
    m = 1e-14 #kg
    for i, index in enumerate(show):
        v = np.linspace(interval[index,0],interval[index,1],100)
        y = 1e14 * A[index] * (m)**a[index] * v**b[index]
        lbl=material[index]
        ax1.loglog(v,y,label=lbl,ls=styles[i],color=colors[i],lw=widths[i])
ax1.legend(fontsize="small",ncol=2)
ax1.set_xlim([9,55])
ax1.set_xticks(np.append([10,15,30],np.arange(20, 41, step=20)))
ax1.set_xticklabels(np.append([10,15,30],np.arange(20, 41, step=20)))
ax1.set_ylim([1e2,5e6])
ax1.set_xlabel("Velocity $[km/s]$")
ax1.set_ylabel("Yeild $[C/kg]$")
fig.tight_layout()
plt.savefig("..\\998_generated\\charge_yield.pdf",format="pdf")
fig.show()
    
