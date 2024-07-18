import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib as mpl
from matplotlib import pyplot as plt

import figure_standards as figstd
axes_size = figstd.set_rcparams_dynamo(mpl.rcParams, num_cols=1, ls='thin')
mpl.rcParams['figure.dpi'] = 600

from paths import psp_model_location
from paths import solo_model_location
from paths import solo_model_nopanels_location

from paths import figures_location

def projection(file,
                elev=0,
                azim=0,
                roll=0,
                show=False,
                dpi=5000,
                distinct_heat_shield=False):
    """
    A function for getting the projection area of PSP when seen 
    from an aribtrary angle.

    Parameters
    ----------
    elev : float, optional
        Elevation angle of the view in degrees. The default is 0.
    azim : float, optional
        Azimuthal angle of the view in degrees. The default is 0.
    roll : float, optional
        Roll angle of the view in degrees. The default is 0.
    show : bool, optional
        Whether to show the plot for visual confirmation. 
        The default is False.
    dpi : int, optional
        The resolution of the render. The default is 2000.

    Returns
    -------
    projection_area : float
        The area of PSP as seen from the angle on input.

    """

    # Create a new plot
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(projection='3d',computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim, roll=roll)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(file)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors,
                                                       facecolors=r"#999999",
                                                       edgecolors="none",
                                                       antialiased=False))
    # Add a unit sphere.
    u = np.linspace(0, 2 * np.pi, 1000)
    v = np.linspace(0, np.pi, 1000)
    x = 5 + 1/np.sqrt(np.pi) * np.outer(np.cos(u), np.sin(v))
    y = -4 + 1/np.sqrt(np.pi) * np.outer(np.sin(u), np.sin(v))
    z = -4 + 1/np.sqrt(np.pi) * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=r"#000000", antialiased=False)

    #shield
    if distinct_heat_shield:
        if -90 < elev < 90 and 0 < azim < 180:
            mask = np.zeros(0,dtype=bool)
            for i in range(len(your_mesh.vectors[:,0,0])):
                face = your_mesh.vectors[i,:,:]
                include = np.any(face[:,1]>=1.480)
                mask = np.append(mask,include)
            shield = your_mesh.vectors[mask,:,:]
            ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
                shield,
                facecolors=r"#cccccc",
                edgecolors="none",
                antialiased=False))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Show the plot to the screen
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Now we can save it to a numpy array.
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    if show:
        plt.show()
    plt.close()

    # Count the PSP pixels.
    unique, counts = np.unique(data, return_counts=True)
    projection_area = counts[1]/counts[0]

    return projection_area


def illustrate_projections(psp_file,
                           solo_file,
                           solo_file_nopanels,
                           axspan=7,
                           save=False):

    tx, ty = 0.35, 0.75
    txt = "${area:.2f} m^2$"
    psp_mesh = mesh.Mesh.from_file(psp_file)
    solo_mesh = mesh.Mesh.from_file(solo_file)

    fig = plt.figure(figsize=(6, 4))
    gs = fig.add_gridspec(nrows = 2, ncols = 3, hspace=.05, wspace=.05)
    ax = gs.subplots(subplot_kw=dict(projection='3d'))

    for a in ax.reshape(-1):
        a.set_xlim(-axspan,axspan)
        a.set_ylim(-axspan,axspan)
        a.set_zlim(-axspan,axspan)
        a.set_proj_type('ortho')
        a.set_aspect('equal')
        a.axis('off')


    # PSP front
    a=ax[0][0]
    a.set_title("PSP - frontal view")
    a.view_init(elev=0, azim=90, roll=0)
    a.text2D(tx, ty, txt.format(area=projection(psp_file,
                                                elev=0,
                                                azim=90,
                                                roll=0)),
             transform=a.transAxes, va="center", ha="right")
    a.text2D(tx, ty-0.1, "("+txt.format(area=projection(psp_file,
                                                elev=0,
                                                azim=90,
                                                roll=0,
                                                distinct_heat_shield=True
                                                        ))+")",
             transform=a.transAxes, va="center", ha="right")
    a.add_collection3d(mplot3d.art3d.Poly3DCollection(psp_mesh.vectors,
                                                      facecolors=r"#999999",
                                                      edgecolors="none",
                                                      antialiased=False))
    # PSP side
    a=ax[0][1]
    a.set_title("PSP - lateral view")
    a.view_init(elev=0, azim=0, roll=0)
    a.text2D(tx, ty, txt.format(area=projection(psp_file,
                                                elev=0,
                                                azim=0,
                                                roll=0)),
             transform=a.transAxes, va="center", ha="right")
    a.add_collection3d(mplot3d.art3d.Poly3DCollection(psp_mesh.vectors,
                                                      facecolors=r"#999999",
                                                      edgecolors="none",
                                                      antialiased=False))
    # PSP top
    a=ax[0][2]
    a.set_title("PSP - top view")
    a.view_init(elev=90, azim=0, roll=0)
    a.text2D(tx, ty, txt.format(area=projection(psp_file,
                                                elev=90,
                                                azim=0,
                                                roll=0)),
             transform=a.transAxes, va="center", ha="right")
    a.add_collection3d(mplot3d.art3d.Poly3DCollection(psp_mesh.vectors,
                                                      facecolors=r"#999999",
                                                      edgecolors="none",
                                                      antialiased=False))
    # SolO front
    a=ax[1][0]
    a.set_title("SolO - frontal view")
    a.view_init(elev=0, azim=0, roll=180)
    a.text2D(tx, ty, txt.format(area=projection(solo_file,
                                                elev=0,
                                                azim=0,
                                                roll=180)),
             transform=a.transAxes, va="center", ha="right")
    a.text2D(tx, ty-0.1, "("+txt.format(area=projection(solo_file_nopanels,
                                                elev=0,
                                                azim=0,
                                                roll=180))+")",
             transform=a.transAxes, va="center", ha="right")
    a.add_collection3d(mplot3d.art3d.Poly3DCollection(solo_mesh.vectors,
                                                      facecolors=r"#999999",
                                                      edgecolors="none",
                                                      antialiased=False))
    # SolO side
    a=ax[1][1]
    a.set_title("SolO - lateral view")
    a.view_init(elev=180, azim=-90, roll=0)
    a.text2D(tx, ty, txt.format(area=projection(solo_file,
                                                elev=180,
                                                azim=-90,
                                                roll=0)),
             transform=a.transAxes, va="center", ha="right")
    a.add_collection3d(mplot3d.art3d.Poly3DCollection(solo_mesh.vectors,
                                                      facecolors=r"#999999",
                                                      edgecolors="none",
                                                      antialiased=False))
    # SolO top
    a=ax[1][2]
    a.set_title("SolO - top view")
    a.view_init(elev=90, azim=0, roll=90)
    a.text2D(tx, ty, txt.format(area=projection(solo_file,
                                                elev=90,
                                                azim=0,
                                                roll=90)),
             transform=a.transAxes, va="center", ha="right")
    a.add_collection3d(mplot3d.art3d.Poly3DCollection(solo_mesh.vectors,
                                                      facecolors=r"#999999",
                                                      edgecolors="none",
                                                      antialiased=False))
    if save:
        fig.savefig(figures_location+"projections.png",dpi=1200)
    fig.show()


#%%
if __name__ == "__main__":
    #print(psp_projection(solo_model_location,
    #                    90,0,0,1,distinct_heat_shield=False))

    illustrate_projections(psp_model_location,
                           solo_model_location,
                           solo_model_nopanels_location,
                           save=True)

