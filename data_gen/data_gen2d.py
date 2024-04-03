from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
from pyeit.mesh import create, set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import random
import csv
import pandas as pd

if __name__ == "__main__":

    """
    This script generates noiseless 2D conductivity maps and surface potentials.

    """
    perm_csv_filename="perm_vals.csv"
    v1_csv_filename="v1_vals.csv"

    perm_vals=[]
    v1_vals=[]

    zero_tumor_record_count=2
    one_tumor_record_count=2
    two_tumor_record_count=2

    tumor_count={
        0:zero_tumor_record_count,
        1:one_tumor_record_count,
        2:two_tumor_record_count
    }

    tissue_perm=(4,6) #breast tissue
    # tissue_perm=(65,90) #brain
    # tissue_perm=(50,70) #muscle
    # tissue_perm=(2.5,5) #fat
    # tissue_perm=(20,40) #skin

    bknd_perm=1.0 # Setting this to a fixed value based on literature search
    tumor_perm=(1.5,10.0) # Need to get better values for this
    
    # Set this variable to True if you want to view the conductivity maps
    show_maps = True

    #number of electrodes
    n_el=16

    #create mesh object
    mesh_obj=create(n_el,h0=0.04)

    # extract node, element, perm from mesh_obj
    xx, yy = mesh_obj.node[:, 0], mesh_obj.node[:, 1]

    tri = mesh_obj.element

    perm=mesh_obj.perm

    for t_count,rec_count in tumor_count.items():
        for count in range(rec_count):

            tissue_perm_rand=random.uniform(min(tissue_perm),max(tissue_perm))
            tumor_perm_rand=random.uniform(min(tumor_perm),max(tumor_perm))

            if t_count ==0:
                invisible_tumor=PyEITAnomaly_Circle(center=[0,0], r=0,perm=bknd_perm)

                mesh_new=set_perm(mesh_obj,anomaly=invisible_tumor,background=bknd_perm)
            else:

                r1 = random.uniform(-0.7,0.7) #tumor1 x coord
                r2 = random.uniform(-0.7,0.7) # tumor1 y coord
                r3 = random.uniform(0.1,0.24) #tumor1 radius
                
                anomaly=[
                    PyEITAnomaly_Circle(center=[r1, r2], r=r3, perm=tumor_perm_rand)
                ]

                if t_count==2:
                    r4 = random.uniform(-0.8,0.8) #tumor2 x coord
                    r5 = random.uniform(-0.8,0.8) #tumor2 y coord
                    r6 = random.uniform(0.1,0.24) #tumor 2 radius

                    tumor_perm_rand_2=random.uniform(min(tumor_perm),max(tumor_perm))

                    anomaly.append(PyEITAnomaly_Circle(center=[r4, r5], r=r6, perm=tumor_perm_rand_2))

                mesh_new=set_perm(mesh_obj,anomaly=anomaly,background=bknd_perm)

            protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=2, parser_meas="std")
            fwd = EITForward(mesh_obj, protocol_obj)
            v1 = fwd.solve_eit(perm=mesh_new.perm)
            v1 = v1.tolist()
            v1.append(t_count)

            perm_vals.append(mesh_new.perm.tolist())
            v1_vals.append(v1)

            if show_maps:
                # plot
                fig, ax = plt.subplots(figsize=(9, 6))
                im = ax.tripcolor(xx, yy, tri, np.real(mesh_new.perm),  edgecolors="k", cmap="Reds", vmin=0, vmax=max(tumor_perm))

                # Plot electrode positions
                for el in mesh_obj.el_pos:
                    ax.plot(xx[el], yy[el], "ro")

                ax.axis("equal")
                ax.set_title(r"Conductivities")
                fig.colorbar(im)
                plt.show(block=True)

    # Write the data to the CSV file
    perm_df = pd.DataFrame(perm_vals)
    perm_df.to_csv(perm_csv_filename, index=False, header=False)

    v1_df = pd.DataFrame(v1_vals)
    v1_df.to_csv(v1_csv_filename, index=False, header=False)
    


        
