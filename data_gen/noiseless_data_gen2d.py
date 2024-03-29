from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.protocol as protocol
from pyeit.eit.fem import EITForward
from pyeit.mesh import create, set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
import random
import datetime
import csv

if __name__ == "__main__":

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

    tumor_perm=(1.2,3.5) #need to get better values for this

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
                invisible_tumor=PyEITAnomaly_Circle(center=[0,0], r=0,perm=tissue_perm_rand)

                mesh_new=set_perm(mesh_obj,anomaly=invisible_tumor,background=tissue_perm_rand)
            else:

                r1 = random.uniform(-0.7,0.7) #tumor1 x coord
                r2 = random.uniform(-0.7,0.7) # tumor1 y coord
                r3 = random.uniform(0.1,0.2) #tumor1 radius
                
                anomaly=[
                    PyEITAnomaly_Circle(center=[r1, r2], r=r3, perm=tumor_perm_rand)
                ]

                if t_count==2:
                    r4 = random.uniform(-0.8,0.8) #tumor2 x coord
                    r5 = random.uniform(-0.8,0.8) #tumor2 y coord
                    r6 = random.uniform(0.1,0.2) #tumor 2 radius

                    tumor_perm_rand_2=random.uniform(min(tumor_perm),max(tumor_perm))

                    anomaly.append(PyEITAnomaly_Circle(center=[r4, r5], r=r6, perm=tumor_perm_rand_2))

                mesh_new=set_perm(mesh_obj,anomaly=anomaly,background=tissue_perm_rand)

            protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=2, parser_meas="std")
            fwd = EITForward(mesh_obj, protocol_obj)
            v1 = fwd.solve_eit(perm=mesh_new.perm)

            perm_vals.append([mesh_new.perm.tolist()])
            v1_vals.append([t_count,v1])

            # plot
            fig, ax = plt.subplots(figsize=(9, 6))
            im = ax.tripcolor(xx, yy, tri, np.real(mesh_new.perm),  edgecolors="k", cmap="Reds", vmin=0, vmax=max(tissue_perm))

            # Plot electrode positions
            for el in mesh_obj.el_pos:
                ax.plot(xx[el], yy[el], "ro")

            ax.axis("equal")
            ax.set_title(r"Conductivities")
            fig.colorbar(im)
            plt.show(block=True)

    # Write the data to the CSV file
    with open(perm_csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(perm_vals)

    with open(v1_csv_filename,'w',newline='') as file:
        writer=csv.writer(file)
        writer.writerows(v1_vals)

    # print(f"Data has been written to '{csv_file}'")

        
