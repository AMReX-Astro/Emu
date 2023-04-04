import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp

base = "N01"
def make_plot(d):
    print(d)
    plt.clf()
    #plt.ylim(-1.5e40,1.5e40)
    ds = yt.load(d)
    t = ds.current_time
    ad = ds.all_data()
    Re = ad['boxlib',base+"_Re"]
    Im = ad['boxlib',base+"_Im"]
    mag = np.sqrt(Re**2+Im**2)
    plt.plot(Re,color="blue")
    plt.plot(Im,color="orange")
    Re = ad['boxlib',base+"_Rebar"]
    Im = ad['boxlib',base+"_Imbar"]
    mag = np.sqrt(Re**2+Im**2)
    plt.plot(Re,color="blue",linestyle="--")
    plt.plot(Im,color="orange",linestyle="--")
#    plt.plot(mag)
    plt.text(0,0,"t="+str(t)+" s")
    plt.savefig(base+"_"+d+".png")

directories = sorted(glob.glob("plt*"))

#pool = mp.Pool(mp.cpu_count())
#pool.map(make_plot, directories)
for d in directories:
    make_plot(d)

# ffmpeg -i %*.png -pix_fmt yuv420p movie.mp4
