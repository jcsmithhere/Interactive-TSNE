import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
import sys, os
from argparse import Namespace
import numpy as np

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
from io import BytesIO
import traceback, functools
import pandas as pd

errorlist = []

# save error message in the global errorlist variable
def save_errors(fun):
    @functools.wraps(fun)
    def wrapper(*args,**kw):
        try:
            return fun(*args,**kw)
        except Exception:
            errorlist.append(traceback.format_exc())
    return wrapper

class InteractivePlot:
    """Standard version selects random sample of objects from cluster"""
    def __init__(self, clusters, tsne_results, image_mapping, spd=None, nside=4,
                 filename=None, highlight=None, highlightpars=None, colors=None):
        """
        Parametrs
        ---------
        clusters : (int array) 
            list of clusters to display in the t-SNE representation
        tsne_results : (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        image_mapping (pandas.DataFrame)  
            table with information on the images
            ['name', 'filename'] pairs for each image 
        
        Optional keyword parameters:
        ----------------------------
        spd is a square array [nobj,nobj] of distances
            Default is to compute distances from tsne_results.
        nside : int 
            determines number of images to display (nside*nside)        
        filename : str
            Name to save the figure when closing (default is None)
        highlight : int list
            is an index list of points to mark by default
        highlightpars : dict
            Plot properties as parameters to pyplot.scatter for the highlighted points
        colors : (float arrray) 
            If not None then use this color value array for the scatter plot and not the clusters
        """
        self.clusters = clusters
        self.colors = colors
        self.tsne_results = tsne_results
        self.image_mapping = image_mapping
        self.clcount = np.bincount(self.clusters)
        self.nside = nside
        self.filename = filename
        self.highlight = highlight
        self.highlightpars = dict(facecolor='none', color='k',s=7)
        if highlightpars:
            self.highlightpars.update(highlightpars)
        if spd is None:
            spd = squareform(pdist(tsne_results))
        else:
            assert spd.shape == (len(image_mapping),len(image_mapping))
        self.pdist = spd
        self.create_plot()

    def create_plot(self):
        plt.rcParams.update({"font.size":8})
        self.fig = plt.figure(1,(15,7.5))
        # t-SNE plot goes in left half of display
        self.ptsne = self.fig.add_subplot(121)
        if self.colors is None:
           #cc = self.colormap()
           #sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=cc[self.clusters-1],s=5)
            sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=self.clusters,s=5)
        else:
            sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=self.colors,s=5)
        self.fig.colorbar(sp, ax=self.ptsne, label='cluster values')
        self.ptsne.set_title(self.title())
        if self.highlight is not None:
            self.ptsne.scatter(self.tsne_results[self.highlight,0],self.tsne_results[self.highlight,1],
                               **self.highlightpars)
        # self.ptsne.legend(loc='best')
        self.prevplt = None
        self.label = self.ptsne.set_xlabel("x")
        # nside x nside cutouts go in right half
        self.subplots = []
        nside = self.nside
        iplot = nside
        for i in range(nside*nside):
            iplot = iplot + 1
            if (iplot % (2*nside))==1:
                iplot = iplot + nside
            self.subplots.append(self.fig.add_subplot(nside,2*nside,iplot))
            self.subplots[-1].set_axis_off()
            self.subplots[-1].set_title(f"Sample {i+1}",size=9)
        self.plotobjects = [None]*len(self.subplots)
        self.fig.tight_layout()
        self.saved = False
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Start with showing some random figures
        self.showcluster(0.0,0.0)

       #plt.ion()
        self.fig.show()

        input('Hit the Any key to quit')

    @save_errors
    def close(self,event):
        if (not self.saved) and self.filename:
            self.saved = True
            self.fig.savefig(self.filename)
            errorlist.append(f"Saved figure to {self.filename}")

    def colormap(self):
        """Create colors for the clusters"""
        nclusters = len(self.clcount)-1
        # create color array for the clusters by wrapping the tab20 array
        # Or use a smaller set if fewer clusters
        if nclusters < 10:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.tab20
        ncolors = len(cmap.colors)
        cc = np.tile(np.array(cmap.colors),((nclusters+ncolors-1)//ncolors,1))
        cc = cc[:nclusters]
        return cc
        
    def findclosest(self,x,y):
        return np.argmin((self.tsne_results[:,0]-x)**2+(self.tsne_results[:,1]-y)**2)

    def title(self):
        return "Click to show random images within cluster"

    def select_sample(self,x,y):
        """Select a list of points near the click position
        
        This version shows the closest point plus a random sample of
        other members of the same cluster.
        """
        k = self.findclosest(x,y)
        i = self.clusters[k]
        nc = len(self.subplots)
        ww = np.where((self.clusters==i)&(np.arange(self.clusters.shape[0])!=k))[0]
        if len(ww) > nc-1:
            j = np.random.choice(ww,size=nc-1,replace=False)
        else:
            # group is too small
            j = ww
        j = np.insert(j,0,k)
        return j
    
    def showcluster(self,x,y):
        """Mark some objects and show images"""
        j = self.select_sample(x,y)
        self.label.set_text(f"x={x:.3f} y={y:.3f}")
        i = self.clusters[j[0]]
        if self.prevplt is not None:
            [x.remove() for x in self.prevplt]
        for sp in self.subplots:
            sp.clear()
            sp.set_axis_off()
        self.prevplt = self.ptsne.plot(self.tsne_results[j,0],self.tsne_results[j,1],'ks',
                                       label=f'Cl {i} ({self.clcount[i]} members)',
                                       fillstyle='none')
        self.ptsne.legend(loc='upper right')
        self.plotobjects = [None]*len(self.subplots)
        for isp, k in enumerate(j):
            im = Image.open(self.image_mapping['filename'][k])
            sp = self.subplots[isp]
            pim = sp.imshow(im,cmap='gray',origin='upper')
            self.plotobjects[isp] = k
            cdist = self.pdist[k,j[0]]
            sp.set_title("{} ({:.1f},{:.1f}) {:.3f}".format(self.image_mapping['name'][k],self.tsne_results[k,0],self.tsne_results[k,1],cdist),
                        size=8)

    @save_errors
    def onclick(self,event):
        if event.key == "alt":
            # Opt+click closes the plot
            plt.close(self.fig)
            if errorlist:
                print('\n'.join(errorlist))
        x = event.xdata
        y = event.ydata
        try:
            # allow clicks on the displayed images
            # this raises a ValueError if not in the list
            i = self.subplots.index(event.inaxes)
            j = self.plotobjects[i]
            if j is None:
                # ignore clicks on unpopulated images
                return
            x = self.tsne_results[j,0]
            y = self.tsne_results[j,1]
        except ValueError:
            pass
        self.showcluster(x,y)
        self.fig.canvas.draw_idle()


class InteractiveClosest(InteractivePlot):
    """Select closest points from within the cluster using pdist"""
    def title(self):
        return "Click to show closest images within cluster"

    def select_sample(self,x,y):
        k = self.findclosest(x,y)
        i = self.clusters[k]
        nc = len(self.subplots)
        ww = np.where(self.clusters==i)[0]
        ww = ww[np.argsort(self.pdist[ww,k])]
        ww = ww[:nc]
        return ww


class InteractiveAllclose(InteractivePlot):
    """Select closest points from entire sample (regardless of cluster)"""
    def title(self):
        return "Click to show closest images from any cluster"

    def select_sample(self,x,y):
        k = self.findclosest(x,y)
        nc = len(self.subplots)
        ww = np.argsort(self.pdist[:,k])
        ww = ww[:nc]
        return ww

class InteractiveFarthest(InteractivePlot):
    """Select farthest points from within the cluster"""
    def title(self):
        return "Click to show farthest images within cluster"

    def select_sample(self,x,y):
        k = self.findclosest(x,y)
        i = self.clusters[k]
        nc = len(self.subplots)
        # sort cluster members from largest to smallest distance to this object
        ww = np.where((self.clusters==i)&(np.arange(self.clusters.shape[0])!=k))[0]
        ww = ww[np.argsort(-self.pdist[ww,k])]
        ww = ww[:nc-1]
        ww = np.insert(ww,0,k)
        return ww

def fakeevent(x,y):
    from argparse import Namespace
    return Namespace(xdata=x,ydata=y,key='none')
