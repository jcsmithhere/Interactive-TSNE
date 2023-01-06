import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser
import sys, os
import copy
from shutil import copyfile
from argparse import Namespace
import numpy as np
from torchvision import transforms
from matplotlib import offsetbox

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
    def __init__(self, clusters, tsne_results, image_mapping, tsne_dist=None, nside=4,
                 filename=None, highlight=None, highlightpars=None, colors=None,
                 plot_thumbnails=False, thumbnail_size=128, plot_max_n_thumbnails=None, save_selection_path=None):
        """

        Generates an interactive t-SNE. When you clock o a point on the t-SNE on the left-hand side fo the figure, a
        selection of images corresponding to the nearest points are displayed ont eh right-hand side of the figure.

        If save_selection_path is not none, then when you click on an image on the right hand side, that images is
        copied to the given directory.

        If save_selection_path is None then no images are copied.

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
        tsne_dist is a square array [nobj,nobj] of distances between all t-SNE points
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
        plot_thumbnails : bool
            If True then plot a thumbnail for each image, otherwise, plot a colored dots
        thumbnail_size : int
            int x int pixel size of thumbnails
        plot_max_n_thumbnails : int
            The maximum number of thumbnails to plot, None means no limit
        save_selection_path = str
            If not None then save copies of selected figures at the given path
        """
        self.clusters = clusters
        self.colors = colors
        self.plot_thumbnails = plot_thumbnails
        self.thumbnail_size = thumbnail_size
        self.plot_max_n_thumbnails = plot_max_n_thumbnails
        self.save_selection_path = save_selection_path
        # If saving selected figures to directory then make sure directory exists
        if self.save_selection_path is not None:
            if not os.path.isdir(self.save_selection_path):
                try:
                    os.makedirs(self.save_selection_path)
                except OSError:
                    raise Exception('Creation of the directory {} failed'.format(self.save_selection_path))



        self.tsne_results = tsne_results
        self.image_mapping = image_mapping
        self.clcount = np.bincount(self.clusters)
        self.nside = nside
        self.filename = filename
        self.highlight = highlight
        self.highlightpars = dict(facecolor='none', color='k',s=7)
        if highlightpars:
            self.highlightpars.update(highlightpars)
        if tsne_dist is None:
            tsne_dist = squareform(pdist(tsne_results, 'euclidean'))
        else:
            assert tsne_dist.shape == (len(image_mapping),len(image_mapping))
        self.tsne_dist = tsne_dist
        # This is the tsne distance but with the identity values NaNed
        self.tsne_dist_naned = copy.copy(self.tsne_dist)
        for i in range(self.tsne_results.shape[0]):
            self.tsne_dist_naned[i,i] = np.nan
        self.create_plot()

        # Save the plot view limits so we can determien when it changes
        self.xlim_save = self.ptsne.get_xlim()
        self.ylim_save = self.ptsne.get_ylim()

        input('Hit the Any key to quit')

    def create_plot(self):
        dot_size = 10
        plt.rcParams.update({"font.size":8})
        self.fig = plt.figure(1,(15,7.5))

        #***
        # t-SNE plot goes in left half of display
        self.ptsne = self.fig.add_subplot(121)
        if self.plot_thumbnails:
            self.thumbnail_plotting()
        else:
            if self.colors is None:
               #cc = self.colormap()
               #sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=cc[self.clusters-1],s=5)
                sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=self.clusters,s=dot_size)
            else:
                sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=self.colors,s=dot_size)
            self.fig.colorbar(sp, ax=self.ptsne, label='cluster values')

        self.ptsne.set_title(self.title())
        if self.highlight is not None:
            self.ptsne.scatter(self.tsne_results[self.highlight,0],self.tsne_results[self.highlight,1],
                               **self.highlightpars)
        # self.ptsne.legend(loc='best')
        self.prevplt = None
        self.label = self.ptsne.set_xlabel("x")

        #***
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


    @save_errors
    def close(self,event):
        if (not self.saved) and self.filename:
            self.saved = True
            self.fig.savefig(self.filename)
            errorlist.append(f"Saved figure to {self.filename}")

    def thumbnail_plotting(self):
        """ Plot thumbnails for each point on the t-SNE

        """

        # Start by removing any existing artists
        if len(self.ptsne.artists) > 0:
            for artist in self.ptsne.artists:
                artist.remove()

       #plt.plot(X[i,0], X[i,1], '.r')
        sp = self.ptsne.scatter(self.tsne_results[:,0],self.tsne_results[:,1],c=self.clusters,s=0.1)

        # Only show N thumbnails within plot view
        # Only process thumbnails within the plot view limits (for speed)
        images_to_plot = np.array([i for i,_ in enumerate(self.tsne_dist) if self.image_within_plot_view(i)])

        # Randomly select images to plot within plot view
        if self.plot_max_n_thumbnails is not None:
            n_images_to_plot = np.min((self.plot_max_n_thumbnails, len(images_to_plot)))
        else:
            n_images_to_plot = len(images_to_plot)
        rng = np.random.default_rng()
        images_to_plot = images_to_plot[rng.permutation(len(images_to_plot))[0:n_images_to_plot]]


      # # Find nearest neighbor distance
      # nearest_neighbor_idx = np.nanargmin(self.tsne_dist_naned[images_to_plot], axis=0)
      # nearest_neighbor_dist = self.tsne_dist_naned[images_to_plot, nearest_neighbor_idx]
      # # Sort by nearest neighbor
      # images_to_plot = np.array(images_to_plot)[np.argsort(nearest_neighbor_dist)[-self.plot_max_n_thumbnails-1:-1]]


       #for i in range(self.tsne_results.shape[0]):
        for i in images_to_plot:
            
            im = Image.open(self.image_mapping['filename'][i]).convert('RGB')
            t = transforms.Compose([
                        transforms.Resize((self.thumbnail_size,self.thumbnail_size))
                        ])

            im = t(im)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(im, cmap=plt.cm.gray_r),
                self.tsne_results[i])
            self.ptsne.add_artist(imagebox)

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
        return "Click to show closest images to selected image"

    def select_sample(self,x,y):
        """Select closest points from entire sample (regardless of cluster)"""
        k = self.findclosest(x,y)
        nc = len(self.subplots)
        ww = np.argsort(self.tsne_dist[:,k])
        ww = ww[:nc]
        return ww

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
        # This records the filenames for the images plotted on right
        self.plotFilenames = np.full(len(self.subplots), None)
        for isp, k in enumerate(j):
            self.plotFilenames[isp] = self.image_mapping['filename'][k]
            im = Image.open(self.image_mapping['filename'][k])
            sp = self.subplots[isp]
            pim = sp.imshow(im,cmap='gray',origin='upper')
            self.plotobjects[isp] = k
            cdist = self.tsne_dist[k,j[0]]
           #sp.set_title("{} ({:.1f},{:.1f}) {:.3f}".format(self.image_mapping['name'][k],self.tsne_results[k,0],self.tsne_results[k,1],cdist), size=8)
            if self.colors is not None:
                sp.set_title("dist={:.3f}, color={:.3f}".format(cdist, self.colors[k]), size=8)
            else:
                sp.set_title("dist={:.3f}".format(cdist), size=8)

    @save_errors
    def onclick(self,event):
        if event.key == "alt":
            # Opt+click closes the plot
            plt.close(self.fig)
            if errorlist:
                print('\n'.join(errorlist))
        x = event.xdata
        y = event.ydata

       #print('Clicked on (x,y): ({}, {})'.format(x,y))

        # If we click on a displayed image then display the image file
        try:
            # allow clicks on the displayed images
            # this raises a ValueError if not in the list
            i = self.subplots.index(event.inaxes)
            j = self.plotobjects[i]
            if j is None:
                # ignore clicks on unpopulated images
                return
            x_image = self.tsne_results[j,0]
            y_image = self.tsne_results[j,1]

            # Save image to provided path
            if self.save_selection_path is not None:
                self.copy_or_create_symlinks(self.plotFilenames[i])


            return

        except ValueError:
            pass

        # Replot the thumbnails to update which images are shown
        # but only if the plot view limits had changed
        if self.plot_thumbnails:
            if not (self.ptsne.get_xlim() == self.xlim_save and
                    self.ptsne.get_ylim() == self.ylim_save):
                self.thumbnail_plotting()

        self.showcluster(x,y)
        self.fig.canvas.draw_idle()

        # If the plot view area has changed then update the saved values
        self.xlim_save = self.ptsne.get_xlim()
        self.ylim_save = self.ptsne.get_ylim()

    def copy_or_create_symlinks(self, targetFileList, createSymlinks=False):

        if not isinstance(targetFileList, list):
            targetFileList = [targetFileList]

        for targetFileStr in targetFileList:
            dirName, fileName = os.path.split(targetFileStr)
            outFileStr = os.path.join(self.save_selection_path, fileName)
            if (not os.path.isfile(outFileStr) and not os.path.islink(outFileStr)):
                print('Copying image file: {}'.format(targetFileStr))
                if createSymlinks: 
                    os.symlink(targetFileStr, outFileStr)
                else:
                    copyfile(targetFileStr, outFileStr)

    def image_within_plot_view(self, i):
        """ Returns True if the image is within the plot view
        """

        xlims = self.ptsne.get_xlim()
        ylims = self.ptsne.get_ylim()

        if  (self.tsne_results[i,0] < xlims[0] or
            self.tsne_results[i,0] > xlims[1] or
            self.tsne_results[i,1] < ylims[0] or
            self.tsne_results[i,1] > ylims[1]):

            return False
        else:
            return True



#*************************************************************************************************************

class InteractiveClosestInCluster(InteractivePlot):
    """Select closest points from within the cluster using pdist"""
    def title(self):
        return "Click to show closest images within cluster"

    def select_sample(self,x,y):
        k = self.findclosest(x,y)
        i = self.clusters[k]
        nc = len(self.subplots)
        ww = np.where(self.clusters==i)[0]
        ww = ww[np.argsort(self.tsne_dist[ww,k])]
        ww = ww[:nc]
        return ww

class InteractiveRandomInCluster(InteractivePlot):
    """This version shows the closest point plus a random sample of
        other members of the same cluster.
    """
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
    


class InteractiveFarthestInCluster(InteractivePlot):
    """Select farthest points from within the cluster"""
    def title(self):
        return "Click to show farthest images within cluster"

    def select_sample(self,x,y):
        k = self.findclosest(x,y)
        i = self.clusters[k]
        nc = len(self.subplots)
        # sort cluster members from largest to smallest distance to this object
        ww = np.where((self.clusters==i)&(np.arange(self.clusters.shape[0])!=k))[0]
        ww = ww[np.argsort(-self.tsne_dist[ww,k])]
        ww = ww[:nc-1]
        ww = np.insert(ww,0,k)
        return ww

def fakeevent(x,y):
    from argparse import Namespace
    return Namespace(xdata=x,ydata=y,key='none')
