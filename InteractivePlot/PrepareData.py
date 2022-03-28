import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import os
from tqdm.notebook import tqdm
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
import random
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import glob

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
import umap

class PrepareData:
    '''
    Superclass for preparing data for interactive t-SNE
    '''

    def get_images(self, DATA_PATH):

        '''
        Get list of images in a folder.

        Args:
            DATA_PATH (str) : Path to ImageFolder Dataset
        
        Returns:
            (list) : Order of images in a folder
        '''

        ims = []
        for folder in os.listdir(DATA_PATH):
            for im in os.listdir(f'{DATA_PATH}/{folder}'):
                ims.append(f'{DATA_PATH}/{folder}/{im}')
        return ims
    
    def embedding_gen(self, num_clusters, method, perplexity, n_neighbors, n_jobs):
        '''
        Helper function to compute the t-SNE or UMAP embeddings from the higher-dimensional data.
        '''

        tsne_results, spd, clusters = self.calculate_similarity(num_clusters, method, perplexity, n_neighbors, n_jobs)
        objects = self.object_creation()
        return tsne_results, spd, clusters, objects

    def calculate_similarity(self, num_clusters, method, perplexity, n_neighbors, n_jobs):

        '''
        Calculates necessary linkage variables for visualization.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to color code for visualization purposes
        
        Returns: 
           tsne_results (numpy.ndarray) : t-SNE coordinate mapping
           spd (numpy.ndarray) : a square array [nobj,nobj] of distances
           clusters (numpy.ndarray) : list of clusters
        '''

        if method == 'umap':
            tsne_results = self.fit_umap(self.data, n_neighbors, n_jobs)
        elif method == 'tsne':
            tsne_results = self.fit_tsne(self.data, perplexity, n_jobs)

       #pdt = pdist(self.data, metric= 'cosine')
        pdt = pdist(tsne_results)
        spd = squareform(pdt)

        # Why do you cluster the raw data and not the t-SNE embeddings?
       #z = linkage(self.data, method = 'centroid')
        z = linkage(tsne_results, method = 'centroid')
        clusters = fcluster(z.astype(float), num_clusters, criterion= 'maxclust')
        print("Linkage variables created.")

        return tsne_results, spd, clusters
    
    def fit_tsne(self, data, perplexity = 30, n_jobs = 1):

        '''
        Fits TSNE for the input embeddings

        Parameters
        ----------

        data : ndarray of shape (n_samples, n_features)
            Model features or SSL embeddings
        perplexity : int
            Used by t-SNE
            The perplexity is the number of nearest neighbors used in manifold learning
        n_jobs : int
            Number of parallel jobs to use
        
        Returns: 
            (numpy.ndarray) : TSNE result embeddings
        '''
        
        n_components = 2
        verbose = 1
        perplexity = perplexity
        n_iter = 1000
        metric = 'euclidean'
        n_jobs= n_jobs

        time_start = time.time()
        tsne_results = TSNE(n_components=n_components,
                            verbose=verbose,
                            perplexity=perplexity,
                            n_iter=n_iter,
                            n_jobs= n_jobs,
                            random_state=42,
                            metric=metric).fit_transform(data)

        print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return tsne_results

    def fit_umap(self, feature_list, n_neighbors = 5, n_jobs = 1):
        time_start = time.time()
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            random_state=42,
            n_components=2,
            verbose = 1,
            n_jobs = n_jobs,
            metric='euclidean')
        
        u = fit.fit_transform(feature_list)
        print('UMAP done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return u

    def object_creation(self, ims):
        '''
        Creates a DataFrame containing filename label mapping

        Args: 
            ims (list) : Order of images in a folder

        Retuns: 
            objects (pandas.DataFrame)  
        '''
        df_lis = []
        for img in ims:
            df_lis.append((img.split('/')[-2], img))

        objects = pd.DataFrame(df_lis, columns = ['name', 'filename'])
        print("Filename mapping done.")
        return objects


#*************************************************************************************************************
class PrepareData_general(PrepareData):
    '''
    Prepares a general set of Data for passing to the Interactive TSNE.
    '''

    def __init__(self, data, image_path, image_sort_key, num_clusters, method = 'tsne', perplexity= 30, n_neighbors= 5, n_jobs = 1):
        """ Initialize the data for general t-SNE use

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            The data to generate a t-SNE for.
        image_path  : str
            The path to the images to display on the interactive t-SNE
        image_sort_key : np.array of str
            A list of strings used to sort the image files
        num_clusters : int
            Number of clusters to color code for visualization purposes
        method : str
            The manifold learnign method to use
            Options: 'tsne' or 'umap'
        perplexity : int
            Used by t-SNE
            The perplexity is the number of nearest neighbors used in manifold learning
        n_neighbors : int
            Used by UMAP
        n_jobs : int
            Number of parallel jobs to use
        


        """
        self.data = data
        self.image_path = image_path
        ims = self.get_images_sorted(image_path, image_sort_key)
        self.objects = self.object_creation(ims)

        self.tsne_results, self.spd, self.clusters = self.calculate_similarity(num_clusters, method, perplexity, n_neighbors, n_jobs)


    def get_images_sorted (self, image_path, sort_key):
        """ Generates a list of images in a directory

        Sorts the image files based on <sort_key> str.
        sort_key is a string that shouls exist in the name of each image file

        Parameters
        ----------
        image_path : str
            Path to the images in a directory
        sort_key : np.array of str
            A list of strings used to sort the files in the output array
            sort_key must be the same length as the number of images in the directory

        Returns
        -------
        ims : list of str
            A list of file names
        """

        ims_raw = glob.glob(os.path.join(image_path, '*.png'), recursive=False)

        assert len(ims_raw) == len(sort_key), 'sort_key must be the same length as the number of images in the directory'

        ims = []
        for key in sort_key:
            idx = np.nonzero([filename.count(str(key)) for filename in ims_raw])[0]
            assert len(idx == 1), 'Error in sorting image files'
            ims.append(ims_raw[idx[0]])

        return ims


#*************************************************************************************************************
class PrepareData_torch_model(PrepareData):
    '''
    Prepares PyTorch model Data for passing to the Interactive TSNE. Specifically linkage variables used to represent relationship between clusters.
    '''

    def __init__(self, model, DATA_PATH, output_size, num_clusters, method = 'tsne', perplexity= 30, n_jobs = 4, n_neighbors= 5):
        self.model = model
        self.data_path = DATA_PATH
        self.embeddings = self.get_matrix(MODEL_PATH = model, DATA_PATH = DATA_PATH, output_size = output_size)
        self.ims = self.get_images(DATA_PATH)
        self.tsne_results, self.spd, self.cl, self.objects = self.embedding_gen(self.embeddings, num_clusters, self.ims, method, perplexity, n_neighbors, n_jobs)

    def get_matrix(self, MODEL_PATH, DATA_PATH, output_size):

        '''      
        Generate Embeddings for a given folder of images and a stored model.

        Args:
            MODEL_PATH (str) : Path to PyTorch Model
            DATA_PATH (str) : Path to ImageFolder Dataset
            output_size(int) : out_features value of the model
        
        Returns:
            (torch.Tensor) : Embeddings of the given model on the specified dataset.
        '''

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print('Using CUDA')
            model = torch.load(MODEL_PATH)
        else: 
            print('Using CPU')
            model = torch.load(MODEL_PATH, map_location = device)
        t = transforms.Compose(
        [transforms.Resize((224, 224)),
            transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))])

        dataset = torchvision.datasets.ImageFolder(DATA_PATH, transform = t)
        model.eval()
        if device == 'cuda':
            model.cuda()
        with torch.no_grad():
            if device == 'cuda':
                data_matrix = torch.empty(size = (0, output_size)).cuda()
            else: 
                data_matrix = torch.empty(size = (0, output_size))
            bs = 64
            if len(dataset) < bs:
                bs = 1
            loader = torch.utils.data.DataLoader(dataset, batch_size = bs, shuffle = False)
            for batch in tqdm(loader):
                if device == 'cuda':
                    x = batch[0].cuda()
                else:
                    x = batch[0]
                embeddings = model(x)[0]
                data_matrix = torch.vstack((data_matrix, embeddings))
        return data_matrix.cpu().detach().numpy() 




