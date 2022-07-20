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
#from tqdm.notebook import tqdm
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
import re

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

from self_supervised_learner.SSL import supported_techniques 


class PrepareData:
    '''
    Superclass for preparing data for interactive t-SNE

    Class Attributes:
    
    '''

    def get_images_sorted (self, image_path, sort_key):
        """ Generates a list of images in a directory

        Sorts the image files based on <sort_key> str.
        sort_key is a string that should exist in the name of each image file.
        The key can contain the file directory name and regular expressions.

        Any sort_keys with no corresponding images will display a dummy image.

        Parameters
        ----------
        image_path : str
            Path and regular expression to the images in a directory
        sort_key : np.array of str
            A list of strings used to sort the files in the output array
            sort_key must be the same length as the number of images in the directory
            If None then the images will be used in whatever order they are in the directory

        Returns
        -------
        ims : list of str
            A list of file names
        """

        dummy_img = '/home/bohr/Pictures/download.jpeg'

        if sort_key is None:
            ims = self.get_images(image_path)

        else:

            ims_raw = glob.glob(image_path, recursive=False)
            
          # assert len(ims_raw) == len(sort_key), 'sort_key must be the same length as the number of images in the directory'
            
            ims = []
            for key in sort_key:
               #idx = np.nonzero([re.search(key, filename) for filename in ims_raw])[0]
                idx = np.nonzero([filename.count(key) for filename in ims_raw])[0]
                assert len(idx) <= 1, 'Error: Two or more images match a sort key'
                if len(idx) < 1:
                    ims.append(dummy_img)
                else:
                    ims.append(ims_raw[idx[0]])

        return ims

    def get_images(self, image_path):

        '''
        Get list of images in a folder.

        Args:
            image_path (str) : Path to ImageFolder Dataset
        
        Returns:
            (list) : Order of images in a folder
        '''

        ims = []
        for folder in os.listdir(image_path):
            for im in os.listdir(f'{image_path}/{folder}'):
                ims.append(f'{image_path}/{folder}/{im}')
        return ims
    
    def calculate_similarity(self, num_clusters, method, perplexity, n_neighbors, n_jobs):

        '''
        Calculates necessary linkage variables for visualization.

        Parameters
        ----------
        num_clusters : int
            Number of clusters to color code for visualization purposes
        
        Returns: 
           tsne_results (numpy.ndarray) : t-SNE coordinate mapping
           tsne_dist (numpy.ndarray) : a square array [nobj,nobj] of distances between all points
           clusters (numpy.ndarray) : list of clusters
        '''

        if method == 'umap':
            tsne_results = self.fit_umap(self.data, n_neighbors, n_jobs)
        elif method == 'tsne':
            tsne_results = self.fit_tsne(self.data, perplexity, n_jobs)

       #pdt = pdist(self.data, metric= 'cosine')
        pdt = pdist(tsne_results)
        tsne_dist = squareform(pdt)

        # Why do you cluster the raw data and not the t-SNE embeddings?
       #z = linkage(self.data, method = 'centroid')
        z = linkage(tsne_results, method = 'centroid')
        clusters = fcluster(z.astype(float), num_clusters, criterion= 'maxclust')
        print("Linkage variables created.")

        return tsne_results, tsne_dist, clusters
    
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
        --------
        tsne_results : (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        '''
        
        n_components = 2
        verbose = 1
        perplexity = perplexity
        n_iter = 1000
        metric = 'euclidean'
        n_jobs= n_jobs

        time_start = time.time()
        tsne_results = TSNE(n_components=n_components,
                            init='pca',
                            verbose=verbose,
                            perplexity=perplexity,
                            learning_rate='auto',
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

        # Remove infs from set
        nanHere = np.union1d(np.nonzero(np.isnan(u[:,0]))[0],np.nonzero(np.isnan(u[:,1]))[0])
        # For now, just replace with 0, 0
        u[nanHere,:] = (0.0, 0.0)

        print('UMAP done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return u

    def image_mapping_creation(self, ims):
        '''
        Creates a DataFrame containing filename label mapping

        Parameters
        ----------
            ims (list) : Order of images in a folder

        Returns
        -------
            image_mapping (pandas.DataFrame)  
               ['name', 'filename'] pairs for each image 
        '''
        df_lis = []
        for img in ims:
            df_lis.append((img.split('/')[-2], img))

        image_mapping = pd.DataFrame(df_lis, columns = ['name', 'filename'])
        print("Filename mapping done.")
        return image_mapping


#*************************************************************************************************************
class PrepareData_general(PrepareData):
    '''
    Prepares a general set of Data for passing to the Interactive TSNE.
    '''

    def __init__(self, data, image_path, num_clusters, image_sort_key=None, method = 'tsne', perplexity= 30, n_neighbors= 5, n_jobs = 1):
        """ Initialize the data for general t-SNE use

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            The data to generate a t-SNE for.
        image_path  : str
            The path to the images to display on the interactive t-SNE
        image_sort_key : np.array of str
            A list of strings used to sort the image files to correspond to the data in <data>
            Each string should correspond to a unique string segment in each figure filename.
            If None then the images will be used in whatever order they are in the directory
        num_clusters : int
            Number of clusters to color code for visualization purposes
            Uses a hierachical clustering method.
        method : str
            The manifold learning method to use
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
        self.image_mapping = self.image_mapping_creation(ims)

        self.tsne_results, self.tsne_dist, self.clusters = self.calculate_similarity(num_clusters, method, perplexity, n_neighbors, n_jobs)



#*************************************************************************************************************
class PrepareData_SSL_model(PrepareData):
    '''
    Prepares SSL model Data for passing to the Interactive TSNE. Specifically, linkage variables used to represent relationship between clusters.
    '''

    def __init__(self, model_path, technique, image_path, num_clusters, method = 'tsne', perplexity= 30, n_neighbors= 5, n_jobs = 1):
        """ Initialize the data for a PyTorch model

        Parameters
        ----------
        model_path : 
            Path to the PyTorch model
        technique : 
            model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
        image_path  : str
            The path to the images to display on the interactive t-SNE using the model to organize
        num_clusters : int
            Number of clusters to color code for visualization purposes
            Uses a hierachical clustering method.
        method : str
            The manifold learning method to use
            Options: 'tsne' or 'umap'
        perplexity : int
            Used by t-SNE
            The perplexity is the number of nearest neighbors used in manifold learning
        n_neighbors : int
            Used by UMAP
        n_jobs : int
            Number of parallel jobs to use
        


        """
        self.model_path = model_path
        self.image_path = image_path
        self.data = self.get_embedding_matrix(model_path = model_path, technique=technique, image_path = image_path)
        ims = self.get_images(image_path)
        self.image_mapping = self.image_mapping_creation(ims)

        self.tsne_results, self.tsne_dist, self.clusters = self.calculate_similarity(num_clusters, method, perplexity, n_neighbors, n_jobs)

    def get_embedding_matrix(self, model_path, technique, image_path):

        '''      
        Generate Embeddings for a given folder of images and a stored model.

        Args:
            model_path : str
                Path to PyTorch Model
            technique : 
                model technique used {SIMCLR, SIMSIAM or CLASSIFIER}
            image_path : str
                Path to ImageFolder Dataset
        
        Returns:
            (torch.Tensor) : Embeddings of the given model on the specified dataset.
        '''

        # Used the correct loader for the specified technique
        technique = supported_techniques[technique]

       #print('USING GPU # 1!!!!!!!!')
       #print('USING GPU # 1!!!!!!!!')
       #print('USING GPU # 1!!!!!!!!')
        gpu_index = 0

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            print('Using CUDA')
           #model = torch.load(model_path)

            model = technique.load_from_checkpoint(model_path)
        else: 
            print('Using CPU')
            model = torch.load(model_path, map_location = device)

        image_size = model.image_size
        embedding_size = model.encoder.embedding_size

        # Permute image so that it is the correct orientation for our model
        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2,0,1).float()

        t = transforms.Compose([
                    transforms.Resize((image_size,image_size)),
                    transforms.Lambda(to_tensor)
                    ])

        model.eval()
        if device == 'cuda':
            model.cuda(gpu_index)


        #***
        # Extract embeddings for all the images
        imageFiles = glob.glob(image_path+'/Unlabelled/*.jpg')
        embedding = np.zeros([len(imageFiles), embedding_size])
        for idx in tqdm(range(len(imageFiles)), 'Computing model image embeddings'):
            f = imageFiles[idx]
            
            im = Image.open(f).convert('RGB')
            
            datapoint = t(im).unsqueeze(0).cuda() #only a single datapoint so we unsqueeze to add a dimension
            
            with torch.no_grad():
                embedding[idx,:] = np.array(model(datapoint)[0].cpu()) #get embedding


        return embedding

        #***********
        """
        # TODO: Get the image loading process to work with torch DataLoader
        # Extract embeddings for all the images
        imageFiles = torchvision.datasets.ImageFolder(image_path, transform = t)
        embedding = torch.empty(size = (0, embedding_size)).cuda(gpu_index)
        bs = 64
        if len(imageFiles) < bs:
            bs = 1
        loader = torch.utils.data.DataLoader(imageFiles, batch_size = bs, shuffle = False)
        for batch in tqdm(loader, 'Computing model image embeddings'):
                    
            x = batch[0].cuda(gpu_index)
            
            with torch.no_grad():
                embedding = torch.vstack((embedding, model(x)))


        return embedding.cpu().detach().numpy()

        """

        """
        dataset = torchvision.datasets.ImageFolder(image_path, transform = t)
        with torch.no_grad():
            if device == 'cuda':
                data_matrix = torch.empty(size = (0, embedding_size)).cuda(gpu_index)
            else: 
                data_matrix = torch.empty(size = (0, embedding_size))
            bs = 64
            if len(dataset) < bs:
                bs = 1
            loader = torch.utils.data.DataLoader(dataset, batch_size = bs, shuffle = False)
            for batch in tqdm(loader, 'Computing model image embeddings'):
                if device == 'cuda':
                    x = batch[0].cuda(gpu_index)
                else:
                    x = batch[0]
                embeddings = model(x)
                data_matrix = torch.vstack((data_matrix, embeddings))
        return data_matrix.cpu().detach().numpy() 
        """




