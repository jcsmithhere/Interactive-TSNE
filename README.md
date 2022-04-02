# Interactive Viewer

Interactive Viewer is a tool that provides a way to visually view a data set's features in a 2-dimnensional space. One
can then select a data point in the representation and view nearest neighbors 

# Features

1. Either visualize a general higher dimensional data set or specifically a PyTorch model.

2. An Interactive plot to view the nearest neighbors of every data point in an embeddding space. Currently supports TSNE/UMAP. 

3. Visualize data in 3D using the TensorBoard Embedding Projector.

# Installation

This is a Fork of [Interactive t-SNE](https://github.com/spaceml-org/Interactive-TSNE). It is in active development and
so is recommended to install in development mode. There are different ways to do this. A simple way is to use the `-e`
option for pip. After activating your Python envirnment, in the top level directory of this repo type
```
pip install -e .
```

# Interactive Plot 



![alt text](https://s4.gifyu.com/images/2021-03-24-03-33-49-2.gif "Interactive Plot")


# Usage

There are two main usages:

### General multi-dimensional scalar feature set:

With this usage, you create a set of PNG figures and a higher dimensional feature set associated with each figures. 
You need to first place all figures associated with the data points in a single directory. Then create a
multi-dimensional array to pass to t-SNE to create the visual 2-D representation.

```
from InteractivePlot.PrepareData import PrepareData_general
from InteractivePlot.Viewer import InteractiveAllclose

# Create the t-SNE representation and image table
data = PrepareData_general(data=image_features, image_path=images, image_sort_key=sort_key, num_clusters=20, method = 'tsne', perplexity=30)

# Generate t-SNE figure allowing one to view figures of closest points to selected point. 
p = InteractiveAllclose(clusters, data.tsne_results, data.image_mapping, nside=3, colors=bolideBeliefArrayTrain)

```

`PrepareData_general` parameters:
`images` is the path to PNG images

`image_features` is the image representation data to cluster with t-SNE (or UMAP), of shape (n_datums, n_dimensions)

`sort_key` is a list of strings used to sort the image files to correspond to the data in `image_features`
Each string should correspond to a unique string segment in each PNG figure filename.

`InteractiveAllclose` parameters:
'clusters' is a list of clusters (array of ints) used to colorize the t-SNE figure. 
'colors' is an optional parmater, if passed then use this color value array (array of floats) for the scatter plot and not the clusters
`nside` is the number of figures to show in a nside x nside grid.

See the module function headers for more details and more advanced usage.

Here is an example generated figure. The `colors=` optional argument is used to colorize the t-SNE points. 
One clicks on a point in the t-SNE. The selected point image is shown in the upper
left corners and its nearest neihgbors are also shown.
<img src="./example/example_tsne_figure_color_by_score.png">


# Everything below is the same as in the original repo: 


### PyTorch model embeddings:
This is the same as in the original repo version: 

```
from InteractivePlot import PrepareData
from InteractivePlot import InteractiveAllclose
%matplotlib notebook

model = 'uc_merced.pt'
data = 'UCMerced_LandUse/Images'

data = PrepareData(model = model, DATA_PATH= data, num_clusters= 21, output_size= 21)

p = InteractiveAllclose(tcl = data.cl, 
                        tsne_obj = data.tsne_obj, 
                        objects = data.objects, 
                        spd =data.spd)
```

num_clusters : the number of clusters to be mapped to a color scheme. For color coding purposes only.

output_size : the number of output dimensions of your model.

**NOTE** : This currently works only on Jupyter notebook instances that support either the __widget__ or the __notebook__ matplotlib backends. 
Does not currently support Colab. 


# TensorBoard Projector


<img src="TensorBoard.gif?raw=true" width="2000px">

## Usage 

```
#Initialize model and data

model = torchvision.model.resnet18(pretrained= True) #Load model
model.cuda()
model.eval()

tfs = transforms.Compose([transforms.Resize((128, 128)), 
                          transforms.ToTensor(),
                          transforms.Normalize(mean = [0.485], std = [0.229])])

dataset = FashionMNIST(root = r'./FMINST', download = True, transform= tfs)
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle= True)
batch_imgs, batch_imgs = next(iter(data_loader))



#Start the projector
from InteractivePlot import Projector

vis = Projector(model = model, EXPT_NAME = 'projector_test', LOG_PATH = '.')
vis.write_embeddings(batch_imgs)
vis.create_tensorboard_log()

```

This will output a log directory where the TensorBoard files are written, and you can directly launch TensorBoard from that directory. 

```
tensorboard --logdir=output path
```



## Coming Updates
- Streamlit hosting
- TensorFlow models
- More Dimensionality Reduction methods
