import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from sklearn.manifold import TSNE


def plot_tsne(features, labels, save_path):
    '''
    features:(N*m) 
    label:(N)
    '''
    tsne = TSNE(n_components=2, init='pca', random_state=7)

    class_num = len(np.unique(labels))  
    tsne_features = tsne.fit_transform(features)  

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    fig = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                          palette=sns.color_palette("hls", class_num),
                          data=df)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig(save_path, dpi=400)
    plt.clf()


def plot_patches(image_list, image_inds, save_path):
    image_size = 256  
    rows = 10  
    cols = 10  
    image = Image.new('RGB', (cols * image_size, rows * image_size))  
    ind = 0
    for y in range(rows):
        for x in range(cols):
            patch = Image.open(image_list[image_inds[ind]])
            ind += 1
            image.paste(patch, (x * image_size, y * image_size))
    image.save(save_path)  



if __name__ == '__main__':
    digits = datasets.load_digits(n_class=5)
    features, labels = digits.data, digits.target
    print(features.shape)
    print(labels.shape)
    features = np.random.randn(10, 1024)
    labels = np.array([0, 1, 2, 3, 4, 4, 3, 2, 1, 0])
    plot_tsne(features, labels)
