import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


class Plotter:

    def __init__(self, input_dict, dict_keys_as_labels = True):
        """
        The class plots high dimensional vectors into 2 or 3 dimensional plot.
        arguments:
        
            - input_dict must be like: {'comedy' : [numpy_matrix, label_number, numpy_array_of_file_names],
                                        'action' : [numpy_matrix, label_number, numpy_array_of_file_names],
                                         .................................................................,
                                        }
            - by default dict keys are used in plot as class labels. keys might be helpful, but you can exclude them in plot
              by setting 'dict_keys_as_labels' to False. you could use any input_dict key (just don't dublicate them), 
              it's for your attention to see what classes you added and their labels;
            - 'numpy_matrix' must be 2 dimentional, samples in rows and their features in cols;
            - 'label_number' is just label integer. That integers are used at plot to generate colors. The numbers must be started
              from zero;
            - 'numpy_array_of_file_names' must be a numpy vector of strings like : np.array(["comedy", "action", ...])
        
        To plot vectors use class method plot (see below)
        """
        self.dict_keys_as_labels = dict_keys_as_labels
        
        step = 0
        for key in input_dict.keys():
            
            # init step
            if step == 0:
                # init numpy arrays for latter usage
                self.x = np.zeros(input_dict[key][0].shape)
                self.x = input_dict[key][0]
                
                # init labels for plotting
                self.y = np.array([])
                repeat_vals = np.repeat(input_dict[key][1], input_dict[key][0].shape[0])
                labels = np.array(repeat_vals)
                self.y = np.append(self.y, labels)
                
                # init file names for plotting
                self.file_names = input_dict[key][2]
                
                # here I use dict keys as additional labels
                if dict_keys_as_labels:
                    self.z = np.array([])
                    self.z = np.append(self.z, key)

                step += 1
                continue

            self.x = np.concatenate((self.x, input_dict[key][0]))

            repeat_vals = np.repeat(input_dict[key][1], input_dict[key][0].shape[0])
            labels = np.array(repeat_vals)
            self.y = np.append(self.y, labels)
            
            self.file_names = np.concatenate((self.file_names, input_dict[key][2]))
            
            # here I use dict key as additional labels
            if dict_keys_as_labels:
                self.z = np.append(self.z, key)

    def plot(self, func_type='tsne', dim='3d', figsize=(10,10)):
        """
        the function plots data in 2 or 3 dimensions with default figsize of (10, 10). "func_type" defines
        which function to use to reduce dimentions TSNE or PCA. Default is TSNE
        - if you use jupyter notebook then excute command '%matplotlib notebook' to use interactive plots
        """
        if func_type == 'tsne':
            if dim == '3d':
                fashion_tsne = TSNE(n_components=3, random_state=123).fit_transform(self.x)
                self.fashion_scatter_3d(fashion_tsne, self.y, self.dict_keys_as_labels, self.file_names, figsize)
            
            elif dim == '2d':
                
                fashion_tsne = TSNE(n_components=2, random_state=123).fit_transform(self.x)
                self.fashion_scatter_2d(fashion_tsne, self.y, self.dict_keys_as_labels, self.file_names, figsize)
            else:
                raise ValueError('Unexpected keyword argument!!!')
            
        elif func_type == 'pca':
            if dim == '3d':
                pca_result = PCA(n_components=3).fit_transform(self.x)
                self.fashion_scatter_3d(pca_result, self.y, self.dict_keys_as_labels, self.file_names, figsize)
            
            elif dim == '2d':
                
                pca_result = PCA(n_components=2).fit_transform(self.x)
                self.fashion_scatter_2d(pca_result, self.y, self.dict_keys_as_labels, self.file_names, figsize)
            else:
                raise ValueError('Unexpected keyword argument!!!')

        else:
            raise ValueError('Unexpected keyword argument. Use lower case letters!!!')

    def cl(self):
        plt.close()

    def fashion_scatter_3d(self, x, colors, dict_keys_as_labels, file_names, figsize):
        # choose a color palette with seaborn.
        num_classes = len(np.unique(colors))
        palette = np.array(sns.color_palette("hls", num_classes))

        # create a scatter plot.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=40, c=palette[colors.astype(np.int)], alpha=1)

        # total number of points to plot 
        n = file_names.shape[0]
        # add annotation to points
        for point in range(n):
            ax.text(x[point, 0], x[point, 1], x[point, 2], file_names[point], fontsize=12, alpha=0.5)
            
        # add the labels for each digit corresponding to the label
        txts = []
        for i in range(num_classes):
            # Position of each label at median of data points.

            xtext, ytext, ztext = np.median(x[colors == i, :], axis=0)
            
            if dict_keys_as_labels: 
                txt = ax.text(xtext, ytext, ztext, (self.z[i]+"_"+str(i)), fontsize=24)
            else: 
                txt = ax.text(xtext, ytext, ztext, str(i), fontsize=24)
                
            txt.set_path_effects([
                patheffects.Stroke(linewidth=5, foreground="w"),
                patheffects.Normal()])
            txts.append(txt)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.tick_params(labelsize=7)
        
        plt.show()
        
    def fashion_scatter_2d(self, x, colors, dict_keys_as_labels, file_names, figsize):
        # choose a color palette with seaborn.
        num_classes = len(np.unique(colors))
        palette = np.array(sns.color_palette("hls", num_classes))

        # create a scatter plot.
        plt.figure(figsize=figsize)
        ax = plt.subplot(aspect='equal')
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # total number of points to plot 
        n = file_names.shape[0]
        # add annotation to points
        for point in range(n):
            ax.annotate(file_names[point], (x[point, 0], x[point, 1]), fontsize=12, alpha=0.5)
        
        # add the labels for each digit corresponding to the label
        txts = []
        for i in range(num_classes):
            # Position of each label at median of data points.

            xtext, ytext = np.median(x[colors == i, :], axis=0)
            
            if dict_keys_as_labels: 
                txt = ax.text(xtext, ytext, (self.z[i]+"_"+str(i)), fontsize=24)
            else: 
                txt = ax.text(xtext, ytext, str(i), fontsize=24)
                
            txt.set_path_effects([
                patheffects.Stroke(linewidth=5, foreground="w"),
                patheffects.Normal()])
            txts.append(txt)

        plt.show()
