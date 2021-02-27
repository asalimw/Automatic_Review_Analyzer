import project1 as p1
import utils
import numpy as np
import matplotlib.pyplot as plt

def plot_toy_data1(algo_name, features, labels, thetas, ax=None):
    """
    Plots the toy data in 2D.
    Arguments:
    * algo_name - the string name of the learning algorithm used
    * features - an Nx2 ndarray of features (points)
    * labels - a length-N vector of +1/-1 labels
    * thetas - the tuple (theta, theta_0) that is the output of the learning algorithm
    * ax - matplotlib axis (optional), for subplots
    """
    # plot the points with labels represented as colors
    if not ax:
        f, ax = plt.subplots()
    colors = ['b' if label == 1 else 'r' for label in labels]
    ax.scatter(features[:, 0], features[:, 1], s=40, c=colors)
    # multiplying by 1.2 works because min is negative and max is positive
    xmin, xmax = features[:, 0].min() * 1.2, features[:, 0].max() * 1.2
    ymin, ymax = features[:, 1].min() * 1.2, features[:, 1].max() * 1.2

    # plot the decision boundary
    theta, theta_0 = thetas
    xs = np.linspace(xmin, xmax)
    ys = -(theta[0] * xs + theta_0) / (theta[1] + 1e-16)  # theta . [xs, ys] + theta0 = 0
    ax.plot(xs, ys, 'k-')

    ax.text(0.96, 0.95, 'w: ' + ', '.join(f'{w:.4f}' for w in theta),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.96, 0.945, 'b: ' + f'{theta_0:.4f}',
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes)
    # show the plot

    algo_name = ' '.join((word.capitalize() for word in algo_name.split(' ')))
    ax.set(title=f'{algo_name}', xlim=[xmin, xmax], ylim=[ymin, ymax])
    # end of plot_toy_data1

f, ax = plt.subplots(1, 3)
plot_toy_data1('Perceptron', toy_features, toy_labels, thetas_perceptron, ax=ax[0])
plot_toy_data1('Average Perceptron', toy_features, toy_labels, thetas_avg_perceptron, ax=ax[1])
plot_toy_data1('Pegasos', toy_features, toy_labels, thetas_pegasos, ax=ax[2])
f.suptitle('Classified Toy Data', y=1.0)
f.set_tight_layout(True)
plt.show()