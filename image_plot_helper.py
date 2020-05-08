import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

def imshow(image, ax=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))


    ax.imshow(image)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

def graphShow(ax, ps):

    ps = ps.data.numpy().squeeze()
    ax.barh(np.arange(2), ps)
    ax.set_aspect(0.1)
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(['Cat','Dog'], size='small')
    ax.set_title('Animal Probability')
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
