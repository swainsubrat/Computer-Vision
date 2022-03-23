from cProfile import label
import pickle
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict

def accuracy(Y: List, predY: List) -> float:
    """
    Get accuracy
    """
    Y = np.array(Y)
    predY = np.array(predY)
    accuracy = (Y == predY).sum()/ float(len(Y))
    accuracy = np.round(accuracy * 100, 2)

    return accuracy

def save(path: str, params: Dict) -> None:
    """
    Save model to path
    """
    outfile = open(path, 'wb')
    pickle.dump(params, outfile)
    outfile.close()


def load(path: str) -> Dict:
    """
    Load model from path
    """
    infile = open(path, 'rb')
    params = pickle.load(infile)
    infile.close()

    return params

def line_plot(train, valid, title, xl, yl, filename):
    plt.plot(train, label="Training")
    plt.plot(valid, label="Validation")
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.legend()

    plt.show()
    # plt.savefig(filename, dpi=600)

    return

def plot_accuracy(accuracies: List[List[float]],
                  title, xl, yl,
                  file_name: str=None) -> None:
    fig, ax = plt.subplots()
    ax.plot(accuracies[0], 'r', marker='o', label="Train")
    ax.plot(accuracies[1], 'g', marker='o', label="Validation")
    # ax.axis('equal')
    ax.legend()
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    plt.savefig(f"./plots/{file_name}", dpi=600)

if __name__ == "__main__":
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(accuracy(a, b))

    line_plot([1, 2, 3,], [3, 2, 1], "asf", "asdfs", "asfds", "./plots/sdf.png")