import os
import sys
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from keras.datasets import mnist

def save_data(save_dir, file_name, data):
    np.save(os.path.join(save_dir, file_name), data)

def download_mnist(save_dir=os.path.join('C:\\', 'Users', 'ash', 'Downloads', 'mnist_npy'), verbose=2):
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    if verbose >= 2:
        print("Loading the data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if verbose >= 2:
        print("Completed")
    if verbose >= 2:
        print("Saving the data...")
    save_data(save_dir, 'x_train.npy', x_train)
    save_data(save_dir, 'x_test.npy', x_test)
    save_data(save_dir, 'y_train.npy', y_train)
    save_data(save_dir, 'y_test.npy', y_test)
    if verbose >= 2:
        print("Completed")

if __name__ == '__main__':
    save_dir=os.path.join('C:\\', 'Users', 'ash', 'Downloads', 'mnist_npy')
    print(save_dir)
    download_mnist(save_dir=save_dir)