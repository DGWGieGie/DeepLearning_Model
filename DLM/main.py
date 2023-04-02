import torch

import LeNet
import AlexNet

from DL_Tool import utils

def getNetwork(netName=""):
    if str.lower(netName) == "lenet":
        net = LeNet.network
    elif str.lower(netName) == "alexnet":
        net = AlexNet.network
    return net

def saveNetwork(net, netName=""):
    torch.save(net.state_dict(),f'model/{netName}Model.pth')
    print(f"Model saved to {netName}Model.pth")


if __name__ == '__main__':
    # netName = "LeNet"
    # network = getNetwork(netName)
    
    # batch_size = 256
    # train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)
    
    # lr, num_epochs = 1e-1, 20
    # utils.train_NN(network, train_iter, test_iter, num_epochs, lr, utils.try_gpu())

    # saveNetwork(network, netName)

    netName = "AlexNet"
    network = getNetwork(netName)
    
    batch_size = 128
    train_iter, test_iter = utils.load_data_CIFAR100(batch_size=batch_size, train_trans=AlexNet.transform('train'), test_trans=AlexNet.transform('test'))
    
    lr, num_epochs = 1e-2, 90
    utils.train_NN(network, train_iter, test_iter, num_epochs, lr, utils.try_gpu())

    saveNetwork(network, netName)
