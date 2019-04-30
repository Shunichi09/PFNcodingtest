import numpy as np
import matplotlib.pyplot as plt

# original modules
from .common.NNfunctions import relu, sigmoid
from .common.NNmodules import GNN, Linear, BinaryCrossEntropyLossWithSigmoid
from .common.NNgrads import numerical_gradient
from .common.NNoptims import SGD
from .Nets import VanillaGNN

def main():
    # network
    net = VanillaGNN(seed=15)

    # input data
    G = np.array([[0., 1., 0., 0., 0., 1., 1., 0., 1., 1.],
                  [1., 0., 1., 1., 0., 0., 1., 1., 1., 0.],
                  [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
                  [0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                  [0., 0., 0., 0., 0., 0., 1., 0., 1., 1.], 
                  [1., 0., 1., 1., 0., 0., 0., 1., 1., 1.], 
                  [1., 1., 0., 1., 1., 0., 0., 0., 1., 1.], 
                  [0., 1., 1., 0., 0., 1., 0., 0., 1., 1.], 
                  [1., 1., 0., 0., 1., 1., 1., 1., 0., 1.],
                  [1., 0., 1., 1., 1., 1., 1., 1., 1., 0.]])
    
    t = [1] # target

    # test
    assert (G == G.T).all(), "input data should be a Symmetric matrix"
    assert (np.diag(G) == np.zeros(G.shape[0])).all()

    # loss func
    loss_fn = BinaryCrossEntropyLossWithSigmoid()

    # optimizer
    optimizer = SGD(net.parameters, alpha=0.0001)

    # training parameters
    EPOCHS = 100
    history_loss = []

    # for numerical grad
    def foward_with_loss(G, t):
        _, _, s = net.forward(G)
        loss = loss_fn(s, t)
        return loss
    
    grad_fn = lambda param : foward_with_loss(G, t)

    for epoch in range(EPOCHS):
        # calc grad
        numerical_gradient(net.parameters, grad_fn)
        optimizer.step()

        # calc loss
        _, predicted, _ = net.forward(G)
        loss = foward_with_loss(G, t)

        print("epoch : {} loss : {} predicted : {}".format(epoch + 1, round(loss,3), predicted[0][0]))
        
        # save
        history_loss.append(loss)
    
    # fig show
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(np.arange(EPOCHS), history_loss)

    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")

    fig.savefig("./src/results/main_2_result.png")
    plt.show()

if __name__ == "__main__":
    main()
