import numpy as np
import matplotlib.pyplot as plt

# original modules
from .Nets import VanillaGIN
from .common.NNmodules import BinaryCrossEntropyLossWithSigmoid
from .common.NNgrads import numerical_gradient
from .common.NNoptims import SGD, MomentumSGD
from .common.Datamodules import DataLoader
from .common.Datafunctions import shuffle, write_prediction_data

class Trainer():
    """trainer class
    Attributes
    ------------
    X_train : numpy.ndarray, shape(number_train, D, D)
        input data of train  
    Y_valid : numpy.ndarray, shape(number_train, 1)
        output data of train
    X_valid : numpy.ndarray, shape(number_valid, D, D)
        input data of validation
    Y_valid : numpy.ndarray, shape(number_valid, 1)
        output data of validation
    """
    def __init__(self, X_train, Y_train, X_valid, Y_valid):
        """
        Parameters
        -----------
        X_train : numpy.ndarray, shape(number_train, D, D)
            input data of train  
        Y_valid : numpy.ndarray, shape(number_train, 1)
            output data of train
        X_valid : numpy.ndarray, shape(number_valid, D, D)
            input data of validation
        Y_valid : numpy.ndarray, shape(number_valid, 1)
            output data of validation 
        """
        # network model
        self.net = VanillaGIN(seed=27)

        # data
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        
        # save
        self.history_train_loss = []
        self.history_train_accuracy = []
        self.history_valid_loss = []
        self.history_valid_accuracy = []

    def fit(self, EPOCHS=100, BATCH_SIZE=100, INTERVAL=1):
        """
        Parameters
        -----------
        EPOCHS : int, optional
            max epoch in traing, default is 100
        BATCH_SIZE : int, optional
            batch size in traing, default is 100
        INTERVAL : int, optional
            reported interval in training, default is 1
        """
        # loss
        self.loss_fn = BinaryCrossEntropyLossWithSigmoid()

        # optimizer
        # self.optimizer = SGD(self.net.parameters, alpha=0.0001)
        self.optimizer = MomentumSGD(self.net.parameters, alpha=0.0001, beta=0.9)

        # training parameters
        number_train = self.X_train.shape[0]
        iterations = number_train // BATCH_SIZE

        if number_train % BATCH_SIZE != 0: # for using all data
            iterations + 1

        # for numerical grad
        def foward_with_loss(G, t):
            _, _, s = self.net.forward(G)
            loss = self.loss_fn(s, t)
            return loss
        
        for epoch in range(EPOCHS):
            # shuffle data
            X, Y = shuffle(self.X_train, self.Y_train, seed=None) 
             
            for i in range(iterations):
                start = i * BATCH_SIZE
                end = start + BATCH_SIZE

                if end > number_train:
                    end = number_train - 1

                # make data set
                G = X[start:end]
                t = Y[start:end]

                # calc grad
                grad_fn = lambda param : foward_with_loss(G, t)
                numerical_gradient(self.net.parameters, grad_fn)
                # update parameter
                self.optimizer.step()

                # check loss
                train_loss, train_accuracy, _ = self._valid(self.X_train, self.Y_train)
                valid_loss, valid_accuracy, _ = self._valid(self.X_valid, self.Y_valid)
                # save
                self.history_train_loss.append(train_loss)
                self.history_train_accuracy.append(train_accuracy)
                self.history_valid_loss.append(valid_loss)
                self.history_valid_accuracy.append(valid_accuracy)
            
            if epoch % INTERVAL == 0:
                print("epoch : {}\ntrain_loss : {} train_acc : {} valid_loss : {} valid_acc : {}"
                       .format(epoch, np.round(train_loss, 3), np.round(train_accuracy, 3),
                               np.round(valid_loss, 3), np.round(valid_accuracy, 3)))
            
    def _valid(self, X_valid, Y_valid):
        """predict for validation data
        Parameters
        -----------
        X_valid : numpy.ndarray, shape(number_valid, D, D)
            input data of validation
        Y_valid : numpy.ndarray, shape(number_valid, 1)
            output data of validation 
        """
        number_valid = X_valid.shape[0]

        G = X_valid.copy()
        t = Y_valid.copy()

        _, predicted, s = self.net.forward(G)
        loss = self.loss_fn(s, t)   
        
        # accuracy
        # print(predicted.flatten() == t.flatten())
        number_correct = np.sum(predicted.flatten() == t.flatten())
        accuracy = round(number_correct / number_valid, 3)

        return loss, accuracy, predicted

    def predict(self, X_test):
        """
        Parameters
        -----------
        X_test : numpy.ndarray
            input data of test
        
        Returns
        ----------
        predicted : numpy.ndarray
            prediction data
        """
        _, predicted, _ = self.net.forward(X_test)

        return predicted

def main():
    # data path
    train_path = "./src/datasets/train/"
    test_path = "./src/datasets/test/"

    # load data
    dataloader = DataLoader()
    dataloader.load(train_path, test_path)
    X_train, Y_train, X_valid, Y_valid = dataloader.hold_out(ratio=0.3, shuffle=True, seed=5)
    X_test = dataloader.get_test_data()

    # make trainer
    trainer = Trainer(X_train, Y_train, X_valid, Y_valid)
    trainer.fit(EPOCHS=50, BATCH_SIZE=10)

    # predict
    prediction_data = trainer.predict(X_test)

    # write data
    path = "prediction.txt"
    write_prediction_data(path, prediction_data)


    # fig show
    # train
    fig_train = plt.figure()
    axis_train_loss = fig_train.add_subplot(111)
    axis_train_loss.set_xlabel("iterations")

    # train loss
    axis_train_loss.plot(np.arange(len(trainer.history_train_loss)), trainer.history_train_loss, label="loss_raw", color="b", alpha=0.15)
    # Moving Average
    num_moving = 75
    kernel = np.ones(num_moving)/num_moving
    axis_train_loss.plot(np.arange(0, len(trainer.history_train_loss)-num_moving), np.convolve(trainer.history_train_loss, kernel, mode="same")[:-num_moving], label="loss_ave", color="b")
    axis_train_loss.set_ylabel("loss")
    
    # train accuracy
    axis_train_accuracy = axis_train_loss.twinx() # add accuracy axis
    axis_train_accuracy.plot(np.arange(len(trainer.history_train_accuracy)), trainer.history_train_accuracy, label="accuracy_row", color="r", alpha=0.15)
    axis_train_accuracy.plot(np.arange(0, len(trainer.history_train_accuracy)-num_moving), np.convolve(trainer.history_train_accuracy, kernel, mode="same")[:-num_moving], label="accuracy_ave", color="r")    
    axis_train_accuracy.set_ylabel("accuracy")

    fig_train.legend(ncol=4)
    fig_train.savefig("./src/results/main_4_result_train.png")

    # valid
    fig_valid = plt.figure()
    axis_valid_loss = fig_valid.add_subplot(111)
    axis_valid_loss.set_xlabel("iterations")

    # valid loss
    axis_valid_loss.plot(np.arange(len(trainer.history_valid_loss)), trainer.history_valid_loss, label="loss_raw", color="y", alpha=0.15)
    axis_valid_loss.plot(np.arange(0, len(trainer.history_valid_loss)-num_moving), np.convolve(trainer.history_valid_loss, kernel, mode="same")[:-num_moving], label="loss_ave", color="y")
    axis_valid_loss.set_ylabel("loss")

    # valid accuracy
    axis_valid_accuracy = axis_valid_loss.twinx() # add accuracy axis
    axis_valid_accuracy.plot(np.arange(len(trainer.history_valid_accuracy)), trainer.history_valid_accuracy, label="accuracy_row", color="g", alpha=0.15)
    axis_valid_accuracy.plot(np.arange(0, len(trainer.history_valid_accuracy)-num_moving), np.convolve(trainer.history_valid_accuracy, kernel, mode="same")[:-num_moving], label="accuracy_ave", color="g")    
    axis_valid_accuracy.set_ylabel("accuracy")

    fig_valid.legend(ncol=4)
    fig_valid.savefig("./src/results/main_4_result_valid.png")
    plt.show()

if __name__ == "__main__":
    main()