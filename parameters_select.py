import model
import utils
import numpy as np
import pandas as pd

batch_size = 256
hidden_size = [128, 256, 512, 1024]     # size of hidden layers
reg = [0, 1e-1, 1e-2, 1e-3]     # regularization strength
lrs = [1e-1, 1e-2, 1e-3, 1e-4]  # learning rate
lr_decay = 0.95
iters_num = 5000

min_epoch = 5 
max_epoch = 50  
decay_num = 5  # If the accuracy of consecutive $decay_num epoches decreases continuously, the subsequent training is considered meaningless and to be stopped

if __name__ == '__main__':

    np.random.seed(7)

    X, y = utils.load_mnist('./mnist_dataset', kind='train', onehot= True)
    ratio = 0.2
    size = X.shape[0]
    valid_size = int(size * ratio)
    train_size = size - valid_size
    x_train = X[valid_size:]
    y_train = y[valid_size:]
    x_valid = X[:valid_size]
    y_valid = y[:valid_size]

    result = {}
    for hidden in hidden_size:
        for l2_reg in reg:
            for lr in lrs:
                network = model.TwoLayerNet(input_size=784, hidden_size=hidden, output_size=10, 
                      learning_rate = lr, reg = l2_reg)
                network.lr = lr
                train_dataloader = utils.DataLoader(x_train, y_train, batch_size=256, drop_last=True)
                valid_acc_list = []
                for e in range(max_epoch):
                    for x_batch, y_batch in train_dataloader:
                        loss = network.train(x_batch, y_batch)
                    network.lr *= lr_decay
                    valid_acc = network.accuracy(x_valid, y_valid)
                    valid_acc_list.append(valid_acc)
                    if e >= min_epoch:
                        if valid_acc_list[-1] < 0.7:
                            break
                        if np.all(np.diff(valid_acc_list[decay_num:]) <= 0):
                            break
                result[(hidden, l2_reg, lr)] = np.mean(valid_acc_list[-decay_num:])
                print('epoch:', e+1)
                print((hidden, l2_reg, lr), ' ', result[(hidden, l2_reg, lr)])

    print('-'*50)
    print('best params:', max(zip(result.values(),result.keys())))
    df = pd.concat({k: pd.Series(v) for k, v in result.items()}).reset_index()
    df.columns = ['hidden','reg','lr','n','acc']
    df = df.drop(['n'], axis=1)
    df.to_csv('parameters_select.csv')   