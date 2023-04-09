
import numpy as np
import matplotlib.pyplot as plt
import model 
import utils

if __name__ == '__main__':
    np.random.seed(7)
    file = 'ckpt/model_parameters'
    
    x_train, y_train = utils.load_mnist('./mnist_dataset', kind='train', onehot= True)
    x_test, y_test = utils.load_mnist('./mnist_dataset', kind='t10k', onehot= True)

    train_dataloader = utils.DataLoader(x_train, y_train, batch_size=256, drop_last=True)

    network = model.TwoLayerNet(input_size=784, hidden_size=512, output_size=10, 
                             learning_rate = 1e-3, reg = 0)
    
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    epochs = 60
    batch_size = 256
    learning_rate_decay = 0.98
    train_size = x_train.shape[0]

    for e in range(epochs):
        print('epoch:', e + 1)

        for x_train_batch, y_train_batch in train_dataloader:
            loss = network.train(x_train_batch, y_train_batch)
            train_loss_list.append(loss)

        network.lr *= learning_rate_decay
        test_loss = network.train(x_test, y_test)
        test_loss_list.append(test_loss)

        train_acc = network.accuracy(x_train, y_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_acc_list.append(test_acc)

        print('train accuracy: %f, test accuracy: %f' % (train_acc, test_acc))

    # save model
    network.savemodel(file)

    # visualize (Train, test)---loss
    x = np.linspace(1, epochs, epochs)
    plt.plot(x, train_loss_list[10:len(train_loss_list):234], label='train loss')
    plt.plot(x, test_loss_list, label='test loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig('img/loss.png')

    plt.clf()

    # visualize (Train, test)---accuracy
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train accuracy')
    plt.plot(x, test_acc_list, label='test accuracy')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    #plt.show()
    plt.savefig('img/accuracy.png')

    # visualize parameters of each layer
    w1 = network.params['W1']
    w2 = network.params['W2']
    b1 = network.params['b1']
    b1 = b1.reshape(b1.shape[0],-1)
    b1 = np.repeat(b1, b1.shape[0], axis = 1)
    b2 = network.params['b2']
    b2 = b2.reshape(b2.shape[0],-1)
    b2 = np.repeat(b2, b2.shape[0], axis = 1)

    fig = plt.figure()
    # draw 2*2 figure, start from #1
    ax1 = fig.add_subplot(221)
    ax1.imshow(w1)
    ax1.set_title('W1')
    plt.xticks([]), plt.yticks([])
    ax2 = fig.add_subplot(222)
    ax2.imshow(b1)
    ax2.set_title('b1')
    plt.xticks([]), plt.yticks([])
    ax3 = fig.add_subplot(223)
    ax3.imshow(w2)
    ax3.set_title('W2')
    plt.xticks([]), plt.yticks([])
    ax4 = fig.add_subplot(224)
    ax4.imshow(b2)
    ax4.set_title('b2')
    plt.xticks([]), plt.yticks([])
     
    #plt.show()
    plt.savefig('img/parameters.png')