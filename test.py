import model

if __name__ == '__main__':
    
    x_test, y_test = model.load_mnist('./mnist_dataset', kind='t10k', onehot= True)

    network = model.TwoLayerNet(input_size=784, hidden_size=512, output_size=10, 
                             learning_rate = 1e-3, reg = 0)
    network.loadmodel('ckpt/model_parameters.npy')

    test_acc = network.accuracy(x_test, y_test)
    test_loss = network.train(x_test, y_test)

    print('The accuracy is: ', test_acc)