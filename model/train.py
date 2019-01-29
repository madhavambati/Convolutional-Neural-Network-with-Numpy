from functions import *
from Build_Optimize_network import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pickle




def train(num_classes=10, alpha=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1,
          f=5, num_filt1=8, num_filt2=8, batch_size=32, num_epochs=2, save_path='params.pkl'):
    m = 50000  # No.of images in the test set
    X = extract_data('train-images.gz', m, img_dim)
    y = extract_labels('train-labels.gz', m).reshape(m, 1)

    X -= int(np.mean(X))
    X /= int(np.std(X))
    train_data = np.hstack((X, y))
    print('   training_data shape:' + str(train_data.shape))
    np.random.shuffle(train_data)

    f1, f2, w3, w4 = (num_filt1, img_depth, f, f), (num_filt2, num_filt1, f, f), (128, 800), (10, 128)
    f1 = Filter_weights(f1)
    f2 = Filter_weights(f2)
    w3 = deep_weights(w3)
    w4 = deep_weights(w4)

    b1 = np.zeros((f1.shape[0], 1))
    b2 = np.zeros((f2.shape[0], 1))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    cost_array = []

    print('   |---------------Batching the train data---------------|')
    print("LR:" + str(alpha) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        print('   |----------------initiating epoch - 0' + str(epoch + 1) + '----------------|')
        print('   |-----------------deployed epoch - 0' + str(epoch + 1) + '----------------|')
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            params, cost_array = optimize_network(batch, num_classes, alpha, img_dim, img_depth, beta1, beta2, params,
                                                  cost_array, E=1e-7)
            t.set_description("Cost: %.2f" % (cost_array[-1]))

    save = [params, cost_array]
    with open(save_path, 'wb') as file:
        pickle.dump(save, file)

    return cost_array


if __name__ == '__main__':

    cost = train()

    print('   |----------------Network trained succesfully----------------|')
    params, cost = pickle.load(open('params.pkl', 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params


    print('   |----------------parameters saved in' + save_path + '----------------|')

    # Plot cost
    plt.plot(cost, 'b')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()