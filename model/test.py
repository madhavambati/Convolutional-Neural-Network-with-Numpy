from functions import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pickle

if __name__ == '__main__':

    cost = train()
    save_path = 'params.pkl'
    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    print('   |----------------Parameters loaded from' + save_path + '----------------|')

    # Get test data
    m = 10000
    X = extract_data('test-images.gz', m, 28)
    y_dash = extract_labels('test-labels.gz', m).reshape(m, 1)
    # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))  # divide by standard deviation
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, params)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Overall Accuracy: %.2f" % (float(corr / len(test_data) * 100)))
