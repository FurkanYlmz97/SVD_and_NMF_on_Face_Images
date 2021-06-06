__Author__: "Furkan Yılmaz"
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import copy
import time


def readData(file_location):
    X = []
    for file in os.listdir(file_location):
        if file.endswith(".pgm"):
            path = os.path.join(file_location, file)
            img = cv.imread(path, cv.IMREAD_GRAYSCALE).flatten()
            # reader = Reader()
            # img = reader.read_pgm(path)
            X.append(img)

    X = np.array(X)
    X = X.transpose()
    return X


def computeEnergy(sigmas):

    energy = []
    dum = 0
    for sigma in sigmas:
        dum += sigma**2
        energy.append(dum)
    return np.array(energy)


def find_indices(energy_n, level):

    ind = 0
    for energy in energy_n:
        ind += 1
        if energy >= level:
            return ind

    return "Error"


def generate_pic(U, s, V, ind):

    gen_img = 0

    for i in range(int(ind)):
        dum = np.outer(U[:, i], V[i, :]) * s[i]
        gen_img += dum
    return gen_img


def hals(X, W, H):
    _, dum = W.shape
    for l in range(dum):
        sum = 0
        for k in range(dum):
            if k != l:
                sum += W[:, k] * np.dot(H[k, :], H[l, :].T)
        # W[:, l] = (X @ H[l, :].T - sum) / (np.linalg.norm(H[l, :]) * np.linalg.norm(H[l, :]))
        # W[:, l] = np.maximum(W[:, l], 1e-16)
        W[:, l] = (X @ H[l, :].T - sum)
        if (np.linalg.norm(H[l, :]) * np.linalg.norm(H[l, :])) > 0:
            W[:, l] = W[:, l] / (np.linalg.norm(H[l, :]) * np.linalg.norm(H[l, :]))
        W[:, l] = np.maximum(W[:, l], 0)
    return W


def two_block_coordinate_descent(X, r, const=0.01, max_iter=500):

    n, m = X.shape
    W = np.random.rand(n, r)
    H = np.random.rand(r, m)

    X = X / 255
    U, s, V = np.linalg.svd(X)
    # s = s / max(s)

    for k in range(r):
        uk_positive = copy.deepcopy(U[:, k])
        uk_positive = np.maximum(uk_positive, 0)

        uk_negative = copy.deepcopy(U[:, k])
        uk_negative = np.maximum(-uk_negative, 0)

        vk_positive = copy.deepcopy(V[:, k])
        vk_positive = np.maximum(vk_positive, 0)

        vk_negative = copy.deepcopy(V[:, k])
        vk_negative = np.maximum(-vk_negative, 0)

        M_plus = (np.outer(uk_positive, vk_positive))
        M_negative = (np.outer(uk_negative, vk_negative))

        if np.linalg.norm(M_plus) >= np.linalg.norm(M_negative):
            W[:, k] = uk_positive / np.linalg.norm(uk_positive)
            H[k, :] = s[k] * np.linalg.norm(uk_positive) * vk_positive
        else:
            W[:, k] = uk_negative / np.linalg.norm(uk_negative)
            H[k, :] = s[k] * np.linalg.norm(uk_negative) * vk_negative

    error = []
    # error.append(np.linalg.norm(X - W @ H, ord='fro') / np.linalg.norm(X, ord='fro'))
    error.append(np.linalg.norm(X - W @ H, ord='fro'))

    w0 = copy.deepcopy(W)
    w_old = copy.deepcopy(W)

    w1 = copy.deepcopy(hals(X, W, H))
    thr = const * np.linalg.norm(w1 - w0, ord='fro')

    W = hals(X, W, H)
    w_new = copy.deepcopy(W)
    H = hals(X.T, H.T, W.T).T
    # error.append(np.linalg.norm(X - W @ H, ord='fro') / np.linalg.norm(X, ord='fro'))
    error.append(np.linalg.norm(X - W @ H, ord='fro'))
    iter = 0

    while iter < max_iter:
        iter += 1
        # print(np.linalg.norm(w_new - w_old, ord='fro') - thr)
        if np.linalg.norm(w_new - w_old, ord='fro') <= thr:
            # print("Threshold reached")
            break
        w_old = copy.deepcopy(W)
        w_new = hals(X, W, H)
        W = w_new
        H = hals(X.T, H.T, W.T).T
        # cwh = compute_cwh(X, W, H)
        # error.append(np.linalg.norm(X - W@H, ord='fro') / np.linalg.norm(X))
        error.append(np.linalg.norm(X - W @ H, ord='fro'))

    return W, H, error


def SVD_reconstruct(X, std, index=False):

    # From the book: Hands on Machine Learning with Scikit-Learn and Tensorflow
    U, s, V = np.linalg.svd(X)

    # Compute Energy & plot
    energy = computeEnergy(s)
    energy_normalized = energy / max(energy)
    # Compute Energy & plot

    # Find indices
    if index is False:
        index = find_indices(energy_normalized, std)
    gen_img = generate_pic(U, s, V, index)
    return gen_img


def NMF_reconstruct(X, r):

    W, H, error = two_block_coordinate_descent(X, int(r))
    return W@H


def Image_recovery(X, Y, Y_noise, r_end):

    # From the book: Hands on Machine Learning with Scikit-Learn and Tensorflow
    U, s, V = np.linalg.svd(X)

    error_svd = []
    error_nmf = []

    y__noise = []
    for m in range(Y_noise.shape[1]):
        y__noise.append(Y_noise[:, m])

    ranks = np.arange(start=1, step=1, stop=r_end)

    for r in ranks:

        print("R = " + str(r))
        # SVD
        # svd_constructed = []
        # for k in range(len(y__noise)):
        #     sum = 0
        #     for p in range(r):
        #         sum += np.dot(y__noise[k], U[:, p]) * U[:, p]
        #     svd_constructed.append(sum)
        # svd_constructed = np.array(svd_constructed).T
        svd_constructed = U[:, 0:r]@U[:, 0:r].T@Y_noise  # To accelarate I did not use vectors
        error_svd.append(np.mean(np.linalg.norm(Y - svd_constructed, axis=0)))
        # SVD

        # NMF
        # nmf_constructed = []
        W, H, error = two_block_coordinate_descent(X, int(r))
        # for k in range(len(y__noise)):
        #     sum = 0
        #     for p in range(r):
        #         sum += (np.linalg.pinv(W.T@W)@W.T @ y__noise[k])[p] * W[:, p]
        #     nmf_constructed.append(sum)
        # nmf_constructed = np.array(nmf_constructed).T
        nmf_constructed = W@np.linalg.pinv(W.T@W)@W.T@Y_noise  # To accelarate I did not use vectors
        error_nmf.append(np.mean(np.linalg.norm(Y - nmf_constructed, axis=0)))
        # NMF

        # fig, ax = plt.subplots(1, 3, figsize=(16, 8))
        # ax = ax.flatten()
        # ax[0].imshow(np.reshape(Y[:, 0], (19, 19)), cmap='gray')
        # ax[0].set_title("Original Image")
        # ax[1].imshow(np.reshape(svd_constructed[:, 0], (19, 19)), cmap='gray')
        # ax[1].set_title("svd Image")
        # ax[2].imshow(np.reshape(nmf_constructed[:, 0], (19, 19)), cmap='gray')
        # ax[2].set_title("nmf  Image")
        # plt.show()
        # print()

    return error_svd, error_nmf


if __name__ == '__main__':

    # Todo: 1.1
    X = readData("Dataset\\train")

    # From the book: Hands on Machine Learning with Scikit-Learn and Tensorflow
    U, s, V = np.linalg.svd(X)
    # From the book: Hands on Machine Learning with Scikit-Learn and Tensorflow

    # Plot singular values
    plt.title("Singular Values Plot")
    plt.xlabel("Singular Values")
    plt.ylabel("Values of Singular Values")
    plt.plot(s)
    plt.show()
    # Plot singular values

    # Compute Energy & plot
    plt.title("Energy Plot")
    energy = computeEnergy(s)
    energy_normalized = energy / max(energy)
    plt.xlabel("Accumulated Energy")
    plt.ylabel("Values of Accumulated Energy")
    plt.plot(energy_normalized)
    plt.show()
    # Compute Energy & plot

    # Find indices
    I9 = find_indices(energy_normalized, 0.9)
    I95 = find_indices(energy_normalized, 0.95)
    I99 = find_indices(energy_normalized, 0.99)
    # Find indices

    gen_img = generate_pic(U, s, V, I99)
    orginal_image = np.reshape(X[:, 0], (19, 19))
    generated_image = np.reshape(gen_img[:, 0], (19, 19))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].imshow(orginal_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(generated_image, cmap='gray')
    ax[1].set_title("Reconstructed Image with 0.99")
    plt.show()

    gen_img = generate_pic(U, s, V, I95)
    orginal_image = np.reshape(X[:, 0], (19, 19))
    generated_image = np.reshape(gen_img[:, 0], (19, 19))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].imshow(orginal_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(generated_image, cmap='gray')
    ax[1].set_title("Reconstructed Image with 0.95")
    plt.show()

    gen_img = generate_pic(U, s, V, I9)
    orginal_image = np.reshape(X[:, 0], (19, 19))
    generated_image = np.reshape(gen_img[:, 0], (19, 19))
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].imshow(orginal_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(generated_image, cmap='gray')
    ax[1].set_title("Reconstructed Image with 0.9")
    plt.show()

    # Lets plot first 9 features
    # fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    # ax = ax.flatten()
    # for i in range(9):
    #     ax[i].imshow(np.reshape(U[:, i], [19, 19]), cmap='gray')
    #     ax[i].set_title(str(i+1) + "th Column of U")
    # plt.show()

    plt.imshow(np.reshape(U[:, 0], [19, 19]), cmap='gray')
    plt.title("First Column of U")
    plt.show()
    # Todo: 1.1


    # Todo: 1.2
    X = readData("Dataset//train")

    start_time = time.time()
    W, H, error = two_block_coordinate_descent(X, 45, const=0.01, max_iter=100)
    print("--- %s seconds ---" % (time.time() - start_time))

    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("||X-WH||_f Frobenius Norm of the Error")
    plt.show()

    X_new = W@H
    orginal_image = np.reshape(X[:, 0], (19, 19))
    generated_image = np.reshape(X_new[:, 0], (19, 19))

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].imshow(orginal_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(generated_image, cmap='gray')
    ax[1].set_title("Reconstructed Image")
    plt.show()

    # Lets plot first 9 features
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(9):
        ax[i].imshow(np.reshape(W[:, i], [19, 19]), cmap='gray')
        ax[i].set_title(str(i+1) + "th Column of W")
    plt.show()
    # Todo: 1.2


    # Todo: 2
    X = readData("Dataset//train")
    Y = readData("Dataset//test")

    Y1 = Y + np.random.rand(Y.shape[0], Y.shape[1])
    Y1[Y1 > 255] = 255

    Y10 = Y + 10 * np.random.rand(Y.shape[0], Y.shape[1])
    Y10[Y10 > 255] = 255

    Y25 = Y + 25 * np.random.rand(Y.shape[0], Y.shape[1])
    Y25[Y25 > 255] = 255

    # Plot Noisy Pics
    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].imshow(np.reshape(Y[:, 0], (19, 19)), cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(np.reshape(Y1[:, 0], (19, 19)), cmap='gray')
    ax[1].set_title("Rand*1 Image")
    ax[2].imshow(np.reshape(Y10[:, 0], (19, 19)), cmap='gray')
    ax[2].set_title("Rand*10  Image")
    ax[3].imshow(np.reshape(Y25[:, 0], (19, 19)), cmap='gray')
    ax[3].set_title("Rand*25  Image")
    plt.show()

    error_svd, error_nmf = Image_recovery(X, Y, Y1, 100)
    plt.plot(error_svd, label='SVD Error')
    plt.plot(error_nmf, label='NMF Error')
    plt.title("Error vs RSVD & RNMF for Noise Rand*1")
    plt.xlabel("Rsvd & Rnmf")
    plt.ylabel("F of Error for SVD & NMF")
    plt.legend()
    plt.show()
    np.save('svdy1.npy', error_svd)
    np.save('nmfy1.npy', error_nmf)

    error_svd, error_nmf = Image_recovery(X, Y, Y10, 100)
    plt.plot(error_svd, label='SVD Error')
    plt.plot(error_nmf, label='NMF Error')
    plt.title("Error vs RSVD & RNMF for Noise Rand*10")
    plt.xlabel("Rsvd & Rnmf")
    plt.ylabel("F of Error for SVD & NMF")
    plt.legend()
    plt.show()
    np.save('svdy10.npy', error_svd)
    np.save('nmfy10.npy', error_nmf)

    error_svd, error_nmf = Image_recovery(X, Y, Y25, 100)
    plt.plot(error_svd, label='SVD Error')
    plt.plot(error_nmf, label='NMF Error')
    plt.title("Error vs RSVD & RNMF for Noise Rand*25")
    plt.xlabel("Rsvd & Rnmf")
    plt.ylabel("F of Error for SVD & NMF")
    plt.legend()
    plt.show()
    np.save('svdy25.npy', error_svd)
    np.save('nmfy25.npy', error_nmf)
    # Todo: 2

    # Todo: 3
    S = np.ones((19, 19))
    for i in range(19):
        for j in range(19):
            if j <= 10:
                S[i, j] = 1
            else:
                S[i, j] = 1 - 0.05 * (j - 10)
    S = S.flatten()
    S = np.array([S, ] * 472).T

    Y = readData("Dataset//test")
    X = readData("Dataset//train")
    Y_masked = np.multiply(Y, S)

    # Plot Noisy Pics
    fig, ax = plt.subplots(2, 2, figsize=(16, 8))
    ax = ax.flatten()
    ax[0].imshow(np.reshape(Y[:, 2], (19, 19)), cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(np.reshape(Y_masked[:, 2], (19, 19)), cmap='gray')
    ax[1].set_title("Masked Image")
    ax[2].imshow(np.reshape(Y[:, 1], (19, 19)), cmap='gray')
    ax[2].set_title("Original Image")
    ax[3].imshow(np.reshape(Y_masked[:, 1], (19, 19)), cmap='gray')
    ax[3].set_title("Masked Image")
    plt.show()

    error_svd, error_nmf = Image_recovery(X, Y, Y_masked, 100)
    plt.plot(error_svd, label='SVD Error')
    plt.plot(error_nmf, label='NMF Error')
    plt.title("Error vs RSVD & RNMF for Masked Set")
    plt.xlabel("Rsvd & Rnmf")
    plt.ylabel("F of Error for SVD & NMF")
    plt.legend()
    plt.show()
    np.save('svdymasked.npy', error_svd)
    np.save('nmfymasked.npy', error_nmf)
    # Todo: 3
