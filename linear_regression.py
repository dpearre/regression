# exploring and practicing gradient descent concepts in native python with
# minimal libraries.

# gradient descent:
# repeat until convergence {
#     w = w - alpha * dJ(w,b)/dw
#     b = b - alpha * dJ(w,b)/db
# }
# w: weight
# b: bias
# alpha: learning rate
# J(w,b): cost function (will use MSE)
# dJ(w,b)/dw: partial derivative of cost function with respect to weight
# dJ(w,b)/db: partial derivative of cost function with respect to bias


# make a simple training set that should be easy to solve, f(x) = 2x.
def get_training_set():
    # return x/y pairs as individual vectors.
    return [[1, 2], [2, 4], [3, 6]]


# extract features (x) and targets (y)
def extract_features_targets_from_set(training_set):
    features_x = [pair[0] for pair in training_set]
    targets_y = [pair[1] for pair in training_set]
    return features_x, targets_y


# simple model returns the predicted value of y = wx + b
def model_function(x, w, b):
    return w * x + b


# evaluates gradient for weight w for a given set of X features and Y targets
def calculate_weight_gradient(X, Y, w, b):

    if len(X) != len(Y):
        raise ValueError("mismatch in features and targets")

    # m is the size of the data set
    m = len(X)

    gradient = 0
    for i in range(0, m):
        x = X[i]
        y = Y[i]
        y_hat = model_function(X[i], w, b)
        # hard-coding the partial derivative of the cost function J with respect
        # to w:
        gradient += (y_hat - y) * x

    gradient = gradient / m

    return gradient


def perform_gradient_decent(X, Y, **params):
    w = params.get("w_initial", 1)
    b = 0  # not worrying about b for now
    alpha = params.get("alpha", 0.5)
    threshold = params.get("threshold", 1e-5)
    max_iterations = params.get("threshold", 1e3)

    # set an initial gradient update that will meet the loop conditions.
    gradient_update = threshold + 1

    i = 0
    while abs(gradient_update) > threshold:
        i += 1
        if i > max_iterations:
            print(f"stopping - {max_iterations} maximum iterations reached")
            break
        gradient_update = alpha * calculate_weight_gradient(X, Y, w, b)
        w = w - gradient_update
        print(f"w (slope) guess {i}: {w}; gradient update: {gradient_update}")
    else:
        print(f"finished - minimum update threshold {threshold} reached")

    print(f"w {w} calculated in {i} steps")

    return w, b


training_set = get_training_set()
X, Y = extract_features_targets_from_set(training_set)

# try the algorithm...
w_final, b_final = perform_gradient_decent(
    X,
    Y,
    w_initial=99,
    alpha=0.22,
)

print("w and b are estimated to be:")
print([w_final, b_final])

print("predicted y for x = 10 (should be 2*10 = 20):")
print(model_function(10, w_final, b_final))
