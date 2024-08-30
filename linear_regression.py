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
    # f(x) = 2x
    # return [[1, 2], [2, 4], [3, 6], [100, 200]]
    # return [[1, 2], [2, 4], [3, 6]]
    # f(x) = 2x + 1
    # return [[1, 3], [2, 5], [3, 7]]
    # f(x) = 2.24*x + 12.6
    return [(37.45401188, 96.41203972), (95.07143064, 225.325133)]


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

    total = 0
    for i in range(m):
        x = X[i]
        y = Y[i]
        y_hat = model_function(x, w, b)
        # hard-coding the partial derivative of the cost function J with respect
        # to w:
        total += (y_hat - y) * x
    return total / m


# evaluates gradient for bias b for a given set of X features and Y targets
def calculate_bias_gradient(X, Y, w, b):

    if len(X) != len(Y):
        raise ValueError("mismatch in features and targets")

    # m is the size of the data set
    m = len(X)

    total = 0
    for i in range(m):
        x = X[i]
        y = Y[i]
        y_hat = model_function(x, w, b)
        # hard-coding the partial derivative of the cost function J with respect
        # to b:
        total += y_hat - y

    return total / m


# returns the mean squared error cost function for given x/y pairs and linear
# model y = wx + b
def calculate_cost(X, Y, w, b):

    if len(X) != len(Y):
        raise ValueError("mismatch in features and targets")

    # m is the size of the data set
    m = len(X)

    total = 0
    for i in range(m):
        x = X[i]
        y = Y[i]
        y_hat = model_function(x, w, b)
        total += (y_hat - y) ** 2

    return total / (2 * m)


def perform_gradient_descent(X, Y, **params):
    w = params.get("w_initial", 1)
    b = params.get("b_initial", 0)
    learning_rate_alpha = params.get("learning_rate_alpha", 0.5)
    threshold_epsilon = params.get("threshold_epsilon", 1e-5)
    max_iterations = params.get("max_iterations", 100)

    # set initial cost that meets loop conditions
    cost = threshold_epsilon + 1

    i = 0
    while cost > threshold_epsilon:
        i += 1
        if i > max_iterations:
            print(f"stopping - {max_iterations} maximum iterations reached")
            break

        # update w first
        weight_update = learning_rate_alpha * calculate_weight_gradient(
            X, Y, w, b
        )
        w = w - weight_update

        # update b only after w
        bias_update = learning_rate_alpha * calculate_bias_gradient(X, Y, w, b)
        b = b - bias_update

        # updating cost to break loop once convergence is reached
        cost = calculate_cost(X, Y, w, b)
        # print(
        #     f"guess {i} -- cost: {cost}, w: {w} ({weight_update}), b: {b} ({bias_update})"
        # )
    else:
        print(
            f"finished - update threshold (episilon) {threshold_epsilon} reached"
        )

    print(f"w {w} and b {b}, cost {cost} calculated in {i} steps")

    return w, b


# get the feature set and extract to unscaled vectors X and Y
training_set = get_training_set()
X, Y = extract_features_targets_from_set(training_set)

# using basic feature scaling (min-max normalization) is required when
# even slightly larger numbers are involved. get the max values of X and Y,
# cache them for unscaling w and b, then scale the vectors.

x_max = max(list(map(abs, X)))
y_max = max(list(map(abs, X)))
X_scaled = [value / x_max for value in X]
Y_scaled = [value / y_max for value in Y]

# use gradient descent to estimate scaled w and b

w_scaled, b_scaled = perform_gradient_descent(
    X_scaled,
    Y_scaled,
    w_initial=0,
    b_initial=0,
    learning_rate_alpha=1,
    threshold_epsilon=2e-5,
    max_iterations=10000,
)

# unscale w and b them given that:
# w = w_scaled * (y_max/x_max)
# b = b_scaled * (y_max)

w_final = w_scaled * y_max / x_max
b_final = b_scaled * y_max

print("w and b are estimated to be:")
print([w_final, b_final])

print("predicted y for x = 10 (should be 2*10 = 20):")
print(model_function(10, w_final, b_final))
