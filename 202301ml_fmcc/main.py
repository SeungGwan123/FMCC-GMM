import numpy as np

def gaussian_mixture_clustering(X, n_components, init_mu=None, init_cov_mat=None, init_weights=None,
                                epsilon=1e-4, max_iter=100, random_state=100):
    from scipy.stats import multivariate_normal
    import random
    # set initial value
    if init_mu is None:
        random.seed(random_state)
        idx = random.sample(range(X.shape[0]), n_components)
        mu = X[idx, :]
    else:
        mu = init_mu

    if init_cov_mat is None:
        np.random.seed(random_state)
        cov_list = []
        for _ in range(n_components):
            arr = np.random.rand(X.shape[1] ** 2) + 0.1
            temp_mat = np.triu(arr.reshape(X.shape[1], X.shape[1]))
            cov_elem = temp_mat.dot(temp_mat.T)
            cov_list.append(cov_elem)

        cov_mat = np.array(cov_list)
    else:
        cov_mat = init_cov_mat

    if init_weights is None:
        weights = np.array([1 / n_components] * n_components)
    else:
        weights = init_weights

    objective_value = -np.infty
    objective_value_history = []
    iteration = 1
    while (iteration < max_iter):
        # E-step
        assign_prob = []
        for i, d in enumerate(X):
            assign_prob_temp = []
            for k in range(n_components):
                assign_prob_temp.append(weights[k] * \
                                        multivariate_normal(mean=mu[k], cov=cov_mat[k]).pdf(d))

            assign_prob_temp = np.array(assign_prob_temp)
            assign_prob_temp = assign_prob_temp / np.sum(assign_prob_temp)
            assign_prob.append(assign_prob_temp)

        assign_prob = np.array(assign_prob)

        # M-step
        temp_sum = np.sum(assign_prob, axis=0)
        next_weights = []
        next_mu = []
        next_cov_mat = []
        for k in range(n_components):
            mu_numerator = np.sum(np.expand_dims(assign_prob[:, k], axis=1) * X, axis=0)
            next_mu_vector = mu_numerator / temp_sum[k]
            next_mu.append(next_mu_vector)
            next_weights.append(temp_sum[k] / X.shape[0])

            t = []
            for i, d in enumerate(X):
                tt = d - next_mu_vector
                tt = np.expand_dims(tt, axis=1)
                tt_cov = np.matmul(tt, tt.transpose())
                tt_term = assign_prob[i][k] * tt_cov
                t.append(tt_term)
            t = np.array(t)
            cov_numerator = np.sum(t, axis=0)
            next_cov_mat.append(cov_numerator / temp_sum[k])

        next_objective_value = 0
        for x in X:
            value = np.log(np.sum([next_weights[k] * multivariate_normal(mean=next_mu[k], cov=next_cov_mat[k]).pdf(x)
                                   for k in range(n_components)]))
            next_objective_value += value
        objective_value_history.append(next_objective_value)
        if next_objective_value - objective_value <= epsilon:
            labels = [np.argmax(x) for x in assign_prob]
            return (labels, iteration, objective_value_history)
        else:
            weights = next_weights
            mu = next_mu
            cov_mat = next_cov_mat
            objective_value = next_objective_value
        iteration += 1
    labels = [np.argmax(x) for x in assign_prob]
    return (labels, iteration, objective_value_history)
