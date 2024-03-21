import pickle

import numpy as np
import pymc as pm
from tqdm.contrib.concurrent import process_map
import cloudpickle


def main():

    # load the binary variables
    with open("P.pickle", "rb") as f:
        P = pickle.load(f)

    with open("W.pickle", "rb") as f:
        W = pickle.load(f)

    with open("T.pickle", "rb") as f:
        T = pickle.load(f)

    # get permutations of W, P, T values for which to conduct an analysis
    perms = []
    w_values = W["poor"]
    p_values = P["black"]
    for t_name, t_values in T.items():
        perms.append((t_name, p_values, w_values, t_values))
        # perms.append((t_name, p_values[:10000], w_values[:10000], t_values[:10000]))

    # runs the analysis in parallel
    process_map(estimator_star, perms)


def estimator_star(t):
    return bernoulli_estimator(*t)


def bernoulli_estimator(name, G, W, T):
    with pm.Model() as model:

        pW = pm.LogitNormal("P(Wi==1)", 0, 1.5)
        pG = pm.LogitNormal("P(Gi==1)", 0, 1.5)

        # note: these c,w,g parameters are not the same as the simulator, as they as subject to expit()
        c = pm.Normal("c", 0, 5)
        w = pm.Normal("w", 0, 5)
        g = pm.Normal("g", 0, 5)

        W_ = pm.Bernoulli("Wi", p=pW, observed=W)
        G_ = pm.Bernoulli("Gi", p=pG, observed=G)

        logit_pT = c + w * W_ + g * G_
        pm.Bernoulli("Ti", logit_p=logit_pT, observed=T)

        trace = pm.sample()

        # save
        dict_to_save = {'model':model, 'trace':trace}
        with open(f'S1_{name}.save', 'wb') as f:
            cloudpickle.dump(dict_to_save, f)
        

if __name__ == "__main__":
    main()
