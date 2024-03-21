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
    p_values = P["derogatory"]
    # for t_name, t_values in T.items():
    for t_name in ['heroin']:
        t_values = T[t_name]
        perms.append((t_name, p_values, w_values, t_values))
        # perms.append((t_name, p_values[:10000], w_values[:10000], t_values[:10000]))

    # runs the analysis in parallel
    process_map(estimator_star, perms)


def estimator_star(t):
    return bernoulli_estimator(*t)


def bernoulli_estimator(name, G, W, T):

    with pm.Model() as model:

        pG = pm.LogitNormal('P(Gi==1)', 0, 1.5)
        G_ = pm.Bernoulli('Gi', p=pG, observed=G)
        
        w1 = pm.Normal('w1', 0, 5)
        w2 = pm.Normal('w2', 0, 5)
        pW = pm.math.invlogit(w1 + w2*G_)
        W_ = pm.Bernoulli("Wi", p=pW, observed=W)

        t1 = pm.Normal('t1', 0, 5) 
        t2 = pm.Normal('t2', 0, 5)
        t3 = pm.Normal('t3', 0, 5)
        pT = pm.math.invlogit(t1 + t2*W_ + t3*G_)
        pm.Bernoulli("Ti", p=pT, observed=T)
        
        trace=pm.sample()

        # save
        dict_to_save = {'model':model, 'trace':trace}
        with open(f'S2_{name}.save', 'wb') as f:
            cloudpickle.dump(dict_to_save, f)
        

if __name__ == "__main__":
    main()
