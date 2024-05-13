import numpy as np
import itertools
import pickle
import os


from skmob.measures import evaluation



def get_exp_measures(lista, paired = False, method = "rmse"):
    exp = []

    if paired:
        insieme = lista
    else:
        insieme = itertools.combinations(lista, r =2)



    if method == "cpc":
        misura =  evaluation.common_part_of_commuters
        exp=[]
        for pair in insieme:
            weights_1 = (pair[0]).flatten()
            weights_2 = (pair[1]).flatten()
            m = misura(weights_1, weights_2)
            exp.append(m)
        return exp

    elif method == "rmse":
        misura = evaluation.rmse
        exp=[]
        for pair in insieme:
            weights_1 = (pair[0]).flatten()
            weights_2 = (pair[1]).flatten()
            rmse = misura(weights_1, weights_2)
            #nrmse = rmse/(max(np.max(weights_1),np.max(weights_2)) - min(np.min(weights_1), np.min(weights_2)))
            exp.append(rmse)
        return exp


with open("./BikeNYC/fake_set.txt", "rb") as fp:   
    fake_set = pickle.load(fp)
with open("./BikeNYC/v_test.txt", "rb") as fp:   
    v_test = pickle.load(fp)

# Fixed metrics
metrics = ["cpc", "rmse"]

# Create experiment folder if it does not exist
experiment_folder = "./BikeNYC/experiments"
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)


# Perform experiments for each metric
for metric in metrics:
    print(f"Running experiment for {metric}")
    exp_sim_1 = get_exp_measures(v_test, method=metric)
    exp_sim_2 = get_exp_measures(fake_set, method=metric)
    mixed_set_pairs = list(itertools.product(v_test, fake_set))
    exp_sim_3 = get_exp_measures(mixed_set_pairs, paired=True, method=metric)

    # Save experiment results
    for i, exp_sim in enumerate([exp_sim_1, exp_sim_2, exp_sim_3], start=1):
        folder_path = f"{experiment_folder}/{metric}/MoGAN"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(f"{folder_path}/{i}.txt", "wb") as fp:
            pickle.dump(exp_sim, fp)
