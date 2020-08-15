import pandas as pd
import numpy as np
import seaborn as sns
import pystan
import matplotlib.pyplot as plt
from pandas import Timestamp


def RunModel(df: pd.DataFrame, ts: Timestamp, model: pystan.model.StanModel, model_type: str, 
    prior_alpha, prior_beta):
    """
    Parameters:
    df: input dataframe. 
    ts: date
    model: StanModel
    model_type: 'conversion' or 'revenue'
    
    Return:
    Result
    """
    
    # pull out data before ts
    dt = pd.to_datetime(df['participation_timestamp_rounded_up'])
    idx = dt < Timestamp(ts)
    subxp = df.loc[idx]

    y = subxp['is_converted'].values.astype(np.int64)
    converted = subxp[subxp['is_converted'] ==1]
    
    if(model_type == 'conversion'):
        # Point estimates:
        print(y.sum()/len(y))
      
        # use an uninformed prior for lambda, uniform on the interval [0, 1]
        dat = {
            'observations': len(y),
            'observation_conversion_count': y.sum(),
            'prior_alpha' :  prior_alpha,
            'prior_beta': prior_beta
        }
 
    elif(model_type == 'revenue'):
        r = converted['data'].values

        # Point estimates:
        print(y.sum()/len(y), r.mean(), r.sum()/len(y))

        # use an uninformed prior for lambda, uniform on the interval [0, 1]
        dat = {
            'N': len(y),
            'C': y.sum(),
            'prior_alpha' : 1,
            'prior_beta': 1,
            'r': r
        }

    fit= model.sampling(data=dat, iter=1000, chains=4)
    
    res =  fit.extract(permuted=True)
    return res

def calculate_expected_loss(post_a,  post_b, variant, model_type):
    """
    post_a : StanResults
        Results of MCMC for variant a
    post_b : StanResults
        Results of MCMC for variant b
    variant : int
        Choosen class
    """
    if model_type == 'conversion':
        col_name = 'theta'
    if model_type == 'revenue':
        col_name = 'gtv_per_participant'
    
    a = post_a[col_name]  # est control
    b = post_b[col_name]  # est bucket
        
    if variant == 0: # obs control
        return np.mean(np.maximum(b - a, 0))
    if variant == 1: # obs bucket
        return np.mean(np.maximum(a - b, 0))
    raise ValueError(f'{variant} is misspecified')  


def visualize_joint_posterior(post_a, post_b, model_type):
    """
    post_a : StanResults
            Results of MCMC for variant a
    post_b : StanResults
            Results of MCMC for variant b
    model_type: str
            "conversion" or "revenue"
    """
    if model_type == 'conversion':
        col_name = 'theta'
    if model_type == 'revenue':
        col_name = 'gtv_per_participant'
    
    x1 = post_a[col_name]  # est control
    x2 = post_b[col_name]  # est bucket
    
    qrts1 = np.percentile(x1, [5, 25, 50, 75, 95])
    qrts2 = np.percentile(x2, [5, 25, 50, 75, 95])
    print(f'Bucket 1 quartiles: {qrts1}')
    print(f'Bucket 2 quartiles: {qrts2}')
    lmin = min(np.min(x1), np.min(x2))
    lmax = min(np.max(x1), np.max(x2))
 
    # Show the joint distribution using kernel density estimation
    g = sns.jointplot(x1, x2, kind="kde", height=7, space=0)
    g.ax_joint.plot(np.linspace(lmin,lmax),np.linspace(lmin,lmax))
    g.ax_joint.set(xlabel='posterior for variant 1 bucket', ylabel='posterior for variant 2 bucket')


def visualize_loss(stats, date_range, error_thres):
    closs, tloss = zip(*stats)
    fig = plt.figure(figsize = (8,8))
    x = np.arange(len(date_range))
    plt.plot(closs, label='Choosing variant1')
    plt.plot(tloss, label='Chossing variant2')
    _ = plt.xticks(x, date_range, rotation=45)
    plt.ylabel('Expected loss')
    plt.legend()
    plt.hlines(error_thres, xmin=0, xmax=len(date_range)-1, color='r', linestyles='dotted')

def calc_stats_between_buckets_for_column(
    df: pd.DataFrame, ts: Timestamp, model:  pystan.model.StanModel, 
    model_type: str, return_results: bool=False, prior_alpha=1, prior_beta=1):
    """
    
    """
    unique_buckets = df.bucket.unique()

    results = []
    for bucket in unique_buckets:
        bucket_df = df.loc[df["bucket"] == bucket]
        res_bucket = RunModel(bucket_df, ts, model, model_type, prior_alpha, prior_beta)
        results.append(res_bucket)
        del res_bucket
        
    if len(results)==2:
        closs = calculate_expected_loss(results[0], results[1], 0, model_type)
        tloss = calculate_expected_loss(results[0], results[1], 1, model_type)
        print(f'loss if choosing variant1: {closs}')
        print(f'loss if choosing variant2: {tloss}')
        
        if return_results is True:
            visualize_joint_posterior(results[0], results[1], model_type)
    
    else:
        print('This tool currently does not support more than 2 variants. Please come back later')
        
            
    return closs, tloss


