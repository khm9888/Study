def outliners(data):
    q1,q3 = np.percentile(data,[25,75])
    iqr = q3-q1
    upper_bound = q3+iqr*1.5
    lower_bound = q1-iqr*1.5
    return np.where((data>upper_bound) | (data<lower_bound))