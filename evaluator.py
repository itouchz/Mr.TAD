import numpy as np

from tqdm.notebook import tqdm
from sklearn.metrics import auc

def _enumerate_thresholds(rec_errors, n=1000):
    # maximum value of the anomaly score for all time steps in the test data
    thresholds, step_size = [], np.max(rec_errors) / n
    th = 0.
    
    for i in range(n):
        thresholds.append(th)
        th = th + step_size
    
    return thresholds

def _compute_anomaly_scores(x, rec_x, x_val=None, scoring='square_mean'):
    if scoring == 'absolute':
        return np.mean(np.abs(rec_x - x), axis=-1)
    elif scoring == 'square_mean':
        return np.mean(np.square(rec_x - x), axis=-1) # ref. S-RNNs
    elif scoring == 'square_median':
        return np.median(np.square(rec_x - x), axis=-1)
    elif scoring == 'probability':
        return None # ref. RAMED expect to fill in

def evaluate(x, rec_x, labels, is_reconstructed=True, n=1000, scoring='square_mean', x_val=None):
    TP, TN, FP, FN = [], [], [], []
    precision, recall, f1, fpr = [], [], [], []
    
    rec_errors = _compute_anomaly_scores(x, rec_x, scoring) if is_reconstructed else rec_x
    if len(rec_errors.shape) > 2:
        if scoring.split('_')[1] == 'mean':
            rec_errors = np.mean(rec_errors, axis=0)
        else:
            rec_errors = np.median(rec_errors, axis=0)
            
    thresholds = _enumerate_thresholds(rec_errors, n)
    
    for th in thresholds: # for each threshold
        TP_t, TN_t, FP_t, FN_t = 0, 0, 0, 0
        for t in range(len(x)): # for each time window
            # if any part of the segment has an anomaly, we consider it as anomalous sequence

            true_anomalies, pred_anomalies = set(np.where(labels[t] == 1)[0]), set(np.where(rec_errors[t] > th)[0])

            if len(pred_anomalies) > 0 and len(pred_anomalies.intersection(true_anomalies)) > 0:
                # correct prediction (at least partial overlap with true anomalies)
                TP_t = TP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) == 0:
                # correct rejection, no predicted anomaly on no true labels
                TN_t = TN_t + 1 
            elif len(pred_anomalies) > 0 and len(true_anomalies) == 0:
                # false alarm (i.e., predict anomalies on no true labels)
                FP_t = FP_t + 1
            elif len(pred_anomalies) == 0 and len(true_anomalies) > 0:
                # predict no anomaly when there is at least one true anomaly within the seq.
                FN_t = FN_t + 1
        
        TP.append(TP_t)
        TN.append(TN_t)
        FP.append(FP_t)
        FN.append(FN_t)
    
    for i in range(len(thresholds)):
        precision.append(TP[i] / (TP[i] + FP[i] + 0.0000001))
        recall.append(TP[i] / (TP[i] + FN[i] + 0.0000001)) # recall or true positive rate (TPR)
        fpr.append(FP[i] / (FP[i] + TN[i] + 0.0000001))
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 0.0000001))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': auc(recall, precision),
        'roc_auc': auc(fpr, recall),
        'thresholds': thresholds
    }