import numpy as np

from utils.getbestF1 import get_bestF1

def evaluate(test_score, test_label=None, PA=True):
    res = {
        'test_score': test_score,
        'test_label': test_label,
    }

    if test_label is not None:
        res_ent = get_bestF1(test_label.copy(), test_score.copy(), True)

        test_pred = (test_score > res_ent['threshold']).astype(int)

        res['precision'] = res_ent['precision']
        res['recall'] = res_ent['recall']
        res['f1_score'] = res_ent['f1_score']
        res['test_pred'] = test_pred
        res['threshold'] = res_ent['threshold']
        res['true_anomaly'] = res_ent['true_anomaly']
        res['pred_anomaly'] = res_ent['pred_anomaly']
        res['anomaly_ratio'] = res_ent['anomaly_ratio']

    return res
