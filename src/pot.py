import numpy as np
import os
from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *
import matplotlib.pyplot as plt
import scikitplot as skplt

styles=['crimson','orange','gold','mediumseagreen','steelblue','mediumpurple','aliceblue','antiquewhite','aqua',
        'aquamarine','azure','beige','bisque','black','blanchedalmond','blue','blueviolet','brown','burlywood',
        'cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','cyan','darkblue',
        'darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange',
        'darkorchid']


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t
        name:
    Returns:
        dict: pot result dict
    """


    lms = lm[0]
    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(level=lms, min_extrema=False, verbose=False)  # initialization step
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run
    # print(len(ret['alarms']))
    # print(len(ret['thresholds']))
    pot_th = np.mean(ret['thresholds']) * lm[1]
    # pot_th = np.percentile(score, 100 * lm[0])
    # np.percentile(score, 100 * lm[0])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    # DEBUG - np.save(f'{debug}.npy', np.array(pred))
    # DEBUG - print(np.argwhere(np.array(pred)))
    p_t = calc_point2point(pred, label)
    # print('POT result: ', p_t, pot_th, p_latency)

    # 绘制“ROC曲线”
    # fpr, tpr, roc_auc = CalculateROCAUCMetrics(label, pred)
    # PlotROCAUC(name,fpr, tpr, roc_auc)
    # 绘制“精度召回曲线”
    # precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(label, pred)
    # PlotPrecisionRecallCurve(name,precision_curve, recall_curve, average_precision)

    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
        # 'pot-latency': p_latency
    }, np.array(pred)


def CalculateROCAUCMetrics(_abnormal_label, _score):
    fpr, tpr, _ = roc_curve(_abnormal_label, _score)
    roc_auc = auc(np.nan_to_num(fpr), np.nan_to_num(tpr))
    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc
    return fpr, tpr, roc_auc

def newPlotROCAUC(name, y_test, y_probas):
    '''
    绘制 ROC曲线 和 精度召回曲线
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''

    skplt.metrics.plot_roc(y_test, y_probas, title=f'{name}-Receiver Operating Characteristic', figsize=(7, 7),
                           # title_fontsize = 24, text_fontsize = 16
                           )
    plt.savefig(f'plots/{name}/{name}-ROC-Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制“精度召回曲线”
    skplt.metrics.plot_precision_recall(y_test, y_probas, title=f'{name}-Precision-Recall Curve', figsize=(7, 7),
                                        # title_fontsize = 24, text_fontsize = 16
                                        )
    plt.savefig(f'plots/{name}/{name}-PlotPrecisionRecallCurve.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

def PlotROCAUC1(name,i,_fpr, _tpr, _roc_auc):
    plt.figure()
    lw = 1.5
    # plt.plot(_fpr, _tpr, color=styles[i], lw=lw, label='ROC curve of class %d (area = %0.3f)' % (i,_roc_auc))
    plt.plot(_fpr, _tpr, lw=lw, label='ROC curve of class %d (area = %0.3f)' % (1,_roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name}-Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/{name}/{name}-ROC-Curves.png')

    # plt.show()
    # plt.close()

def PlotROCAUC(name,_fpr, _tpr, _roc_auc):
# def PlotROCAUC(name,i,_fpr, _tpr, _roc_auc):
    plt.figure()
    lw = 1.5
    # plt.plot(_fpr, _tpr, color=styles[i], lw=lw, label='ROC curve of class %d (area = %0.3f)' % (i,_roc_auc))
    plt.plot(_fpr, _tpr, lw=lw, label='ROC curve of class %d (area = %0.3f)' % (1,_roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name}-Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'plots/{name}/{name}-ROC-Curves.png')

    # plt.show()
    # plt.close()

def CalculatePrecisionRecallCurve(_abnormal_label, _score):
    precision_curve, recall_curve, _ = precision_recall_curve(_abnormal_label, _score)
    average_precision = average_precision_score(_abnormal_label, _score)
    if average_precision < 0.5:
        average_precision = 1 - average_precision
    return precision_curve, recall_curve, average_precision

# def PlotPrecisionRecallCurve(name,i,_precision, _recall, _average_precision):
def PlotPrecisionRecallCurve(name,_precision, _recall, _average_precision):
    plt.figure()
    lw = 2
    plt.step(_recall, _precision, lw=lw, alpha=1, where='post', label='PR curve of class %d (AUC = %0.3f)' % (1,_average_precision))
    # plt.step(_recall, _precision, color=styles[int(i)], lw=lw, alpha=1, where='post', label='PR curve of class %d (AUC = %0.3f)' % (i,_average_precision))
    # plt.fill_between(_recall, _precision, step='post', alpha=0.2, color='b')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{name}-Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.legend('AP={0:0.2f}'.format(_average_precision))
    plt.legend(loc="lower right")
    plt.savefig(f'plots/{name}/{name}-PlotPrecisionRecallCurve.png')

    # plt.show()
    plt.close()


