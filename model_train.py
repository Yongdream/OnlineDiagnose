from PySide2.QtCore import Qt
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QTableWidgetItem

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from pf_main import *


def set_layer(layer : nn.Module, freeze):
    if freeze:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True

def model_training(ui_pa_tw_result, model1,data1,epoch1,model_training):

    args.model = model1
    args.dataset = data1
    if model_training =='--retrain':
        args.retrain = model_training
    else:
        args.test='--test'

    epochs =epoch1


    train_loader, test_loader, labels = load_dataset(args.dataset)   # 加载数据(训练数据、测试数据、测试标签)

    if args.model in ['MERLIN']:
        eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')

    if args.model == 'TPAD':
        model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, train_loader.dataset.train.shape[1]) #特征维度=28
    else:
        model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    ## Prepare data
    if args.dataset == 'BATTERY':
        trainD, testD = train_loader, test_loader
        trainO, testO = trainD, testD
    else:
        trainD, testD = next(iter(train_loader)), next(iter(test_loader)) # 迭代数据集获取训练与测试数据(从不同的文件中分别读取数据)
        trainO, testO = trainD, testD
        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT',
                          'MAD_GAN'] or 'TranAD' in model.name :
            trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    ### Training phase
    if not args.test:
        print(f'Training {args.model} on {args.dataset}')
        num_epochs = epochs   # 训练5个epoch,继承模型的训练总epoch和时间（后改为10）
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)   # 训练网络  返回损失,学习率

            accuracy_list.append((lossT, lr)) # 添加训练结果

        print('Training time: ' + "{:10.4f}".format(time() - start) + ' s')
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')   # 绘制训练结果

    ### Testing phase
    torch.zero_grad = True  # 测试时无需梯度更新
    model.eval()  # 切换至测试模式,Dropout层激活所有单元,BN层停止计算和更新平均值和方差
    print(f'Testing {args.model} on {args.dataset}')
    if 'TPAD' in model.name:   #TPAD在测试时需要用到训练集，所以单独处理
        combineD=[trainD, testD]
        combine0 = [trainO, testO]
        loss, y_pred = backprop(0, model, combineD, combine0, optimizer, scheduler,
                                training=False)  # 返回损失,预测值
    else:
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)       # 返回损失,预测值

    ### Plot curves
    if not args.test:
        if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)   # 绘制结果

    ### Scores
    df = pd.DataFrame()
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)   # trainD为使用时间窗截取后的trainO,接收训练集重构Loss

    ls_list, pred_list = [], []
    for i in range(loss.shape[1]):  # 取i在0到特征维数之间
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l,ls)
        ls = ls.astype(np.int)
        pred = pred.astype(np.int)
        preds.append(pred)
        df = df.append(result, ignore_index=True)

        ls.tolist()
        pred.tolist()
        ls_list.append(ls)
        pred_list.append(pred)
        # # 绘制“ROC曲线”
        # fpr, tpr, roc_auc = CalculateROCAUCMetrics(ls[0], pred[0])
        # PlotROCAUC(f'{args.model}_{args.dataset}', i,loss.shape[1], fpr, tpr, roc_auc)
        # # 绘制“精度召回曲线”
        # precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(ls, pred)
        # PlotPrecisionRecallCurve(f'{args.model}_{args.dataset}', i, precision_curve, recall_curve, average_precision)

    # ls_array = np.array(ls_list).flatten()
    # pred_array = np.array(pred_list).flatten()
    # print(ls_array, pred_array)
    # newPlotROCAUC(f'{args.model}_{args.dataset}', ls_array, pred_array )

    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)   # 计算一列中的LossT和Loss平均值
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0   # 如果一列中有任何一个特征存在故障,即判定为故障
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)  # 使用POT阈值诊断
    result.update(hit_att(loss, labels))  # 把loss作为异常分数
    result.update(ndcg(loss, labels))

    # 绘制“ROC曲线”
    fpr, tpr, roc_auc = CalculateROCAUCMetrics(ls, pred)
    PlotROCAUC(f'{args.model}_{args.dataset}', fpr, tpr, roc_auc)
    # # 绘制“精度召回曲线”
    precision_curve, recall_curve, average_precision = CalculatePrecisionRecallCurve(ls, pred)
    PlotPrecisionRecallCurve(f'{args.model}_{args.dataset}', precision_curve, recall_curve, average_precision)
    print(df)
    df = np.array(df)
    df = np.around(df, decimals=3)
    df_row = df.shape[0]
    df_column = df.shape[1]
    # print(df.shape, df_row, df_column)
    for row in range(df_row):
        for column in range(df_column):
            item = QTableWidgetItem()  # 模型
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 文本显示位置
            # print(str(df[row, column]))
            item.setText(str(df[row, column]))
            item.setFont(QFont("微软雅黑", 12, QFont.Black))  # 设置字体
            ui_pa_tw_result.setItem(row, column, item)
    pprint(result)

    print('训练结束')
if __name__ == '__main__':
    model_training('TRAN','NAB',5,'--retrain')
