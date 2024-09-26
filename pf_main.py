import pickle
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class BATTERYSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        data = pd.read_csv(data_path + '/battery_train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        test_data = pd.read_csv(data_path + '/battery_test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/battery_test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

def set_layer(layer : nn.Module, freeze):
    if freeze:
        for param in layer.parameters():
            param.requires_grad = False
    else:
        for param in layer.parameters():
            param.requires_grad = True

def calculate_ratio2(vali_score):
    # 生成随机序列作为示例数据
    data = vali_score
    # 将数据转化为2维数组
    X = data.reshape(-1, 1)
    # 使用K-means聚类将数据分成2类
    kmeans = KMeans(n_clusters=2).fit(X)
    # 获取每个数据点所属的簇的标签
    labels = kmeans.labels_
    # 统计每个簇中数据点的数量
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # 计算每个簇的比例
    proportions = label_counts / len(data)
    # 输出结果
    for i, proportion in enumerate(proportions):
        print(f"Cluster {i}: {proportion:.2%}")
        cluster_data = data[labels == i]
        if len(cluster_data) > 0:
            min_val = np.min(cluster_data)
            max_val = np.max(cluster_data)
            print(f"  Range: [{min_val:.2f}, {max_val:.2f}]")


def calculate_sort(vali_score):
    # 生成随机序列作为示例数据
    data = vali_score
    # 将数据分成3个区间
    bins = [0, 0.01, 0.1, 1]
    # 统计每个区间内数据点的数量
    counts, _ = np.histogram(data, bins=bins)
    # 计算每个区间的比例
    proportions = counts / len(data)

    # 输出结果
    for i, bin in enumerate(zip(bins[:-1], bins[1:])):
        print(f"{bin[0]:.2f}-{bin[1]:.2f}: {counts[i]} ({proportions[i]:.2%})")

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]   #如果索引大于窗口长度,则从中截取
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])     # 如果索引小于窗口长度,则对首位进行复制操作,使得w达到窗口长度要求
		windows.append(w if 'TranAD' in args.model or 'TPAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')

	if args.dataset == 'BATTERY':
		train_dataset = BATTERYSegLoader('data/BATTERY', win_size=10, step=1, mode='train')
		test_dataset = BATTERYSegLoader('data/BATTERY', win_size=10, step=1, mode='test')

		train_loader = DataLoader(dataset=train_dataset,
								 batch_size=128,
								 shuffle=True,
								 num_workers=0)

		test_loader = DataLoader(dataset=test_dataset,
								  batch_size=128,
								  shuffle=False,
								  num_workers=0)

		labels = np.load(os.path.join(folder, 'labels.npy'))

	else:

		loader = []
		for file in ['train', 'test', 'labels']:
			if dataset == 'SMD': file = 'machine-1-1_' + file
			if dataset == 'SMAP': file = 'P-1_' + file
			if dataset == 'MSL': file = 'C-1_' + file
			if dataset == 'UCR': file = '136_' + file
			if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
			loader.append(np.load(os.path.join(folder, f'{file}.npy')))   #用npy读取文件
		# loader = [i[:, debug:debug+1] for i in loader]
		if args.less: loader[0] = cut_array(0.2, loader[0])            # 如果less为真,就只选取20%的训练数据
		train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])    # 将一整个时间序列长度作为batch_size输入
		test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])     # 将一整个时间序列长度作为batch_size输入
		labels = loader[2]


	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)    #使用model_class获取模型的模型名
	model = model_class(dims).double() # 将所有类型的浮点和缓冲转化为双浮点类型
	if modelname == 'TPAD':
		optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
	else:
		optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)# AdamW优化器,即Adam + weight decate,效果与Adam + L2正则化相同
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)   #根据epoch训练次数调整学习率
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'

	if os.path.exists(fname) and (not args.retrain or args.test):    # 如果已存在训练好的模型且命令行没有设置重新训练的要求
		print(f"Loading pre-trained model: {model.name}")
		checkpoint = torch.load(fname)    # 加载训练好的模型
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"Creating new model: {model.name}")    # 否则创建新模型
		epoch = -1; accuracy_list = []   # 在训练时epoch+1变为0

		return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):  # data是使用时间窗截取后的dataO
	if 'TPAD' not in model.name:
		l = nn.MSELoss(reduction = 'mean' if training else 'none')  # 训练的时候求平均,否则求和(training默认为true)
		feats = dataO.shape[1]  # 获取数据集的特征数

	if 'DAGMM' in model.name:   # 训练DAGMM模型
		l = nn.MSELoss(reduction = 'none')  # 求和
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:  # 训练GDN,MTAD_GAT,MSCRED,CAE_M模型
		l = nn.MSELoss(reduction = 'none')  # 求和
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name:   # 训练MTAD_GAT模型
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name:   # 训练MTAD_GAT模型
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'GAN' in model.name:  # 训练GAN模型
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:  # 训练TranAD模型
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x) # 将样本和标签包装为dataset,该处计算重构损失,故输入输出为同一值
		bs = model.batch if training else len(data)  # 如果继续训练就沿用旧的batch_size,否则取data的长度作为batch_size
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window   # 获取模型参数并保留
		l1s, l2s = [], []
		if training:     # 如果为训练命令
			for d, _ in dataloader:
				print(d.shape)
				local_bs = d.shape[0]  # 获得当前的batch_size
				window = d.permute(1, 0, 2)  # 对张量d进行维度重排并赋值给window
				elem = window[-1, :, :].view(1, local_bs, feats) # 取window最后一个索引的数据并改变其形状（目标数据）
				z = model(window, elem) # window为src,elem为tgt
				# 如果z不为元组,则取前一公式,否则,使用后面公式（二段式Loss公式）
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1] # 如果z是元组类型，则取1号索引的数据
				l1s.append(torch.mean(l1).item()) # 取MSE平均值加入列表
				loss = torch.mean(l1)
				optimizer.zero_grad() # 梯度清零
				loss.backward(retain_graph=True) # 计算图不会被立即释放
				optimizer.step() # 参数更新
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else: # 如果不为训练命令
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]  # 如果z是元组类型，则取1号索引的数据
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]  # 返回与loss和z[0]相同的新的tensor,但其不具备梯度

	elif 'TPAD' in model.name:  # 训练TranAD模型
		l = nn.MSELoss()
		loss1_list = []
		loss2_list = []
		loss3_list = []
		n = epoch + 1; w_size = model.n_window   # 获取模型参数并保留
		if training:
			for i, (input_data, labels) in enumerate(tqdm(data)):
				device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
				input = input_data.double().to(device)
				output1, output2, output3, series, prior, _ = model(input)  #input.shape=128,10,28

				# Association discrepancy Minimax strategy
				# calculate Association discrepancy
				series_loss = 0.0
				prior_loss = 0.0
				for u in range(len(prior)):
					series_loss += (torch.mean(my_kl_loss(series[u], (
							prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																								   w_size)).detach())) + torch.mean(
						my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach(),
								   series[u])))
					prior_loss += (torch.mean(my_kl_loss(
						(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																								w_size)),
						series[u].detach())) + torch.mean(
						my_kl_loss(series[u].detach(), (
								prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																									   w_size)))))
				series_loss = series_loss / len(prior)
				prior_loss = prior_loss / len(prior)

				#total_assdis.append(self.k * series_loss.item())

				loss1_list.append((- 3 * series_loss).item())
				loss_p = 3 * prior_loss
				loss_s = -3 * series_loss

				set_layer(model.decoder, freeze=True)
				set_layer(model.decoder1, freeze=True)
				set_layer(model.decoder2, freeze=True)

				optimizer.zero_grad()
				loss_s.backward(retain_graph=True)
				loss_p.backward()
				optimizer.step()

				set_layer(model.decoder, freeze=False)
				set_layer(model.decoder1, freeze=False)
				set_layer(model.decoder2, freeze=False)

				# Train encoder+decoder1
				# loss_g = e^-n||O1-W||2 + (1-e^-n)||O2'-W||2
				output1, output2, output3, series, prior, _ = model(input)
				loss_g = (math.pow(1+8e-3, -i)) * l(output1, input) + (
						1 - (math.pow(1+8e-3, -i))) * l(output3, input)

				# loss_g = (1-(i+1)/train_steps) * self.criterion(output1, input) + (
				#         (i+1)/train_steps) * self.criterion(output3, input)

				loss2_list.append(loss_g.item())

				set_layer(model.decoder2, freeze=True)

				optimizer.zero_grad()
				loss_g.backward()
				optimizer.step()

				set_layer(model.decoder2, freeze=False)

				# Train encoder+decoder2
				output1, output2, output3, series, prior, _ = model(input)
				# loss_d = e^-n||O2-W||2 - (1-e^-n)||O2'-W||2

				loss_d = (math.pow(1+8e-3, -i)) * l(output2, input) - (
						1 - (math.pow(1+8e-3, -i))) * l(output3, input)

				# loss_d = (1-(i+1)/train_steps) * self.criterion(output2, input) - (
				#         (i+1)/train_steps) * self.criterion(output3, input)


				loss3_list.append(loss_d.item())

				set_layer(model.decoder1, freeze=True)

				optimizer.zero_grad()
				loss_d.backward()
				optimizer.step()

				set_layer(model.decoder1, freeze=False)

			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(loss1_list)},\tL2 = {np.mean(loss2_list)},\tL3 = {np.mean(loss3_list)}')
			return np.mean(loss1_list) + np.mean(loss2_list)+ np.mean(loss3_list), optimizer.param_groups[0]['lr']

		else:
			if len(data) == 2:		#data=[train_dataloader,test_dataloader]
				temperature = 50
				criterion = nn.MSELoss(reduce=False)
				# (1) stastic on the train set
				print("======================TEST MODE: First Phase======================")
				attens_energy = []
				for i, (input_data, labels) in enumerate(tqdm(data[0])):
					device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
					input = input_data.double().to(device)
					output1, output2, output3, series, prior, sigmas = model(input)
					# plot_series(input)
					loss_rec = torch.mean(0.4 * criterion(input, output1), dim=-1)
					loss_gan = torch.mean(0.6 * criterion(input, output3), dim=-1)

					series_loss = 0.0
					prior_loss = 0.0
					lenp = len(prior)
					for u in range(lenp):
						if u == 0:
							series_loss = my_kl_loss(series[u], (
									prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach()) * temperature
							prior_loss = my_kl_loss(
								(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										w_size)),
								series[u].detach()) * temperature
						else:
							series_loss += my_kl_loss(series[u], (
									prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach()) * temperature
							prior_loss += my_kl_loss(
								(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										w_size)),
								series[u].detach()) * temperature

					metric = torch.softmax((-series_loss - prior_loss), dim=-1)
					cri = metric * 2 * (loss_rec + loss_gan)
					cri = cri.detach().cpu().numpy()
					attens_energy.append(cri)

				attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
				train_energy = np.array(attens_energy)

				# (2) find the threshold
				print("======================TEST MODE: Second Phase======================")
				attens_energy = []
				for i, (input_data, labels) in enumerate(tqdm(data[1])):
					device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
					input = input_data.double().to(device)
					output1, output2, output3, series, prior, _ = model(input)


					loss_rec = torch.mean(0.4 * criterion(input, output1), dim=-1)
					loss_gan = torch.mean(0.6 * criterion(input, output3), dim=-1)

					series_loss = 0.0
					prior_loss = 0.0
					for u in range(len(prior)):
						if u == 0:
							series_loss = my_kl_loss(series[u], (
									prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach()) * temperature
							prior_loss = my_kl_loss(
								(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										w_size)),
								series[u].detach()) * temperature
						else:
							series_loss += my_kl_loss(series[u], (
									prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach()) * temperature
							prior_loss += my_kl_loss(
								(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										w_size)),
								series[u].detach()) * temperature
					# Metric
					metric = torch.softmax((-series_loss - prior_loss), dim=-1)
					cri = metric * 2 * (loss_rec + loss_gan)
					cri = cri.detach().cpu().numpy()
					attens_energy.append(cri)

				attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
				test_energy = np.array(attens_energy)
				# calculate_ratio2(test_energy)
				# calculate_sort(test_energy)
				combined_energy = np.concatenate([train_energy, test_energy], axis=0)
				thresh = np.percentile(combined_energy, 100 - 1)
				print("Threshold :", thresh)

				# (3) evaluation on the test set
				print("======================TEST MODE: Third Phase======================")
				test_labels = []
				attens_energy = []

				for i, (input_data, labels) in enumerate(tqdm(data[1])):
					device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
					input = input_data.double().to(device)
					output1, output2, output3, series, prior, _ = model(input)

					loss_rec = torch.mean(0.4 * criterion(input, output1), dim=-1)
					loss_gan = torch.mean(0.6 * criterion(input, output3), dim=-1)

					series_loss = 0.0
					prior_loss = 0.0
					for u in range(len(prior)):
						if u == 0:
							series_loss = my_kl_loss(series[u], (
									prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach()) * temperature
							prior_loss = my_kl_loss(
								(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										w_size)),
								series[u].detach()) * temperature
						else:
							series_loss += my_kl_loss(series[u], (
									prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										   w_size)).detach()) * temperature
							prior_loss += my_kl_loss(
								(prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
																										w_size)),
								series[u].detach()) * temperature
					metric = torch.softmax((-series_loss - prior_loss), dim=-1)

					cri = metric * 2 * (loss_rec + loss_gan)
					cri = cri.detach().cpu().numpy()

					attens_energy.append(cri)
					test_labels.append(labels)

				attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
				test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
				test_energy = np.array(attens_energy)
				test_labels = np.array(test_labels)
				pred = (test_energy > thresh).astype(int)

				gt = test_labels.astype(int)

				# detection adjustment
				anomaly_state = False
				for i in range(len(gt)):
					if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
						anomaly_state = True
						for j in range(i, 0, -1):
							if gt[j] == 0:
								break
							else:
								if pred[j] == 0:
									pred[j] = 1
						for j in range(i, len(gt)):
							if gt[j] == 0:
								break
							else:
								if pred[j] == 0:
									pred[j] = 1
					elif gt[i] == 0:
						anomaly_state = False
					if anomaly_state:
						pred[i] = 1

				pred = np.array(pred)
				gt = np.array(gt)

				return test_energy, pred   #返回损失,预测值

	else: # 如果不为训练命令
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

