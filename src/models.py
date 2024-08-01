from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv, SAGEConv, APPNPConv, GINConv, ChebConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
torch.manual_seed(42)

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5 # MHA w_size = 5
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)		
		return g, ats

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden


## GDN Model (AAAI 21)
class GDN(nn.Module):
	def __init__(self, feats):
		super(GDN, self).__init__()
		self.name = 'GDN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats
		src_ids = np.repeat(np.array(list(range(feats))), feats)
		dst_ids = np.array(list(range(feats))*feats)
		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
		self.g = dgl.add_self_loop(self.g)
		self.feature_gat = GATConv(1, 1, feats)
		self.attention = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
		)
		self.fcn = nn.Sequential(
			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
		)

	def forward(self, data):
		# Bahdanau style attention
		att_score = self.attention(data).view(self.n_window, 1)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.matmul(data.permute(1, 0), att_score)
		# GAT convolution on complete graph
		feat_r = self.feature_gat(self.g, data_r)
		feat_r = feat_r.view(self.n_feats, self.n_feats)
		# Pass through a FCN
		x = self.fcn(feat_r)
		return x.view(-1)

# Learning stable graph (our)
class LSG(nn.Module):
	def __init__(self, feats):
		super(LSG, self).__init__()
		self.name = 'LSG'
		self.batch = 256  # original 64  SWaT_a  64
		self.lr = lsg_lr
		self.device = args.device
		self.n_feats = feats
		self.embed_dim = 32
		self.n_window = n_win
		self.topk = top_k
		self.noise_embed_layer = nn.Sequential(
			nn.Linear(self.n_window,self.n_window),
            nn.Sigmoid() #SMAP_va 删除
		)
		if args.GNN == 'GIN':
			self.gnn_layer = dgl.nn.GINConv(nn.Linear(self.n_window,self.embed_dim),
											aggregator_type='sum',init_eps=0.1,learn_eps=False)
		elif args.GNN == 'GAT':
			self.gnn_layer = GATConv(self.n_window,self.embed_dim,num_heads=1,activation=None)
		elif args.GNN == 'GCN':
			#self.pseuduo =  
			self.gnn_layer = dgl.nn.GraphConv(self.n_window,self.embed_dim, norm='both', 
			                                  weight=True,bias=True,activation=None)
		elif args.GNN == 'GraphSAGE':
			self.gnn_layer = SAGEConv(self.n_window,self.embed_dim,
			                          aggregator_type='mean',activation=None)
		elif args.GNN == 'ChebyNet':
			self.gnn_layer = ChebConv(self.n_window,self.embed_dim,
			                          k=8,activation=None)
		else:
			self.gnn_layer = dgl.nn.GINConv(nn.Linear(self.n_window,self.embed_dim),
											aggregator_type='sum',init_eps=0.1,learn_eps=False)

		
		self.lstm = nn.LSTM(self.n_window,self.embed_dim)

		self.bn_outlayer_in = nn.BatchNorm1d(self.embed_dim*2)
		self.dp = nn.Dropout(0.2)
		self.out_layer=nn.Sequential(
			nn.Linear(self.embed_dim*2,256),
			nn.BatchNorm1d(self.n_feats),
			nn.PReLU(),  #SMAP_va 删除
			nn.Linear(256,1),
			nn.ReLU(), 			
		)	

	def forward(self, data, hn, cn):
		#init data = [batch, window, feats]
		data = data.permute(0, 2, 1)
		# denoise 
		noise_embed = self.noise_embed_layer(data).to(self.device)
		x = data - noise_embed 
		bs = data.shape[0]
		h0 = torch.zeros(1,self.n_feats,self.embed_dim,dtype=torch.double).to(self.device)
		c0 = torch.zeros(1,self.n_feats,self.embed_dim,dtype=torch.double).to(self.device)
		gcn_out = []
		graph_out = []
		hidden_t = []
		for i in range(bs):
			x_ = x[i,:,:]
			x_o = x_.view(1,-1,self.n_window).contiguous()
			if torch.mean(hn)==0:
				ht,(hn,cn)=self.lstm(x_o,(h0,c0))
			else:
				ht,(hn,cn)=self.lstm(x_o,(hn,cn))
		    
			weights_arr = torch.cat([hn,x_o],dim=-1).squeeze(dim=0) #.detach().clone()
			#weights_arr = hn.squeeze(dim=0)  #  MSL_va  SMAP_va  SMD,
			weights = weights_arr.view(self.n_feats,-1).to(self.device)
			cos_ij_mat = torch.matmul(weights,weights.T)
			normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
			cos_ij_mat = cos_ij_mat/normed_mat

			topk_indices_ij = torch.topk(cos_ij_mat,self.topk,dim=-1)[1]
			graph_out.append(cos_ij_mat)

			gated_i = torch.arange(0,self.n_feats).T.unsqueeze(1).repeat(1,self.topk).flatten().to(self.device) #.unsqueeze(0)
			gated_j = topk_indices_ij.flatten().to(self.device).to(self.device) #.unsqueeze(0)
			#gated_edge_index = torch.cat((gated_i,gated_j),dim=0)

			graph = dgl.graph((gated_i,gated_j)).to(self.device)
			graph = dgl.add_self_loop(graph)

			gcn_o = self.gnn_layer(graph,x_.squeeze(dim=0)).to(self.device)
			gcn_o = gcn_o.view(self.n_feats,-1)
			gcn_out.append(gcn_o)
			hidden_t.append(ht.squeeze(dim=0))
		
		out = torch.cat(gcn_out,dim=0)
		out = out.view(bs,self.n_feats,self.embed_dim)
		hidden_out = torch.cat(hidden_t,dim=0)
		hidden_out = hidden_out.view(bs,self.n_feats,self.embed_dim)

		out = torch.cat([out, hidden_out],dim=-1)    #  MSL_va  SMAP_va  SMD,
		#out  = torch.cat([out,out],dim=-1)

		out = out.permute(0,2,1)
		out = F.relu(self.bn_outlayer_in(out))
		out = out.permute(0,2,1)
		
		out = self.dp(out)
		out = self.out_layer(out)
		out = out.squeeze(dim=-1)
		return out, graph_out, hn.detach().clone(), cn.detach().clone()


