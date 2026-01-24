from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms import CompositeTransform
import torch
import torch.nn as nn
from utils import AvgL1Norm
import torch.nn.functional as F

class ProbabilisticActor(nn.Module):
	def __init__(self, backbone_state_dim, action_dim, encoder_dim=256, hdim=256, activ=nn.ReLU):
		super(ProbabilisticActor, self).__init__()

		self.l0 = nn.Linear(backbone_state_dim, hdim)
		self.context_dim = encoder_dim + hdim
		self.final_activation = nn.Tanh()

		self.action_dist = ConditionalDiagonalNormal(
			shape=[action_dim],
			context_encoder = nn.Sequential(
				nn.Linear(self.context_dim, hdim),
				activ(inplace=True),
				nn.Linear(hdim, hdim),
				activ(inplace=True),
				nn.Linear(hdim, 2*action_dim),
			)
		)
		self.transform = CompositeTransform([])
		self.flow = Flow(self.transform, self.action_dist)

	def forward(self, backbone_state, zs):
		s_emb = AvgL1Norm(self.l0(backbone_state))
		context = torch.cat([s_emb, zs], dim=1)
		action = self.flow.sample(1, context=context)
		action = action.squeeze(1)
		action = self.final_activation(action)
		return action

class DeterministicActor(nn.Module):
	def __init__(self, backbone_state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.relu):
		super(DeterministicActor, self).__init__()

		self.activ = activ
		self.l0 = nn.Linear(backbone_state_dim, hdim)
		self.context_dim = hdim + encoder_dim
		self.l1 = nn.Linear(self.context_dim, hdim)
		self.l2 = nn.Linear(hdim, hdim)
		self.l3 = nn.Linear(hdim, action_dim)


	def forward(self, backbone_state, zs):
		s_emb = AvgL1Norm(self.l0(backbone_state))
		context = torch.cat([s_emb, zs], dim=1)
		a = self.activ(self.l1(context))
		a = self.activ(self.l2(a))
		a = self.l3(a)
		return torch.tanh(a)
	
class Critic(nn.Module):
	def __init__(self, backbone_state_dim, action_dim, encoder_dim=256, hdim=256, activ=F.elu):
		super(Critic, self).__init__()

		self.activ = activ

		self.q01 = nn.Linear(backbone_state_dim + action_dim, hdim)
		self.q1 = nn.Linear(2*encoder_dim + hdim, hdim)
		self.q2 = nn.Linear(hdim, hdim)
		self.q3 = nn.Linear(hdim, 1)

		self.q02 = nn.Linear(backbone_state_dim + action_dim, hdim)
		self.q4 = nn.Linear(2*encoder_dim + hdim, hdim)
		self.q5 = nn.Linear(hdim, hdim)
		self.q6 = nn.Linear(hdim, 1)


	def forward(self, backbone_state, action, zsa, zs):
		sa = torch.cat([backbone_state, action], 1)
		embeddings = torch.cat([zsa, zs], 1)

		q1 = AvgL1Norm(self.q01(sa))
		q1 = torch.cat([q1, embeddings], 1)
		q1 = self.activ(self.q1(q1))
		q1 = self.activ(self.q2(q1))
		q1 = self.q3(q1)

		q2 = AvgL1Norm(self.q02(sa))
		q2 = torch.cat([q2, embeddings], 1)
		q2 = self.activ(self.q4(q2))
		q2 = self.activ(self.q5(q2))
		q2 = self.q6(q2)
		return torch.cat([q1, q2], 1)
	
class RewardPredictor(nn.Module):
	def __init__(self, zs_dim, action_dim, hdim=256, activ=nn.ELU):
		super(RewardPredictor, self).__init__()

		self.activ = activ(inplace=True)

		self.r1 = nn.Linear(zs_dim + action_dim + zs_dim, hdim)
		self.r2 = nn.Linear(hdim, hdim)
		self.r3 = nn.Linear(hdim, 1)
	def forward(self, zs, action, zsa):
		x = torch.cat([zs, action, zsa], 1)
		x = self.activ(self.r1(x))
		x = self.activ(self.r2(x))
		reward = self.r3(x)
		return reward
