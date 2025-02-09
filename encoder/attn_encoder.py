import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import queue
import shap
from sklearn.tree import DecisionTreeRegressor
from utils import flatten_tree, generate_plan_tree, _pad_and_combine, exec_analyze_query, generate_latency


def left_child(x):
    if len(x) != 3:
        return None
    return x[1]


def right_child(x):
    if len(x) != 3:
        return None
    return x[2]


def features(x):
    return x[0]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=3):
        # inputs
        # Q: [batch_size x n_heads x len_q x d_k]
        # K: [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_k x d_v]
        # scores : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # attn_mask
        if(attn_mask != None):
            # scores += attn_mask
            attn_mask = attn_mask.float()
            scores = torch.matmul(scores, attn_mask.transpose(-1, -2))
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=9, n_heads=3, d_q=3, d_k=3, d_v=3):
        # d_model = nheads * d_q
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        # param matrix
        self.W_Q = nn.Linear(d_model, d_q * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        # inputs: Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_v x d_model]
        residual, batch_size = Q, len(Q)

        # proj && split
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_v x d_v]

        # attn_mask: batch_size x len_q x len_k -> [batch_size x n_heads x len_q x len_k]
        if(attn_mask != None):
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_dim1=9, in_dim2=9, k_dim=3, v_dim=3, num_heads=3):
        # in_dim1 = k_dim * num_heads
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)
    
    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)
        
        # q1(batch_size, num_heads, seq_len1, k_dim)
        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).transpose(1,2)
        # k2(batch_size, num_heads, k_dim, seq_len2)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        # v2(batch_size, num_heads, seq_len2, v_dim)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).transpose(1,2)
        
        # attention(batch_size, num_heads, seq_len1, seq_len2)
        attention = torch.matmul(q1, k2) / self.k_dim ** 0.5
        
        if mask is not None:
            mask = mask.reshape(batch_size, self.num_heads, seq_len1, seq_len2)
            attention = attention + mask
        
        attention = F.softmax(attention, dim=1)
        # output(batch_size, num_heads, seq_len1, v_dim)=>(batch_size, seq_len1, num_heads*v_dim)
        output = torch.matmul(attention, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        # output(batch_size, seq_len1, in_dim1)
        output = self.proj_o(output)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=9, d_ff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class QueryPlanEncoderLayer(nn.Module):
    def __init__(self):
        super(QueryPlanEncoderLayer, self).__init__()
        self.multihead = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, query, latency):
        # process Q,K,V
        q_list = []
        l_list = []
        for i in range(0, len(query)):
            q_l = flatten_tree(query[i], [])
            q_list.append(np.array(q_l))
            l_l = flatten_tree(latency[i], [])
            l_l = np.expand_dims(l_l, axis=1)
            l_list.append(np.array(l_l))
        q_list = _pad_and_combine(q_list)
        l_list = _pad_and_combine(l_list)
        L = torch.Tensor(l_list)  # L: [batch_size x len_q x 1]
        Q = torch.Tensor(q_list)
        # Q: [batch_size x len_q x d_model] example: 1x24x9
        # enc_inputs to same Q,K,V
        len_q = Q.size(1)
        pos_encode = PositionalEncoding()
        attn_mask_list = []
        for q in query:
            attn_mask = pos_encode([q], len_q)
            attn_mask_list.append(attn_mask)
        attn_mask = torch.tensor(attn_mask_list) # batch_size x len_q x len_q
        enc_outputs, attn = self.multihead(Q, Q, Q, attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn, L


class PairEncoderLayer(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, n_heads):
        super(PairEncoderLayer, self).__init__()
        self.multihead = MultiHeadCrossAttention(in_dim1, in_dim2, k_dim, v_dim, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(in_dim1)

    def forward(self, plan, knobs, W):
        # process Q,K,V
        # 需要plan和knobs的第三维度对齐len_qk
        K = knobs.repeat_interleave(plan.size(2), dim=2)
        plan = plan.repeat_interleave(knobs.size(2), dim=2)
        enc_outputs = self.multihead(plan, K, W) # batch_size x len_k x len_qk
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class KnobsEncoderLayer(nn.Module):
    def __init__(self, knobs_dim):
        super(KnobsEncoderLayer, self).__init__()
        self.knobs_dim = knobs_dim
        self.multihead = MultiHeadAttention(d_model=knobs_dim, n_heads=knobs_dim, d_q=1, d_k=1, d_v=1)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=knobs_dim)

    def forward(self, query):
        # process Q,K,V
        # Q: [batch_size x len_q x d_model] example: 1x1x3
        query_list = []
        for q in query:
            query_list.append([q])
        Q = torch.tensor(query_list)
        attn_mask = torch.ones(Q.size(0), Q.size(1), Q.size(1)) # batch_size x len_q x len_q
        enc_outputs, attn = self.multihead(Q, Q, Q, attn_mask=None)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def list_to_tuple(self, lst):
        if isinstance(lst, list):
            return tuple(self.list_to_tuple(item) for item in lst)
        else:
            return lst
    
    def isEmptyNode(self, root):
        return root == [0. for _ in range(9)] or root == [[0. for _ in range(9)]]
    
    def buildGraph(self, root, u, graph, map_):
        if self.isEmptyNode(root):
            return
        if (self.isEmptyNode(root[1]) == False):
            tmp_node = self.list_to_tuple(root[1])
            v = map_[tmp_node]
            graph[u][v] = graph[v][u] = 1
            self.buildGraph(root[1], v, graph, map_)
        if (self.isEmptyNode(root[2]) == False):
            tmp_node = self.list_to_tuple(root[2])
            v = map_[tmp_node]
            graph[u][v] = graph[v][u] = 1
            self.buildGraph(root[2], v, graph, map_)
    
    def getNodeCount(self, tree):
        q = queue.Queue()
        q.put(tree[0])
        n = 0
        while(q.empty() == False):
            size = q.qsize()
            while(size):
                front_node = q.get()
                n = n + 1
                if len(front_node) > 1 and front_node[1] != [0. for _ in range(9)] and front_node[1] != [[0. for _ in range(9)]]:
                    q.put(front_node[1])
                if len(front_node) > 1 and front_node[2] != [0. for _ in range(9)] and front_node[2] != [[0. for _ in range(9)]]:
                    q.put(front_node[2])
                size = size - 1
        return n
    
    def convertTreeToMatrix(self, tree, num):
        # 层序遍历统计树中节点数以及每个节点的编号
        q = queue.Queue()
        q.put(tree[0])
        map_ = {}
        n = 0
        idx = 0
        while(q.empty() == False):
            size = q.qsize()
            while(size):
                front_node = q.get()
                n = n + 1
                tmp_tuple = self.list_to_tuple(front_node)
                map_[tmp_tuple] = idx
                idx = idx + 1
                if len(front_node) > 1 and front_node[1] != [0. for _ in range(9)] and front_node[1] != [[0. for _ in range(9)]]:
                    q.put(front_node[1])
                if len(front_node) > 1 and front_node[2] != [0. for _ in range(9)] and front_node[2] != [[0. for _ in range(9)]]:
                    q.put(front_node[2])
                size = size - 1
        # matrix = [[0] * n for _ in range(n)]
        matrix = [[0] * num for _ in range(num)]
        for i in range(0, n):
            matrix[i][i] = 1
        tmp_root = self.list_to_tuple(tree[0])
        self.buildGraph(tree[0], map_[tmp_root], matrix, map_)
        return matrix

    def forward(self, root, num):
        if len(root) == 0:
            return []
        matrix = self.convertTreeToMatrix(root, num)
        return matrix


class AttrEncoder(nn.Module):
    def __init__(self, plan_in_channel, knobs_dim):
        super(AttrEncoder, self).__init__()
        self.channel = plan_in_channel
        self.knobs_dim = knobs_dim
        self.query_plan_layer = QueryPlanEncoderLayer()
        self.knob_layer = KnobsEncoderLayer(knobs_dim)
        self.pair_layer = PairEncoderLayer(plan_in_channel * knobs_dim, plan_in_channel * knobs_dim, plan_in_channel, plan_in_channel, knobs_dim)
        self.single_split_model = DecisionTreeRegressor(max_depth=1)
        self.l = nn.Linear(plan_in_channel * knobs_dim, 1)

    def train_latency(self, pair_vec, latency):
        assert pair_vec.shape[0] == latency.shape[0]
        # 逐个学习查询
        for i in range(0, len(latency)):
            assert pair_vec[i].shape[0] == latency[i].shape[0]
            self.single_split_model.fit(pair_vec[i].detach().numpy(), latency[i].detach().numpy())
        # 计算shape
        W = []
        for i in range(0, len(latency)):
            shape_values = shap.TreeExplainer(self.single_split_model).shap_values(pair_vec[i].detach().numpy())
            W.append(shape_values)
        W = torch.tensor(W)
        # 只需要取后面N个参数特征
        W = W[:, :, -self.knobs_dim:].float()
        return W

    def forward(self, plans, knobs):
        assert len(plans) == len(knobs)
        plan = []
        latency = []
        for p in plans:
            latency.append(generate_latency(p))
            plan.append(generate_plan_tree(p))
        # query plan self-attention
        query_plan_out, query_plan_attn, L = self.query_plan_layer(plan, latency) # 1x24x9
        # knob self-attention
        knob_out, knob_attn = self.knob_layer(knobs) # 1x1x3
        # cross-attention
        # 模型训练每一类结点：vec(plan,knob)和latency之间的关系
        pair_vec = torch.cat((query_plan_out, knob_out.repeat(1, query_plan_out.size(1), 1)), dim=2) # 1x24x12
        # 计算knob-node矩阵
        W = self.train_latency(pair_vec, L)
        pair_out = self.pair_layer(query_plan_out, knob_out, W)
        pair_out = self.l(pair_out)
        # pair_out = pair_out.mean(dim=1)
        return pair_out


if __name__ == "__main__":
    encoder = AttrEncoder(9, 3)
    sql = "SELECT COUNT(*) FROM cast_info ci WHERE ci.person_id>583692 AND ci.role_id=2"
    plan_analyze, latency = exec_analyze_query(sql, 'imdb', {'enable_optimizer': 1})
    plan_analyze_list = [plan_analyze, plan_analyze]
    vec = encoder([plan_analyze, plan_analyze], [[0., 0.3, 0.4], [0.2, 0.1, 0.]])
    print(vec)