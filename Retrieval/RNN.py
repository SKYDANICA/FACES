import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, GlobalAttention
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import ast
import networkx as nx
from transformers import BertModel, BertTokenizer
import numpy as np
from scipy.spatial.distance import cosine

# 首先定义模型类，与训练时一致
class EdgeAwareGNN(MessagePassing):
    def __init__(self, node_dim=768, edge_dim=768, hidden_dim=256, dropout=0.1):
        super(EdgeAwareGNN, self).__init__(aggr='add')
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        self.message_fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_fc(torch.cat([x, out], dim=1))
        return out

class CodeSimilarityModel(nn.Module):
    def __init__(self, node_dim=768, edge_dim=768, hidden_dim=256, graph_dim=128, dropout=0.1):
        super(CodeSimilarityModel, self).__init__()
        
        self.gnn1 = EdgeAwareGNN(node_dim, edge_dim, hidden_dim, dropout)
        self.gnn2 = EdgeAwareGNN(hidden_dim, edge_dim, hidden_dim, dropout)
        
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.node_pooling = GlobalAttention(gate_nn=self.node_gate)
        
        self.edge_pooling = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.graph_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.Dropout(dropout)
        )
        
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        h1 = self.gnn1(x, edge_index, edge_attr)
        h2 = self.gnn2(h1, edge_index, edge_attr)
        
        node_embedding = self.node_pooling(h2, batch)
        
        edge_batch = batch[edge_index[0]]
        
        if len(edge_batch) > edge_attr.size(0):
            edge_batch = edge_batch[:edge_attr.size(0)]
        elif len(edge_batch) < edge_attr.size(0):
            edge_batch = torch.cat([edge_batch, edge_batch[-1:].repeat(edge_attr.size(0) - len(edge_batch))])
        
        edge_embedding = global_add_pool(edge_attr, edge_batch)
        edge_embedding = self.edge_pooling(edge_embedding)
        
        graph_embedding = torch.cat([node_embedding, edge_embedding], dim=1)
        graph_embedding = self.graph_fusion(graph_embedding)
        
        return graph_embedding
    
    def compute_similarity(self, code1, code2):
        embedding1 = self.forward(code1)
        embedding2 = self.forward(code2)
        
        similarity = F.cosine_similarity(embedding1, embedding2)
        return similarity

# 代码图构建器，与训练时一致
class CodeGraphBuilder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval()
        
    @torch.no_grad()
    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)
    
    def parse_code_to_graph(self, code):
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"语法错误: {e}")
            return None
        
        graph = nx.DiGraph()
        
        def get_node_label(node):
            if isinstance(node, ast.FunctionDef):
                return f"Function: {node.name}"
            elif isinstance(node, ast.Name):
                return f"Variable: {node.id}"
            elif isinstance(node, ast.Call):
                return f"Function Call: {getattr(node.func, 'id', 'unknown')}"
            elif isinstance(node, ast.Assign):
                return f"Assignment: {len(node.targets)} targets"
            elif isinstance(node, ast.If):
                return "If Statement"
            elif isinstance(node, ast.For):
                return "For Loop"
            elif isinstance(node, ast.While):
                return "While Loop"
            return type(node).__name__
        
        def get_edge_type(node):
            if isinstance(node, ast.If):
                return 'control_flow_if'
            elif isinstance(node, ast.For) or isinstance(node, ast.While):
                return 'control_flow_loop'
            elif isinstance(node, ast.Assign):
                return 'data_flow_assignment'
            elif isinstance(node, ast.Call):
                return 'function_call'
            elif isinstance(node, ast.Return):
                return 'return_statement'
            return 'syntax_connection'
        
        def traverse_ast(node, parent=None):
            node_id = id(node)
            node_label = get_node_label(node)
            node_embedding = self.get_bert_embedding(node_label).cpu().numpy()
            
            graph.add_node(node_id, label=node_label, embedding=node_embedding)
            
            if parent is not None:
                edge_type = get_edge_type(node)
                edge_label = f"{edge_type}: {node_label}"
                edge_embedding = self.get_bert_embedding(edge_label).cpu().numpy()
                graph.add_edge(parent, node_id, label=edge_label, embedding=edge_embedding, type=edge_type)
            
            for child in ast.iter_child_nodes(node):
                traverse_ast(child, parent=node_id)
        
        traverse_ast(tree)
        return graph
    
    def convert_to_pyg_data(self, graph, label=None):
        if graph is None or len(graph.nodes) == 0:
            return None
            
        node_features = []
        node_ids = list(graph.nodes)
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for node_id in node_ids:
            node_features.append(graph.nodes[node_id]['embedding'])
        
        edge_index = []
        edge_features = []
        
        for u, v, data in graph.edges(data=True):
            edge_index.append([id_to_idx[u], id_to_idx[v]])
            edge_features.append(data['embedding'])
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        
        return data

def load_model(model_path, device):
    """加载训练好的模型"""
    model = CodeSimilarityModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def code_to_embedding(code, builder, model, device):
    """将代码转换为嵌入向量"""
    graph = builder.parse_code_to_graph(code)
    if graph is None:
        return None
    
    data = builder.convert_to_pyg_data(graph)
    if data is None:
        return None
    
    data = data.to(device)
    with torch.no_grad():
        embedding = model(data).cpu().numpy()
    return embedding

def calculate_similarity(embedding1, embedding2):
    """计算两个嵌入向量的余弦相似度"""
    if embedding1 is None or embedding2 is None:
        return None
    
    # 余弦相似度 = 1 - 余弦距离
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型路径
    model_path = 'code_similarity_model.pth'  # 请替换为实际的模型路径
    
    # 初始化代码图构建器
    builder = CodeGraphBuilder()
    
    # 加载模型
    model = load_model(model_path, device)
    print("模型加载完成")
    
    # 示例代码片段，可替换为实际需要比较的代码
    code1 = """
def calculate_sum(a, b):
    if a > 0 and b > 0:
        return a + b
    else:
        return 0
    """
    
    code2 = """
def add_numbers(x, y):
    if x > 0 or y > 0:
        return x + y
    else:
        return 0
    """
    
    print("正在处理代码1...")
    embedding1 = code_to_embedding(code1, builder, model, device)
    
    print("正在处理代码2...")
    embedding2 = code_to_embedding(code2, builder, model, device)
    
    if embedding1 is not None and embedding2 is not None:
        similarity = calculate_similarity(embedding1, embedding2)
        print(f"代码1和代码2的余弦相似度: {similarity:.4f}")
    else:
        print("无法计算相似度，至少有一个代码片段无法解析")

if __name__ == "__main__":
    main()