import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, GlobalAttention
from torch_geometric.utils import add_self_loops
import torch.optim as optim
from torch_geometric.data import Data, DataLoader, Batch
import ast
import networkx as nx
from transformers import BertModel, BertTokenizer
import json
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os

# ====================== 代码图构建模块 ======================
class CodeGraphBuilder:
    def __init__(self, model_name='bert-base-uncased'):
        # 初始化BERT模型和tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_model.eval()  # 设置为评估模式
        
    @torch.no_grad()  # 不计算梯度，加速推理
    def get_bert_embedding(self, text):
        """获取文本的BERT嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        outputs = self.bert_model(**inputs)
        # 使用[CLS]标记的输出作为文本嵌入
        return outputs.last_hidden_state[:, 0, :].squeeze(0)  # [768]
    
    def parse_code_to_graph(self, code):
        """将代码解析为带语义嵌入的图结构"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"语法错误: {e}")
            return None
        
        graph = nx.DiGraph()
        
        # 获取节点标签和类型
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
        
        # 递归遍历AST并构建图
        def traverse_ast(node, parent=None):
            node_id = id(node)
            node_label = get_node_label(node)
            node_embedding = self.get_bert_embedding(node_label).cpu().numpy()
            
            # 添加节点到图
            graph.add_node(node_id, label=node_label, embedding=node_embedding)
            
            # 添加边
            if parent is not None:
                edge_type = get_edge_type(node)
                edge_label = f"{edge_type}: {node_label}"
                edge_embedding = self.get_bert_embedding(edge_label).cpu().numpy()
                graph.add_edge(parent, node_id, label=edge_label, embedding=edge_embedding, type=edge_type)
            
            # 递归处理子节点
            for child in ast.iter_child_nodes(node):
                traverse_ast(child, parent=node_id)
        
        traverse_ast(tree)
        return graph
    
    def convert_to_pyg_data(self, graph, label=None):
        """将NetworkX图转换为PyTorch Geometric数据对象"""
        if graph is None or len(graph.nodes) == 0:
            return None
            
        # 节点特征
        node_features = []
        node_ids = list(graph.nodes)
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        for node_id in node_ids:
            node_features.append(graph.nodes[node_id]['embedding'])
        
        # 边索引和边特征
        edge_index = []
        edge_features = []
        
        for u, v, data in graph.edges(data=True):
            edge_index.append([id_to_idx[u], id_to_idx[v]])
            edge_features.append(data['embedding'])
        
        # 转换为张量
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float)
        
        # 创建PyG数据对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        
        return data

# ====================== 边感知GNN模型 ======================
class EdgeAwareGNN(MessagePassing):
    def __init__(self, node_dim=768, edge_dim=768, hidden_dim=256, dropout=0.1):
        super(EdgeAwareGNN, self).__init__(aggr='add')
        
        # 节点和边的投影层
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # 消息传递网络
        self.message_fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新网络
        self.update_fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # 添加自环边
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # 消息传递
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_fc(torch.cat([x, out], dim=1))
        return out

class CodeSimilarityModel(nn.Module):
    def __init__(self, node_dim=768, edge_dim=768, hidden_dim=256, graph_dim=128, dropout=0.1):
        super(CodeSimilarityModel, self).__init__()
        
        # GNN层
        self.gnn1 = EdgeAwareGNN(node_dim, edge_dim, hidden_dim, dropout)
        self.gnn2 = EdgeAwareGNN(hidden_dim, edge_dim, hidden_dim, dropout)
        
        # 节点注意力池化
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        self.node_pooling = GlobalAttention(gate_nn=self.node_gate)
        
        # 边特征池化 - 关键修复：使用edge_dim作为输入维度
        self.edge_pooling = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim//2),  # 输入维度改为edge_dim(768)，输出为hidden_dim//2(128)
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 图嵌入融合
        self.graph_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.Dropout(dropout)
        )
        
        # 对比学习温度参数
        self.temperature = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 图卷积
        h1 = self.gnn1(x, edge_index, edge_attr)
        h2 = self.gnn2(h1, edge_index, edge_attr)
        
        # 节点池化
        node_embedding = self.node_pooling(h2, batch)
        
        # 边池化 - 使用源节点的batch索引确定边所属图
        edge_batch = batch[edge_index[0]]  # 通过源节点的batch索引确定边所属图
        
        # 确保edge_batch长度与edge_attr一致
        if len(edge_batch) > edge_attr.size(0):
            edge_batch = edge_batch[:edge_attr.size(0)]
        elif len(edge_batch) < edge_attr.size(0):
            # 处理边数多于节点数的情况（理论上边数不应超过节点数*2）
            edge_batch = torch.cat([edge_batch, edge_batch[-1:].repeat(edge_attr.size(0) - len(edge_batch))])
        
        edge_embedding = global_add_pool(edge_attr, edge_batch)  # [batch_size, edge_dim=768]
        edge_embedding = self.edge_pooling(edge_embedding)  # [batch_size, hidden_dim//2=128]
        
        graph_embedding = torch.cat([node_embedding, edge_embedding], dim=1)  # [batch_size, 256+128=384]
        graph_embedding = self.graph_fusion(graph_embedding)  # [batch_size, graph_dim=128]
        
        return graph_embedding
    
    def info_nce_loss(self, batch):
        batch_size = batch.num_graphs
        
        embeddings = self.forward(batch)  # 直接处理整个批次
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        sim_matrix = torch.matmul(embeddings, embeddings.transpose(0, 1)) / self.temperature.exp()
        
        labels = torch.eye(batch_size, device=sim_matrix.device)
        
        mask = ~torch.eye(batch_size, dtype=bool, device=sim_matrix.device)
        
        pos_mask = labels.bool()
        pos_sim = sim_matrix[pos_mask]
        
        neg_sim = sim_matrix[mask].view(batch_size, batch_size-1)
        
        numerator = torch.exp(pos_sim)
        denominator = torch.sum(torch.exp(neg_sim), dim=1)
        loss = -torch.mean(torch.log(numerator / denominator))
        
        return loss
    
    def compute_similarity(self, code1, code2):
        embedding1 = self.forward(code1)
        embedding2 = self.forward(code2)
        
        similarity = F.cosine_similarity(embedding1, embedding2)
        return similarity

def evaluate_similarity(model, test_loader, device, n_clusters=10):
    model.eval()
    all_embeddings = []
    all_data = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            embeddings = model.forward(batch)
            all_embeddings.append(embeddings.cpu())
            
            for i in range(batch.num_graphs):
                all_data.append(batch.get_example(i))
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    n_samples = all_embeddings.shape[0]
    print(f"Evaluating {n_samples} code embeddings...")
    
    embeddings_np = all_embeddings.numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_np)
    
    # 计算聚类评估指标
    silhouette = None
    ch_score = None
    try:
        if n_clusters < n_samples:  # 避免样本数少于聚类数
            silhouette = silhouette_score(embeddings_np, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings_np, cluster_labels)
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Calinski-Harabasz Score: {ch_score:.4f}")
        else:
            print(f"样本数({n_samples})少于聚类数({n_clusters})，无法计算聚类指标")
    except ValueError as e:
        print(e)
    
    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T)
    norms = torch.norm(all_embeddings, dim=1, keepdim=True)
    sim_matrix = sim_matrix / (norms * norms.T)
    sim_matrix = sim_matrix.numpy()
    
    # 提取上三角矩阵（避免重复计算）
    sim_values = sim_matrix[np.triu_indices(n_samples, k=1)]
    
    # 分析相似度分布
    mean_sim = np.mean(sim_values)
    std_sim = np.std(sim_values)
    max_sim = np.max(sim_values)
    min_sim = np.min(sim_values)
    print(f"平均相似度: {mean_sim:.4f}")
    print(f"相似度标准差: {std_sim:.4f}")
    print(f"最大相似度: {max_sim:.4f}")
    print(f"最小相似度: {min_sim:.4f}")
    
    # 绘制相似度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(sim_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(mean_sim, color='r', linestyle='dashed', linewidth=2, label=f'均值: {mean_sim:.4f}')
    plt.title('代码嵌入相似度分布')
    plt.xlabel('余弦相似度')
    plt.ylabel('频率')
    plt.legend()
    plt.tight_layout()
    if not os.path.exists('evaluation_plots'):
        os.makedirs('evaluation_plots')
    plt.savefig('evaluation_plots/similarity_distribution.png')
    plt.close()
    
    if n_samples > 5:
        query_indices = np.random.choice(n_samples, 5, replace=False)
    else:
        query_indices = list(range(n_samples))
    
    for idx in query_indices:
        query_emb = all_embeddings[idx:idx+1]
        query_data = all_data[idx]
        
        query_sim = torch.matmul(query_emb, all_embeddings.T).squeeze()
        norms_query = torch.norm(query_emb)
        norms_all = torch.norm(all_embeddings, dim=1)
        query_sim = query_sim / (norms_query * norms_all)
        query_sim = query_sim.numpy()
        
        similar_indices = np.argsort(-query_sim)[1:6] 
        
        print(f"\n查询样本 {idx} 的最近邻:")
        print(f"查询代码相似度分布: {np.mean(query_sim):.4f} ± {np.std(query_sim):.4f}")
        print(f"查询代码相似度最大值: {np.max(query_sim):.4f}, 最小值: {np.min(query_sim):.4f}")
        
        for i, sim_idx in enumerate(similar_indices):
            if sim_idx < len(query_sim):  # 确保索引有效
                sim_score = query_sim[sim_idx]
                print(f"  {i+1}. 相似度: {sim_score:.4f}, 样本索引: {sim_idx}")
    
    
    eval_results = {
        'n_samples': n_samples,
        'cluster_metrics': {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'ch_score': ch_score
        },
        'similarity_stats': {
            'mean': mean_sim,
            'std': std_sim,
            'max': max_sim,
            'min': min_sim
        }
    }
    
    return eval_results

def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        loss = model.info_nce_loss(batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def load_code_samples(file_path, max_samples=None):
    code_samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                js = json.loads(line)
                code_samples.append(js['clean_code'])
            except Exception as e:
                print(e)
    return code_samples

def create_dataset(code_samples, builder):
    dataset = []
    for i, code in enumerate(code_samples):
        graph = builder.parse_code_to_graph(code)
        if graph is not None:
            data = builder.convert_to_pyg_data(graph)
            dataset.append(data)
    return dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    builder = CodeGraphBuilder()
    
    code_samples = load_code_samples('../dataset/python/clean_test.jsonl', max_samples=1000)
    
    dataset = create_dataset(code_samples, builder)
    
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    def collate_fn(batch):
        return Batch.from_data_list(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 初始化模型
    model = CodeSimilarityModel().to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
    
    model_save_path = 'code_similarity_model.pth'
    torch.save(model.state_dict(), model_save_path)
    eval_results = evaluate_similarity(model, test_loader, device)
    
    # 输出评估结果摘要
    print("\n=== 评估结果摘要 ===")
    print(f"评估样本数: {eval_results['n_samples']}")
    print(f"聚类数量: {eval_results['cluster_metrics']['n_clusters']}")
    if eval_results['cluster_metrics']['silhouette_score'] is not None:
        print(f"轮廓系数: {eval_results['cluster_metrics']['silhouette_score']:.4f}")
    if eval_results['cluster_metrics']['ch_score'] is not None:
        print(f"Calinski-Harabasz分数: {eval_results['cluster_metrics']['ch_score']:.4f}")
    print(f"平均相似度: {eval_results['similarity_stats']['mean']:.4f}")
    print(f"相似度标准差: {eval_results['similarity_stats']['std']:.4f}")

if __name__ == "__main__":
    main()