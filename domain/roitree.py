"""
domain/roitree.py
ROI Tree data structure and utilities
"""
from typing import List, Dict, Optional, Tuple, Any
import re

from domain.schemas import ROIData, NodeRiskReturn


class ROINode:
    """ROIツリーのノードを表すクラス"""
    
    def __init__(self, node_id: str, name: str, value: Optional[str] = None):
        """
        ROIノードの初期化
        
        Args:
            node_id: ノードID
            name: ノード名
            value: 数値（オプション）
        """
        self.id = node_id
        self.name = name
        self.value = value
        self.children = []
        self.parent = None
        
        # 新規: ROI関連フィールド
        self.cost = None
        self.benefit = None
        self.roi_percentage = None
        self.implementation_period = None
        self.priority = None
    
    def add_child(self, child: 'ROINode') -> None:
        """
        子ノードを追加する
        
        Args:
            child: 追加する子ノード
        """
        self.children.append(child)
        child.parent = self
    
    def get_full_mermaid(self) -> str:
        """
        Mermaid記法の完全なツリー図を生成する
        
        Returns:
            Mermaid記法の文字列
        """
        lines = ["flowchart TD"]
        
        # ノード定義を生成
        node_definitions = []
        node_connections = []
        
        def traverse(node, connections):
            # ノード定義
            node_label = f'{node.name}'
            if node.value:
                node_label += f' ({node.value})'
                
            # 新規: ROI情報が含まれている場合はそれも表示
            if node.roi_percentage is not None:
                node_label += f' [ROI:{node.roi_percentage:.1f}%]'
                
            node_definitions.append(f'    {node.id}["{node_label}"]')
            
            # エッジ（接続）を追加
            for child in node.children:
                connections.append(f'    {node.id} --> {child.id}')
                traverse(child, connections)
        
        traverse(self, node_connections)
        
        # 定義とエッジを結合
        return "\n".join(lines + node_definitions + node_connections)
    
    # 新規: ROI関連の情報を設定する
    def set_roi_data(self, cost: float, benefit: float, implementation_period: str = None, priority: int = None) -> None:
        """
        ROI関連の情報を設定する
        
        Args:
            cost: コスト
            benefit: 効果
            implementation_period: 実装期間（オプション）
            priority: 優先度（オプション）
        """
        self.cost = cost
        self.benefit = benefit
        
        # ROIを計算（コストが0の場合は無限大とする）
        if cost > 0:
            self.roi_percentage = (benefit - cost) / cost * 100
        else:
            self.roi_percentage = float('inf')
            
        self.implementation_period = implementation_period
        self.priority = priority
    
    # 新規: ROI情報を取得する
    def get_roi_data(self) -> Optional[ROIData]:
        """
        ノードのROI情報を取得する
        
        Returns:
            ROI情報。ROI関連フィールドがNoneの場合はNoneを返す
        """
        if self.cost is None or self.benefit is None:
            return None
            
        return ROIData(
            node_id=self.id,
            node_name=self.name,
            cost=self.cost,
            benefit=self.benefit,
            roi_percentage=self.roi_percentage,
            implementation_period=self.implementation_period,
            priority=self.priority
        )


def mermaid_to_roi_tree(mermaid_text: str) -> Optional[ROINode]:
    """
    Mermaid記法からROIツリーを構築する
    
    Args:
        mermaid_text: Mermaid記法の文字列
        
    Returns:
        ROIツリーのルートノード、解析に失敗した場合はNone
    """
    # 前処理: 先頭行が 'graph TD' または 'flowchart TD' であれば削除
    lines = mermaid_text.strip().split('\n')
    if lines[0].startswith('graph TD') or lines[0].startswith('flowchart TD'):
        lines = lines[1:]
    
    # ノード定義と接続を分離
    node_defs = {}
    connections = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
            
        # ノード定義をパース（例: node1["テキスト"]）
        node_match = re.match(r'(\w+)\s*\[\s*"([^"]+)"\s*\]', line)
        if node_match:
            node_id = node_match.group(1)
            node_text = node_match.group(2)
            
            # 数値を抽出（例: テキスト (100)）
            value_match = re.search(r'\(([^)]+)\)', node_text)
            if value_match:
                value = value_match.group(1)
                name = node_text.replace(f' ({value})', '')
            else:
                value = None
                name = node_text
                
            # 新規: ROI情報を抽出
            roi_match = re.search(r'\[ROI:([^%]+)%\]', node_text)
            if roi_match:
                roi_percentage = float(roi_match.group(1))
                name = name.replace(f' [ROI:{roi_percentage}%]', '')
            else:
                roi_percentage = None
                
            node = ROINode(node_id, name, value)
            
            # 新規: ROI情報がある場合は設定
            if roi_percentage is not None:
                # ROI情報のみからコストと効果を計算できないため仮のデータを設定
                node.roi_percentage = roi_percentage
                
            node_defs[node_id] = node
        else:
            # 接続をパース（例: node1 --> node2）
            connection_match = re.match(r'(\w+)\s*-->\s*(\w+)', line)
            if connection_match:
                parent_id = connection_match.group(1)
                child_id = connection_match.group(2)
                connections.append((parent_id, child_id))
    
    # ツリーを構築
    if not node_defs:
        return None
        
    # 接続に従ってノードを結合
    for parent_id, child_id in connections:
        if parent_id in node_defs and child_id in node_defs:
            node_defs[parent_id].add_child(node_defs[child_id])
    
    # ルートノードを探す（親がないノード）
    root_candidates = {}
    for node_id, node in node_defs.items():
        if node.parent is None:
            root_candidates[node_id] = node
    
    # ルートノードが複数ある場合は、接続で子になっていないノードをルートとする
    root_nodes = []
    for parent_id, child_id in connections:
        if parent_id in root_candidates and child_id in root_candidates:
            del root_candidates[child_id]
    
    # 残ったルート候補からルートノードを選択（複数ある場合は最初のものを使用）
    root_nodes = list(root_candidates.values())
    if not root_nodes:
        return None
        
    return root_nodes[0]


def get_leaf_nodes(root_node: ROINode) -> List[ROINode]:
    """
    ROIツリーの末端ノード（子を持たないノード）を取得する
    
    Args:
        root_node: ROIツリーのルートノード
        
    Returns:
        末端ノードのリスト
    """
    leaf_nodes = []
    
    def traverse(node):
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children:
                traverse(child)
    
    traverse(root_node)
    return leaf_nodes


def create_default_roi_tree() -> ROINode:
    """
    デフォルトのROIツリーを作成する
    
    Returns:
        デフォルトROIツリーのルートノード
    """
    root = ROINode("root", "ROI")
    
    cost_reduction = ROINode("cost", "コスト削減")
    revenue_increase = ROINode("revenue", "売上拡大")
    
    root.add_child(cost_reduction)
    root.add_child(revenue_increase)
    
    return root


# 新規: ノードのコストと効果を抽出する
def extract_cost_benefit_from_node(node_text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    ノードテキストからコストと効果を抽出する
    
    Args:
        node_text: ノードテキスト
        
    Returns:
        (コスト, 効果)のタプル。抽出できない場合はNone
    """
    cost_pattern = r'コスト[:：]?(\d+[,.]*\d*)\s*[億万千]?円'
    benefit_pattern = r'効果[:：]?(\d+[,.]*\d*)\s*[億万千]?円'
    
    # コスト抽出
    cost_match = re.search(cost_pattern, node_text)
    cost = None
    if cost_match:
        cost_str = cost_match.group(1).replace(',', '')
        try:
            cost = float(cost_str)
            if '億円' in node_text:
                cost *= 100000000
            elif '万円' in node_text:
                cost *= 10000
            elif '千円' in node_text:
                cost *= 1000
        except ValueError:
            pass
    
    # 効果抽出
    benefit_match = re.search(benefit_pattern, node_text)
    benefit = None
    if benefit_match:
        benefit_str = benefit_match.group(1).replace(',', '')
        try:
            benefit = float(benefit_str)
            if '億円' in node_text:
                benefit *= 100000000
            elif '万円' in node_text:
                benefit *= 10000
            elif '千円' in node_text:
                benefit *= 1000
        except ValueError:
            pass
    
    return cost, benefit


# 新規: ROIツリーを計算して更新する
def calculate_roi_for_tree(root_node: ROINode) -> None:
    """
    ROIツリー全体のROIを計算して更新する
    
    Args:
        root_node: ROIツリーのルートノード
    """
    # 葉ノードから上に遡りながらROIを計算
    def traverse(node):
        if not node.children:
            # 末端ノードは既に値が設定されているはず
            return node.cost or 0, node.benefit or 0
        else:
            total_cost = 0
            total_benefit = 0
            
            # 子ノードのコストと効果を合計
            for child in node.children:
                child_cost, child_benefit = traverse(child)
                total_cost += child_cost
                total_benefit += child_benefit
            
            # 非末端ノードのROIを計算
            node.cost = total_cost
            node.benefit = total_benefit
            if total_cost > 0:
                node.roi_percentage = (total_benefit - total_cost) / total_cost * 100
            else:
                node.roi_percentage = float('inf') if total_benefit > 0 else 0
                
            return total_cost, total_benefit
    
    traverse(root_node)


# 新規: ROIツリーからリスク-リターン情報を生成する
def generate_risk_return_analysis(root_node: ROINode) -> List[NodeRiskReturn]:
    """
    ROIツリーからノードごとのリスク-リターン分析を生成する
    
    Args:
        root_node: ROIツリーのルートノード
        
    Returns:
        ノードごとのリスク-リターン情報のリスト
    """
    result = []
    
    # まず末端ノードを取得
    leaf_nodes = get_leaf_nodes(root_node)
    
    # 各ノードのROI値を抽出
    roi_values = [node.roi_percentage for node in leaf_nodes if node.roi_percentage is not None]
    
    # ROI値の最大値と最小値を取得
    max_roi = max(roi_values) if roi_values else 100
    min_roi = min(roi_values) if roi_values else 0
    
    # 各末端ノードのリスク-リターン情報を生成
    for node in leaf_nodes:
        if node.roi_percentage is None:
            continue
            
        # ROIからリターンレベルを計算
        if max_roi == min_roi:
            return_level = 50  # すべて同じROIの場合は中間値
        else:
            return_level = 100 * (node.roi_percentage - min_roi) / (max_roi - min_roi)
            
        # コストからリスクレベルを計算（コストが高いほどリスクも高い）
        if node.cost is None:
            risk_level = 50  # コスト不明の場合は中間値
        else:
            # 所有する末端ノードの中でのコスト相対値
            max_cost = max([n.cost for n in leaf_nodes if n.cost is not None]) if any(n.cost is not None for n in leaf_nodes) else 1
            risk_level = 100 * (node.cost / max_cost) if max_cost > 0 else 0
        
        # 実装難易度は優先度の逆数に比例（優先度が高いほど実装は容易）
        implementation_difficulty = 100 - (node.priority * 20) if node.priority is not None else 50
        
        # 戦略的重要度はROIとリターンの積に比例
        strategic_importance = (return_level * node.roi_percentage) / 100 if node.roi_percentage is not None else 50
        
        result.append(NodeRiskReturn(
            node_id=node.id,
            node_name=node.name,
            risk_level=risk_level,
            return_level=return_level,
            implementation_difficulty=implementation_difficulty,
            strategic_importance=strategic_importance,
            roi_percentage=node.roi_percentage,
            estimated_timeframe=node.implementation_period or "不明"
        ))
    
    return result