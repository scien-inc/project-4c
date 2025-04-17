"""
domain/roitree.py
Enhanced ROI Tree data structure with DX tool support and parameter tracking
"""
from typing import List, Dict, Optional, Tuple, Any, Set
import re

from domain.schemas import ROIData, NodeRiskReturn


class ROINode:
    """ROIツリーのノードを表すクラス（DXツール対応とパラメータ追跡機能を追加）"""
    
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
        
        # ROI関連フィールド
        self.cost = None
        self.benefit = None
        self.roi_percentage = None
        self.implementation_period = None
        self.priority = None
        
        # DXツール関連フィールド
        self.dx_tools = []
        self.required_parameters = []
        self.granularity_issue = None
    
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
                
            # ROI情報が含まれている場合はそれも表示
            roi_info = []
            if node.roi_percentage is not None:
                roi_info.append(f"ROI:{node.roi_percentage:.1f}%")
            if node.cost is not None:
                if node.cost >= 100000000:  # 1億円以上
                    roi_info.append(f"コスト:{node.cost/100000000:.1f}億円")
                elif node.cost >= 10000:  # 1万円以上
                    roi_info.append(f"コスト:{node.cost/10000:.1f}万円")
                else:
                    roi_info.append(f"コスト:{node.cost:.0f}円")
            if node.benefit is not None:
                if node.benefit >= 100000000:  # 1億円以上
                    roi_info.append(f"効果:{node.benefit/100000000:.1f}億円")
                elif node.benefit >= 10000:  # 1万円以上
                    roi_info.append(f"効果:{node.benefit/10000:.1f}万円")
                else:
                    roi_info.append(f"効果:{node.benefit:.0f}円")
            if node.implementation_period:
                roi_info.append(f"期間:{node.implementation_period}")
            if node.priority:
                roi_info.append(f"優先度:{node.priority}")
            
            # # DXツール情報を追加
            # if node.dx_tools and not node.children:  # 末端ノードの場合のみ
            #     dx_info = f"DXツール:{','.join(node.dx_tools)}"
            #     roi_info.append(dx_info)
            
            # # パラメータ情報を追加
            # if node.required_parameters and not node.children:  # 末端ノードの場合のみ
            #     param_info = f"パラメータ:{','.join(node.required_parameters)}"
            #     roi_info.append(param_info)
            
            if roi_info:
                node_label += f" [{'/'.join(roi_info)}]"
                
            node_definitions.append(f'    {node.id}["{node_label}"]')
            
            # エッジ（接続）を追加
            for child in node.children:
                connections.append(f'    {node.id} --> {child.id}')
                traverse(child, connections)
        
        traverse(self, node_connections)
        
        # 定義とエッジを結合
        return "\n".join(lines + node_definitions + node_connections)
    
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
    
    def set_dx_tools(self, dx_tools: List[str]) -> None:
        """
        このノードに適用可能なDXツールリストを設定する
        
        Args:
            dx_tools: DXツールのリスト
        """
        self.dx_tools = dx_tools
    
    def set_required_parameters(self, parameters: List[str]) -> None:
        """
        このノードのROI計算に必要なパラメータリストを設定する
        
        Args:
            parameters: パラメータのリスト
        """
        self.required_parameters = parameters
    
    def set_granularity_issue(self, issue: Dict[str, Any]) -> None:
        """
        このノードの粒度に関する問題を設定する
        
        Args:
            issue: 粒度の問題を表す辞書
        """
        self.granularity_issue = issue


def mermaid_to_roi_tree(mermaid_text: str) -> Optional[ROINode]:
    """
    Mermaid記法からROIツリーを構築する（パラメータ情報とDXツール情報を抽出）
    
    Args:
        mermaid_text: Mermaid記法の文字列
        
    Returns:
        ROIツリーのルートノード、解析に失敗した場合はNone
    """
    # 前処理: 先頭行が 'graph TD' または 'flowchart TD' であれば削除
    lines = mermaid_text.strip().split('\n')
    if lines and (lines[0].startswith('graph TD') or lines[0].startswith('flowchart TD')):
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
                
            # ROI情報を抽出
            roi_info = {}
            roi_match = re.search(r'\[(.*?)\]', node_text)
            if roi_match:
                roi_text = roi_match.group(1)
                name = name.replace(f' [{roi_text}]', '')
                
                # ROIの各部分を抽出
                roi_parts = roi_text.split('/')
                for part in roi_parts:
                    if part.startswith('ROI:'):
                        try:
                            roi_info['roi_percentage'] = float(part.replace('ROI:', '').replace('%', ''))
                        except ValueError:
                            pass
                    elif part.startswith('コスト:'):
                        cost_text = part.replace('コスト:', '')
                        multiplier = 1
                        if '億円' in cost_text:
                            multiplier = 100000000
                            cost_text = cost_text.replace('億円', '')
                        elif '万円' in cost_text:
                            multiplier = 10000
                            cost_text = cost_text.replace('万円', '')
                        elif '千円' in cost_text:
                            multiplier = 1000
                            cost_text = cost_text.replace('千円', '')
                        else:
                            cost_text = cost_text.replace('円', '')
                            
                        try:
                            roi_info['cost'] = float(cost_text) * multiplier
                        except ValueError:
                            pass
                    elif part.startswith('効果:'):
                        benefit_text = part.replace('効果:', '')
                        multiplier = 1
                        if '億円' in benefit_text:
                            multiplier = 100000000
                            benefit_text = benefit_text.replace('億円', '')
                        elif '万円' in benefit_text:
                            multiplier = 10000
                            benefit_text = benefit_text.replace('万円', '')
                        elif '千円' in benefit_text:
                            multiplier = 1000
                            benefit_text = benefit_text.replace('千円', '')
                        else:
                            benefit_text = benefit_text.replace('円', '')
                            
                        try:
                            roi_info['benefit'] = float(benefit_text) * multiplier
                        except ValueError:
                            pass
                    elif part.startswith('期間:'):
                        roi_info['implementation_period'] = part.replace('期間:', '')
                    elif part.startswith('優先度:'):
                        try:
                            roi_info['priority'] = int(part.replace('優先度:', ''))
                        except ValueError:
                            pass
                    # DXツール情報を抽出
                    elif part.startswith('DXツール:'):
                        dx_tools = part.replace('DXツール:', '').split(',')
                        roi_info['dx_tools'] = [tool.strip() for tool in dx_tools]
                    # パラメータ情報を抽出
                    elif part.startswith('パラメータ:'):
                        parameters = part.replace('パラメータ:', '').split(',')
                        roi_info['parameters'] = [param.strip() for param in parameters]
            
            node = ROINode(node_id, name, value)
            
            # ROI情報があれば設定
            if 'roi_percentage' in roi_info:
                node.roi_percentage = roi_info['roi_percentage']
            if 'cost' in roi_info:
                node.cost = roi_info['cost']
            if 'benefit' in roi_info:
                node.benefit = roi_info['benefit']
            if 'implementation_period' in roi_info:
                node.implementation_period = roi_info['implementation_period']
            if 'priority' in roi_info:
                node.priority = roi_info['priority']
            if 'dx_tools' in roi_info:
                node.dx_tools = roi_info['dx_tools']
            if 'parameters' in roi_info:
                node.required_parameters = roi_info['parameters']
                
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


def get_leaf_nodes_with_dx_tools(root_node: ROINode) -> List[ROINode]:
    """
    DXツール情報を持つ末端ノードを取得する
    
    Args:
        root_node: ROIツリーのルートノード
        
    Returns:
        DXツール情報を持つ末端ノードのリスト
    """
    leaf_nodes = get_leaf_nodes(root_node)
    return [node for node in leaf_nodes if node.dx_tools]


def get_nodes_with_parameters(root_node: ROINode) -> List[ROINode]:
    """
    パラメータ情報を持つノードを取得する
    
    Args:
        root_node: ROIツリーのルートノード
        
    Returns:
        パラメータ情報を持つノードのリスト
    """
    nodes_with_parameters = []
    
    def traverse(node):
        if node.required_parameters:
            nodes_with_parameters.append(node)
        for child in node.children:
            traverse(child)
    
    traverse(root_node)
    return nodes_with_parameters


def get_common_parameters(root_node: ROINode) -> Set[str]:
    """
    ROIツリー全体で共通して使用されるパラメータを取得する
    
    Args:
        root_node: ROIツリーのルートノード
        
    Returns:
        共通パラメータのセット
    """
    nodes_with_parameters = get_nodes_with_parameters(root_node)
    
    # 複数のノードで使用されるパラメータをカウント
    parameter_counts = {}
    for node in nodes_with_parameters:
        for param in node.required_parameters:
            if param in parameter_counts:
                parameter_counts[param] += 1
            else:
                parameter_counts[param] = 1
    
    # 複数のノードで使用されるパラメータを共通パラメータとみなす
    common_threshold = max(2, len(nodes_with_parameters) // 3)  # 全ノードの1/3以上で使用されるパラメータ
    common_parameters = {param for param, count in parameter_counts.items() if count >= common_threshold}
    
    # 明らかに共通と思われるパラメータを追加
    always_common = {'人件費', '時給', '工数', '工数単価', '年間稼働日数', '月間稼働時間'}
    common_parameters.update({param for param in parameter_counts.keys() if any(common in param.lower() for common in always_common)})
    
    return common_parameters


def extract_node_specific_parameters(node: ROINode, common_parameters: Set[str]) -> List[str]:
    """
    ノード固有のパラメータを抽出する
    
    Args:
        node: 対象ノード
        common_parameters: 共通パラメータのセット
        
    Returns:
        ノード固有のパラメータリスト
    """
    if not node.required_parameters:
        return []
    
    return [param for param in node.required_parameters if param not in common_parameters]


def create_default_roi_tree() -> ROINode:
    """
    デフォルトのROIツリーを作成する（より多くの分岐を持つ複雑な構造）
    
    Returns:
        デフォルトROIツリーのルートノード
    """
    root = ROINode("root", "ROI")
    
    cost_reduction = ROINode("cost", "コスト削減")
    revenue_increase = ROINode("revenue", "売上拡大")
    
    root.add_child(cost_reduction)
    root.add_child(revenue_increase)
    
    # コスト削減の下位カテゴリ（4つのサブカテゴリ）
    process_automation = ROINode("process", "業務プロセス自動化")
    resource_optimization = ROINode("resource", "リソース最適化")
    it_infrastructure = ROINode("infra", "IT基盤最適化")
    operational_efficiency = ROINode("ops", "運用効率化")
    
    cost_reduction.add_child(process_automation)
    cost_reduction.add_child(resource_optimization)
    cost_reduction.add_child(it_infrastructure)
    cost_reduction.add_child(operational_efficiency)
    
    # 業務プロセス自動化の下位項目（3つ）
    rpa_implementation = ROINode("rpa", "RPAによる定型業務自動化")
    ai_document_processing = ROINode("ai_doc", "AI文書処理自動化")
    workflow_digitization = ROINode("workflow", "ワークフローのデジタル化")
    
    process_automation.add_child(rpa_implementation)
    process_automation.add_child(ai_document_processing)
    process_automation.add_child(workflow_digitization)
    
    # リソース最適化の下位項目（3つ）
    inventory_management = ROINode("inventory", "在庫管理システム最適化")
    workforce_planning = ROINode("workforce", "人員配置最適化システム")
    energy_optimization = ROINode("energy", "エネルギー使用効率化")
    
    resource_optimization.add_child(inventory_management)
    resource_optimization.add_child(workforce_planning)
    resource_optimization.add_child(energy_optimization)
    
    # IT基盤最適化の下位項目（3つ）
    cloud_migration = ROINode("cloud", "クラウド移行によるインフラコスト削減")
    license_optimization = ROINode("license", "ソフトウェアライセンス最適化")
    server_consolidation = ROINode("server", "サーバー統合・仮想化")
    
    it_infrastructure.add_child(cloud_migration)
    it_infrastructure.add_child(license_optimization)
    it_infrastructure.add_child(server_consolidation)
    
    # 運用効率化の下位項目（3つ）
    predictive_maintenance = ROINode("maintenance", "予知保全システム導入")
    remote_monitoring = ROINode("monitoring", "遠隔監視システム構築")
    facility_automation = ROINode("facility", "施設管理自動化")
    
    operational_efficiency.add_child(predictive_maintenance)
    operational_efficiency.add_child(remote_monitoring)
    operational_efficiency.add_child(facility_automation)
    
    # 売上拡大の下位カテゴリ（4つのサブカテゴリ）
    customer_experience = ROINode("cx", "顧客体験向上")
    market_expansion = ROINode("market", "市場拡大")
    product_innovation = ROINode("product", "商品・サービス革新")
    sales_effectiveness = ROINode("sales", "販売効率向上")
    
    revenue_increase.add_child(customer_experience)
    revenue_increase.add_child(market_expansion)
    revenue_increase.add_child(product_innovation)
    revenue_increase.add_child(sales_effectiveness)
    
    # 顧客体験向上の下位項目（3つ）
    chatbot_implementation = ROINode("chatbot", "AIチャットボット導入")
    personalization_engine = ROINode("personalize", "パーソナライゼーションエンジン実装")
    omnichannel_integration = ROINode("omnichannel", "オムニチャネル連携基盤構築")
    
    customer_experience.add_child(chatbot_implementation)
    customer_experience.add_child(personalization_engine)
    customer_experience.add_child(omnichannel_integration)
    
    # 市場拡大の下位項目（3つ）
    digital_marketing = ROINode("digital_mkt", "デジタルマーケティング強化")
    global_ecommerce = ROINode("ecommerce", "グローバルEコマース展開")
    new_segment_analytics = ROINode("segment", "新規顧客セグメント分析・開拓")
    
    market_expansion.add_child(digital_marketing)
    market_expansion.add_child(global_ecommerce)
    market_expansion.add_child(new_segment_analytics)
    
    # 商品・サービス革新の下位項目（3つ）
    predictive_analytics = ROINode("pred_analytics", "予測分析による製品開発")
    digital_product_dev = ROINode("digital_prod", "デジタル製品・サービス開発")
    ai_innovation = ROINode("ai_innov", "AI活用による商品革新")
    
    product_innovation.add_child(predictive_analytics)
    product_innovation.add_child(digital_product_dev)
    product_innovation.add_child(ai_innovation)
    
    # 販売効率向上の下位項目（3つ）
    sales_automation = ROINode("sales_auto", "営業プロセス自動化")
    customer_data_platform = ROINode("cdp", "顧客データプラットフォーム構築")
    pricing_optimization = ROINode("pricing", "価格最適化システム")
    
    sales_effectiveness.add_child(sales_automation)
    sales_effectiveness.add_child(customer_data_platform)
    sales_effectiveness.add_child(pricing_optimization)
    
    # 末端ノードに必要なパラメータとDXツールを設定
    rpa_implementation.set_required_parameters(["初期導入コスト", "月額費用", "自動化対象プロセス数", "プロセスあたり工数", "人件費単価"])
    rpa_implementation.set_dx_tools(["UiPath", "Automation Anywhere", "Blue Prism"])
    
    chatbot_implementation.set_required_parameters(["初期導入コスト", "月額費用", "問い合わせ削減率", "月間問い合わせ数", "問い合わせ対応時間", "人件費単価"])
    chatbot_implementation.set_dx_tools(["Dialogflow", "IBM Watson Assistant", "Amazon Lex"])
    
    cloud_migration.set_required_parameters(["現行インフラコスト", "クラウド移行コスト", "移行後月額費用", "運用工数削減量", "人件費単価"])
    cloud_migration.set_dx_tools(["AWS", "Microsoft Azure", "Google Cloud Platform"])
    
    digital_marketing.set_required_parameters(["マーケティング投資額", "顧客獲得単価", "コンバージョン率向上率", "顧客生涯価値"])
    digital_marketing.set_dx_tools(["Adobe Experience Cloud", "HubSpot", "Salesforce Marketing Cloud"])
    
    predictive_analytics.set_required_parameters(["分析システム導入コスト", "データサイエンティスト人件費", "製品開発期間短縮率", "新製品売上予測"])
    predictive_analytics.set_dx_tools(["Tableau", "Power BI", "DataRobot"])
    
    # 他の末端ノードにもパラメータとDXツールを設定（例示的に一部のみ）
    ai_document_processing.set_required_parameters(["OCRシステム導入コスト", "月額利用料", "書類処理数", "処理時間削減率", "人件費単価"])
    ai_document_processing.set_dx_tools(["ABBYY FineReader", "Google Document AI", "Microsoft Azure Form Recognizer"])
    
    workflow_digitization.set_required_parameters(["ワークフローシステム導入コスト", "月額費用", "デジタル化対象プロセス数", "プロセス効率化率", "年間処理件数"])
    workflow_digitization.set_dx_tools(["Pega", "ServiceNow", "Appian"])
    
    inventory_management.set_required_parameters(["システム導入コスト", "在庫削減率", "倉庫スペースコスト", "在庫金額", "発注業務効率化率"])
    inventory_management.set_dx_tools(["SAP Inventory Management", "Oracle SCM", "Manhattan Associates"])
    
    personalization_engine.set_required_parameters(["システム導入費用", "運用コスト", "コンバージョン率向上率", "顧客単価向上率", "訪問者数"])
    personalization_engine.set_dx_tools(["Adobe Target", "Dynamic Yield", "Optimizely"])
    
    return root


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


def extract_required_parameters_from_node(node_text: str) -> List[str]:
    """
    ノードテキストから必要なパラメータを抽出する
    
    Args:
        node_text: ノードテキスト
        
    Returns:
        必要なパラメータのリスト
    """
    # パラメータパターン（例: 必要パラメータ:初期コスト,月額費用,削減工数）
    params_pattern = r'(?:必要パラメータ|パラメータ)[:：]([^[\]]+)'
    
    params_match = re.search(params_pattern, node_text)
    if params_match:
        params_text = params_match.group(1)
        # カンマで分割してトリム
        return [param.strip() for param in params_text.split(',')]
    
    return []


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