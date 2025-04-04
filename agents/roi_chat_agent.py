"""
agents/roi_chat_agent.py
ROI Chat Agent for interactive data collection and analysis with cost-benefit focus
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import re
import random

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.nlu_agent import NLUAgent
from agents.conversion_agent import ConversionAgent
from domain.schemas import NumericalValue
from domain.roitree import get_leaf_nodes


class ROIChatAgent:
    """
    ROIツリーを対話的に構築するチャットエージェント
    """
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True  # 実際のストリーミング
        )
        
        self.analysis_llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            streaming=False  # 分析には高速レスポンス
        )
        
        self.nlu_agent = NLUAgent(model_name)
        self.conversion_agent = ConversionAgent(model_name)
        
        # チャット用のプロンプト（ROI計算が可能なデータ収集に焦点）
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析のエキスパートです。ユーザーと対話しながら、ROIツリーの各ノードに適切な数値を設定し、ビジネス分析を支援します。

現在のROIツリーには、まだ数値が設定されていないノードがあります。各ノードについて、以下の情報を収集することが重要です：
1. コスト情報 - 実装や対応にかかる費用
2. ベネフィット情報 - 期待される効果や利益
3. 時間的情報 - 実装期間や効果が現れるまでの時間

これにより、各ノードのROI（投資対効果）を計算できます：ROI = (ベネフィット - コスト) / コスト × 100%

レスポンスでは以下を心がけてください：
- コストとベネフィットの両方の情報を収集するよう心掛ける
- 会話は親しみやすく自然な流れを保つ
- 数値情報のリクエストは押し付けがましくならないよう配慮する
- ユーザーが提供した情報を受け止め、適切なフィードバックを提供する
- 具体的な数値を引き出すための適切な質問をする
- 一度に複数のノードの情報を求めないよう注意する

以下のノードについての情報を対話的に収集してください：
{nodes_without_values}

次のステップ：
{next_step}
"""),
            ("human", "{user_message}")
        ])
        
        # 単位変換ヒアリング用のプロンプト（セルフリフレクションを促す）
        self.conversion_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI分析の専門家であり、数値の単位変換が必要な状況です。
ユーザーとの自然な会話の中で、単位変換に必要な追加情報を収集してください。

現在、ユーザーは以下の数値情報を提供しました：
値: {value}
単位: {unit}
ノード: {node_name}

この情報をROI計算に使用するためには、以下の追加情報が必要です：
{required_info}

自然な対話を通じてこれらの情報を収集してください。
押し付けがましくならないよう注意し、ユーザーが回答しやすい質問の仕方を心がけてください。

収集した情報は慎重に評価し、以下を考慮してください：
- 変換の正確性と信頼性
- 前提条件の妥当性
- 代替的な解釈の可能性
- 業界標準や一般的な指標との整合性
"""),
            ("human", "{user_message}")
        ])
        
        # コスト・ベネフィット情報収集用のプロンプト（新規）
        self.cost_benefit_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI分析の専門家であり、特にコストとベネフィットの情報収集に長けています。
ユーザーとの自然な会話の中で、特定のノードに関するコストとベネフィットの情報を収集してください。

現在のノード: {node_name}

このノードについて、以下の情報を収集することが重要です：
1. 実装コスト - どれくらいの費用がかかるか
2. 期待効果/ベネフィット - どれくらいの効果や利益が見込めるか
3. 実装期間 - 効果が出るまでにどれくらいの時間がかかるか

これらの情報が揃えば、ROI = (ベネフィット - コスト) / コスト × 100% の計算が可能になります。

自然な対話を通じてこれらの情報を収集してください。
押し付けがましくならないよう注意し、ユーザーが回答しやすい質問の仕方を心がけてください。
"""),
            ("human", "{user_message}")
        ])
        
        # ノード優先度ヒアリング用のプロンプト
        self.prioritization_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析の専門家です。
ユーザーと対話しながら、どの末端ノードを優先的に解決すべきかを特定します。

現在のROIツリーには複数の末端ノードがあり、解決策を提案するためにはどのノードを優先するかをユーザーに確認する必要があります。

以下の末端ノードについて、ユーザーがどれを優先したいかを自然な対話で尋ねてください：
{leaf_nodes}

ユーザーが明確な優先順位を示していない場合は、以下の視点から質問を行い、優先度を判断する材料を集めてください：
1. ビジネスインパクト - どのノードが最も大きな効果をもたらすか
2. ROIの大きさ - どのノードが最も高いROIを期待できるか
3. 実現のしやすさ - どのノードが最も実装が容易か
4. 緊急性 - どのノードが最も早急に対応すべきか
"""),
            ("human", "{user_message}")
        ])
    
    def get_response(self, user_message: str, conversation_history: List, mermaid_diagram: str, current_focus: str = "data_collection", callback=None):
        """ユーザーメッセージに対する応答を生成する"""
        # 数値が設定されていないノードを取得
        nodes_without_values = self.extract_nodes_without_values(mermaid_diagram)
        nodes_text = "\n".join([f"- {label}" for _, label in nodes_without_values])
        
        # 次のステップを決定
        next_step = "データ収集を継続してください。各ノードのコストとベネフィットの両方を収集するよう心がけてください。"
        if current_focus == "data_collection" and not nodes_without_values:
            next_step = "すべてのノードにデータが揃いました。次は優先ノードの選択に進むべきです。ROIが最も高いノードを特定し、どのノードを優先的に解決したいか尋ねてください。"
        elif current_focus == "prioritization":
            next_step = "優先ノードの選択を行なってください。ユーザーにとって最も重要な課題は何か、ROIが最も高いノードはどれか、どのノードから解決すべきかを尋ねてください。"
        elif current_focus == "unit_conversion":
            # 変換情報から次のステップを設定
            conversion_info = conversation_history[-1].get("conversion_info", {})
            required_info = ", ".join(conversion_info.get("required_info", []))
            next_step = f"単位変換に必要な情報を収集してください。必要な情報: {required_info}。変換後にはROI計算ができるようにコストとベネフィットを明確にしてください。"
        elif current_focus == "cost_benefit":
            # コスト・ベネフィット情報収集モード
            node_name = st.session_state.get("current_node_name", "選択されたノード")
            next_step = f"「{node_name}」のコストとベネフィットの両方の情報を収集してください。これによりROIの計算が可能になります。"
        
        # プロンプトを選択
        if current_focus == "unit_conversion":
            # 変換情報を取得
            conversion_info = conversation_history[-1].get("conversion_info", {})
            value = conversion_info.get("value", "")
            unit = conversion_info.get("unit", "")
            node_name = conversion_info.get("node_name", "")
            required_info = ", ".join(conversion_info.get("required_info", []))
            
            prompt = self.conversion_prompt.format(
                value=value,
                unit=unit,
                node_name=node_name,
                required_info=required_info,
                user_message=user_message
            )
        elif current_focus == "cost_benefit":
            # コスト・ベネフィット情報収集モード
            node_name = st.session_state.get("current_node_name", "選択されたノード")
            
            prompt = self.cost_benefit_prompt.format(
                node_name=node_name,
                user_message=user_message
            )
        elif current_focus == "prioritization":
            # 末端ノードを取得
            leaf_nodes = self.extract_leaf_nodes(mermaid_diagram)
            leaf_nodes_text = "\n".join([f"- {label}" for _, label in leaf_nodes])
            
            prompt = self.prioritization_prompt.format(
                leaf_nodes=leaf_nodes_text,
                user_message=user_message
            )
        else:
            # デフォルトは通常のチャット
            prompt = self.chat_prompt.format(
                nodes_without_values=nodes_text,
                next_step=next_step,
                user_message=user_message
            )
        
        # LLMで応答を生成 （コールバック付きの場合はストリーミング）
        if callback:
            response = self.llm.with_config({"callbacks": [callback]}).invoke(
                prompt
            )
        else:
            response = self.llm.invoke(
                prompt
            )
        
        return response.content
    
    def analyze_message(self, message: str, conversation_history: List, mermaid_diagram: str) -> Dict:
        """メッセージを分析し、ノード更新情報を抽出する"""
        return self.nlu_agent.analyze_conversation(conversation_history, message, mermaid_diagram)
    
    def analyze_unit_conversion(self, message: str) -> Dict:
        """メッセージを分析し、単位変換情報を抽出する"""
        return self.nlu_agent.analyze_unit_conversion(message)
    
    def check_conversion_needs(self, value: float, unit: str, description: str = "") -> Dict:
        """数値の単位変換が必要かを判断し、必要な情報を特定する"""
        return self.conversion_agent.analyze_conversion_needs(value, unit, description)
    
    def perform_conversion(self, value: float, unit: str, description: str = "", additional_info: str = "") -> Dict:
        """数値の単位変換を実行する（セルフリフレクション付き）"""
        return self.conversion_agent.perform_conversion(value, unit, description, additional_info)
    
    def generate_node_question(self, nodes_without_values) -> str:
        """
        数値が欠けているノードについて質問を生成する
        コストとベネフィットの両方の情報を求める質問を優先
        """
        if not nodes_without_values:
            return "すべてのノードには既に数値が設定されています。ROIツリーについて他に質問はありますか？"
        
        # ランダムに1つのノードを選択
        node_id, node_label = random.choice(nodes_without_values)
        
        # コストとベネフィットの両方を求める質問のバリエーション
        questions = [
            f"「{node_label}」について、実装コストと期待される効果の両方を教えていただけますか？これによりROIを計算できます。",
            f"「{node_label}」に関して、どのくらいのコストがかかり、どのくらいのベネフィットが期待できますか？",
            f"「{node_label}」の投資額と期待リターンについて具体的な数値をお持ちでしょうか？",
            f"「{node_label}」にかかるコストと、それによってもたらされる効果を数値で表すとどうなりますか？"
        ]
        
        return random.choice(questions)
    
    def extract_cost_benefit_from_message(self, message: str) -> Dict[str, Any]:
        """
        メッセージからコストとベネフィット情報を抽出する（新規）
        """
        # 金額のパターン
        cost_patterns = [
            r'コスト[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'費用[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'投資[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?コスト',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?費用',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?投資'
        ]
        
        benefit_patterns = [
            r'効果[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'ベネフィット[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'リターン[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'利益[は：:]*\s*(\d+[,.]?\d*\s*[億万千]?円)',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?効果',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?ベネフィット',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?リターン',
            r'(\d+[,.]?\d*\s*[億万千]?円)[の]?利益'
        ]
        
        # コスト情報を抽出
        cost_value = None
        for pattern in cost_patterns:
            matches = re.search(pattern, message)
            if matches:
                cost_value = matches.group(1)
                break
        
        # ベネフィット情報を抽出
        benefit_value = None
        for pattern in benefit_patterns:
            matches = re.search(pattern, message)
            if matches:
                benefit_value = matches.group(1)
                break
        
        return {
            "found_cost": cost_value is not None,
            "cost_value": cost_value,
            "found_benefit": benefit_value is not None,
            "benefit_value": benefit_value
        }
    
    def extract_nodes_without_values(self, mermaid_code):
        """
        数値情報を持たない末端ノードを抽出する
        """
        leaf_nodes = self.extract_leaf_nodes(mermaid_code)
        nodes_without_values = []
        
        for node_id, node_label in leaf_nodes:
            if not self.has_numerical_value(node_label):
                nodes_without_values.append((node_id, node_label))
        
        return nodes_without_values
    
    def has_numerical_value(self, node_label: str) -> bool:
        """
        ノードラベルに数値情報（金額、パーセンテージなど）が含まれているかをチェック
        """
        # 金額のパターン（例: 1億円, 500万円, 1,000円）
        amount_pattern = r'(\d+[,\.]?\d*\s*[億万千]?円)'
        # パーセンテージのパターン（例: 25%, 3.5%）
        percentage_pattern = r'(\d+[,\.]?\d*\s*%)'
        # 一般的な数値パターン（例: 100件, 50人）
        general_pattern = r'(\d+[,\.]?\d*\s*[件人時分秒])'
        
        return bool(re.search(amount_pattern, node_label)) or \
               bool(re.search(percentage_pattern, node_label)) or \
               bool(re.search(general_pattern, node_label))
    
    def extract_leaf_nodes(self, mermaid_code):
        """
        Mermaidコードから末端ノード（他のノードの親になっていないノード）を抽出する
        """
        return self.nlu_agent.extract_leaf_nodes(mermaid_code)
    
    # 新規: ROIを計算する関数
    def calculate_roi(self, cost_value: str, benefit_value: str) -> float:
        """
        コストとベネフィットからROIを計算する
        
        Args:
            cost_value: コスト値（文字列形式、単位付き）
            benefit_value: ベネフィット値（文字列形式、単位付き）
            
        Returns:
            ROI値（%）、計算できない場合はNone
        """
        try:
            # 単位を統一して数値に変換
            cost = self._convert_to_yen(cost_value)
            benefit = self._convert_to_yen(benefit_value)
            
            if cost <= 0:
                return None
            
            # ROI計算: (ベネフィット - コスト) / コスト × 100%
            roi = (benefit - cost) / cost * 100
            return roi
        except:
            return None
    
    def _convert_to_yen(self, value_str: str) -> float:
        """
        金額文字列を円単位の数値に変換する
        
        Args:
            value_str: 金額文字列（例: 1億円, 500万円, 1,000円）
            
        Returns:
            円単位の数値
        """
        if not value_str:
            return 0
        
        # カンマを削除
        value_str = value_str.replace(',', '')
        
        # 単位に基づいて変換
        if '億円' in value_str:
            # 億円 → 円
            num_str = value_str.replace('億円', '')
            return float(num_str) * 100000000
        elif '万円' in value_str:
            # 万円 → 円
            num_str = value_str.replace('万円', '')
            return float(num_str) * 10000
        elif '千円' in value_str:
            # 千円 → 円
            num_str = value_str.replace('千円', '')
            return float(num_str) * 1000
        elif '円' in value_str:
            # すでに円単位
            num_str = value_str.replace('円', '')
            return float(num_str)
        else:
            # 単位がない場合はそのまま変換
            return float(value_str)