"""
agents/proposal_agent.py
Enhanced Proposal Agent for ROI calculation with priority node support and simplified ROI focus
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import ProposalResult, SolutionRecommendation, NumericalValue, ROICalculation
from domain.roitree import ROINode, get_leaf_nodes, mermaid_to_roi_tree


# 改良されたProposal agent system prompt（単年ROI中心）
PROPOSAL_SYSTEM_PROMPT = """# ROI Proposal Expert

あなたはビジネス提案の専門エージェントです。
ユーザーから与えられた「課題ツリー」(Mermaid記法)をもとに、指定された末端ノードに対して提案を生成してください。

## 提案生成の条件:
- 指定された課題の末端ノードに対し、具体的な打ち手・DXソリューションを提案する
- 提案は現実的かつ具体的な内容とする
- 提案には予想される実装コストを設定する
- 提案には期待される効果（金額または数値）を設定する
- 提案には実装期間の目安を設定する
- 提案には優先度（1＝最高、5＝最低）を設定する
- 「投資コスト」と「期待効果」から、単年ROIを試算する
- ROIは「(期待効果-投資コスト)/投資コスト×100%」の式で計算する

## 出力形式:
- 提案ツリーはMermaid記法で表現し、必ず「flowchart BT」で始めること
- 各ノード定義は独立した行に記述すること
- 各エッジ（接続）も独立した行に記述すること
- 具体的な書式例:
```
flowchart BT
    P_A["提案A (コスト:500万円/効果:1000万円/優先度:1/ROI:100%)"]
    P_ROI["最終ROI: 100%"]
    P_A --> P_ROI
```
- ROIを頂点ノードに配置し、その下に提案ノードを配置する
- 提案ノードには、具体的な名前とコスト情報と効果予測、ROIを含める
- 提案ノードのIDには「P_」というプレフィックスをつける（例: P_A, P_B）

## 提案内容には以下の要素を含めてください:
1. 具体的なソリューション名（実在するDXツール名を含む）
2. 実装に必要なコスト
3. 期待される効果（可能な限り数値化）
4. 実装期間の目安
5. ROI値
6. 優先度とその理由

## ROIの計算方法:
- 各提案ノードでは「(期待効果-投資コスト)/投資コスト×100%」でROIを計算する
- コストと効果は同じ単位（通常は円）で計算すること
- 単年ROIを基本としてください（複数年のROI計算は行わない）
"""


# 単一ノード提案system prompt（単年ROI中心）
SINGLE_NODE_PROPOSAL_PROMPT = """# 特定末端ノードのROI提案エキスパート

あなたはビジネス提案の専門エージェントです。
ユーザーから与えられた「課題ツリー」(Mermaid記法)の中から、指定された特定の末端ノードに対してのみ提案を生成してください。

## 提案生成の条件:
- 指定された課題の末端ノードに対し、具体的なDXソリューションを提案する
- 複数の代替案（最大3つ）を提示し、それぞれに単年ROIを計算する
- 各提案は現実的かつ具体的な内容とし、具体的なDXツール名を含める
- 各提案には予想される実装コストを設定する
- 各提案には期待される効果（金額または数値）を設定する
- 各提案には実装期間の目安を設定する
- 各提案には優先度（1＝最高、5＝最低）を設定する
- 「投資コスト」と「期待効果」から、単年ROIを試算する
- ROIは「(期待効果-投資コスト)/投資コスト×100%」の式で計算する

## 出力形式:
- 提案ツリーはMermaid記法で表現し、必ず「flowchart BT」で始めること
- 各ノード定義は独立した行に記述すること
- 各エッジ（接続）も独立した行に記述すること
- 具体的な書式例:
```
flowchart BT
    P_A["提案A (コスト:500万円/効果:1000万円/優先度:1/ROI:100%)"]
    P_B["提案B (コスト:300万円/効果:800万円/優先度:2/ROI:167%)"]
    P_C["提案C (コスト:100万円/効果:200万円/優先度:3/ROI:100%)"]
    P_ROI["最適ROI: 167% (提案B)"]
    P_A --> P_ROI
    P_B --> P_ROI
    P_C --> P_ROI
```
- 最適なROIを頂点ノードに配置し、その下に提案ノードを配置する
- 各提案ノードには、具体的な名前とコスト情報と効果予測、ROIを含める
- 提案ノードのIDには「P_」というプレフィックスをつける（例: P_A, P_B）

## 提案内容には以下の要素を含めてください:
1. 具体的なソリューション名（実在するDXツール名を含む）
2. 実装に必要なコスト
3. 期待される効果（可能な限り数値化）
4. 実装期間の目安
5. ROI値
6. 優先度とその理由

## ROIの計算方法:
- 各提案ノードでは「(期待効果-投資コスト)/投資コスト×100%」で単年ROIを計算する
- コストと効果は同じ単位（通常は円）で計算すること
- 最も高いROIを持つ提案を「最適ROI」として頂点ノードに表示する
"""


class ProposalAgent:
    """Enhanced agent for generating proposals based on ROI trees with DX tools and simplified ROI"""
    
    def __init__(self, model_name: str = "o1"):
        """
        Initialize the proposal agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            #temperature=0.2,
            streaming=True  # ストリーミングを有効化
        )
        
        # Create prompt templates
        self.proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", PROPOSAL_SYSTEM_PROMPT),
            ("human", """以下のMermaid記法で表された課題ツリーをもとに、提案ツリーを生成してください。
末端ノードと対応する提案は1対1になるようにし、下から上に生えるツリー（flowchart BT）として表現してください。
各提案には具体的なDXツール名を含め、単年ROIを計算して頂点ノードに配置してください。

【課題ツリー】
{{mermaid_diagram}}

【優先したい末端ノード】
{{priority_nodes}}

【提案の方向性（任意）】
{{proposal_guidance}}

提案ツリーのMermaid記法と、単年ROI試算を出力してください。
正しいMermaid構文に従って、各ノード定義とエッジ定義を別々の行に記述してください。

各提案には以下の要素を含めてください:
1. 具体的なDXソリューション名
2. 実装コスト
3. 期待効果
4. 優先度（1〜5）
5. 単年ROI値（%）

また、各提案の詳細について、以下の形式で補足説明を加えてください:
【提案名】: 提案の詳細説明
- 実装期間: xx週間/xx月
- 優先度: x（理由: xxxx）
- 実施内容: xxxxxx
- 期待効果: xxxxx
- ROI: xx%
- 必要な資源: xxxxx
""")
        ])
        
        # 単一ノード用の提案プロンプト
        self.single_node_proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", SINGLE_NODE_PROPOSAL_PROMPT),
            ("human", """以下のMermaid記法で表された課題ツリーの中から、特定の末端ノードに対して単年ROIを最大化する提案を生成してください。

【課題ツリー】
{{mermaid_diagram}}

【対象とする末端ノード】
{{target_node}}

【提案の方向性（任意）】
{{proposal_guidance}}

対象の末端ノードに対して、単年ROIを最大化する複数の提案（最大3つ）を生成し、それぞれのROIを計算してください。
すべての提案をツリー（flowchart BT）として表現し、最も高いROIを持つ提案を頂点ノードに配置してください。
各提案には、実在する具体的なDXツール名を含めてください。

各提案には以下の要素を含めてください:
1. 具体的なDXソリューション名
2. 実装に必要なコスト
3. 期待される効果（数値化）
4. ROI値（%）
5. 優先度（1〜5）

また、各提案の詳細について、以下の形式で補足説明を加えてください:
【提案名】: 提案の詳細説明
- 実装期間: xx週間/xx月
- 優先度: x（理由: xxxx）
- 実施内容: xxxxxx
- 期待効果: xxxxx
- ROI: xx%
- 必要な資源: xxxxx
""")
        ])
        
        # Prompt for prioritized node selection
        self.prioritization_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリーの優先度分析の専門家です。
末端ノードをビジネスインパクトと実装難易度の観点から分析し、優先的に解決すべきノードを提案してください。

分析では以下の点を考慮してください:
1. ビジネスインパクト（コスト削減額や売上増加額など）
2. 実装の難易度（期間、必要リソース、リスクなど）
3. 前提条件や依存関係（他のノードを先に解決する必要があるか）
4. 全体ROIへの貢献度
5. 適切なDXツールの有無

レスポンスはJSON形式で返してください:
```json
{{
  "prioritized_nodes": [
    {{"name": "ノード名1", "reason": "優先する理由", "expected_impact": "期待されるインパクト", "suggested_dx_tools": ["ツール1", "ツール2"]}}
  ],
  "deprioritized_nodes": [
    {{"name": "ノード名3", "reason": "優先度を下げる理由"}}
  ],
  "dependencies": [
    {{"node": "ノード名", "depends_on": "依存するノード名", "reason": "依存理由"}}
  ]
}}
```"""),
            ("human", """以下のROIツリーの末端ノードを分析し、優先的に解決すべきノードを提案してください。

【ROIツリー】
{{mermaid_diagram}}

【末端ノード一覧】
{{leaf_nodes}}

特に優先すべき末端ノードを3つ以内で選出し、その理由と期待されるインパクトを説明してください。
各優先ノードに対して適用可能なDXツールも提案してください。
また、他のノードよりも優先度を下げるべきノードがあれば、その理由も説明してください。
ノード間に依存関係がある場合は、それも特定してください。

JSON形式で回答してください。
""")
        ])
        
        # Prompt for extracting summary information - simplified for single year ROI
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI提案の分析専門家です。提案ツリーとROI試算結果から、主要な情報を抽出してください。

以下の情報を抽出し、JSON形式で返してください:
1. 総投資額（円）
2. 総期待効果額（円）
3. 単年ROI率（%）
4. 実装期間の目安
5. 主要な提案ポイント（最大5つ）
6. 各提案の優先度ランキング
7. 各提案に含まれるDXツール名

レスポンスは以下のJSON形式で返してください:
```json
{{
  "total_investment": 数値,
  "total_benefit": 数値,
  "roi_percentage": 数値,
  "implementation_timeframe": "期間の説明",
  "key_recommendations": ["提案1", "提案2", "提案3"],
  "priority_ranking": [
    {{"proposal": "提案名", "priority": 優先度, "cost": コスト, "benefit": 効果, "roi": ROI, "dx_tools": ["ツール名1", "ツール名2"]}}
  ]
}}
```"""),
            ("human", """以下の提案ツリーとROI試算から、主要情報を抽出してください：

【提案ツリーと試算】
{{proposal_text}}
""")
        ])
        
        # 単一ノード提案の要約プロンプト
        self.single_node_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI提案の分析専門家です。特定の末端ノードに対する提案から、主要な情報を抽出してください。

以下の情報を抽出し、JSON形式で返してください:
1. 対象ノード名
2. 最適提案の名前
3. 最適提案の投資額（円）
4. 最適提案の期待効果額（円）
5. 最適提案の単年ROI率（%）
6. 実装期間の目安
7. 最適提案に含まれるDXツール名
8. 提案の詳細ポイント（最大3つ）
9. 代替提案のリスト（各提案の名前、コスト、効果、ROI、DXツールを含む）

レスポンスは以下のJSON形式で返してください:
```json
{{
  "target_node": "対象ノード名",
  "best_proposal": "最適提案名",
  "investment": 数値,
  "benefit": 数値,
  "roi_percentage": 数値,
  "implementation_timeframe": "期間の説明",
  "dx_tools": ["ツール名1", "ツール名2"],
  "key_points": ["要点1", "要点2", "要点3"],
  "alternative_proposals": [
    {{"name": "代替提案名", "cost": コスト, "benefit": 効果, "roi": ROI, "dx_tools": ["ツール名1", "ツール名2"]}}
  ]
}}
```"""),
            ("human", """以下の特定ノードに対する提案から、主要情報を抽出してください：

【対象ノード】
{{target_node}}

【提案内容】
{{proposal_text}}
""")
        ])
    
    def generate_proposal(self, challenge_tree_mermaid: str, priority_nodes: str = "", proposal_guidance: str = "") -> Tuple[str, ProposalResult]:
        """
        Generate a proposal based on a challenge tree
        
        Args:
            challenge_tree_mermaid: Mermaid diagram of the challenge tree
            priority_nodes: Optional prioritized nodes to focus on
            proposal_guidance: Optional guidance for proposal generation
            
        Returns:
            Tuple of (proposal text, proposal summary)
        """
        # Generate proposal using LLM
        messages = self.proposal_prompt.format_messages(
            mermaid_diagram=challenge_tree_mermaid,
            priority_nodes=priority_nodes,
            proposal_guidance=proposal_guidance
        )
        
        response = self.llm.invoke(messages)
        proposal_text = response.content
        
        # Extract summary information
        summary = self.extract_summary(proposal_text)
        
        return proposal_text, summary
    
    def generate_single_node_proposal(self, challenge_tree_mermaid: str, target_node: str, proposal_guidance: str = "", callback=None) -> Tuple[str, ProposalResult]:
        """
        Generate a proposal for a specific leaf node
        
        Args:
            challenge_tree_mermaid: Mermaid diagram of the challenge tree
            target_node: The specific leaf node to focus on
            proposal_guidance: Optional guidance for proposal generation
            callback: Optional streaming callback
            
        Returns:
            Tuple of (proposal text, proposal summary)
        """
        # Generate proposal using LLM
        messages = self.single_node_proposal_prompt.format_messages(
            mermaid_diagram=challenge_tree_mermaid,
            target_node=target_node,
            proposal_guidance=proposal_guidance
        )
        
        if callback:
            # ストリーミングコールバックを使用
            response = self.llm.with_config({"callbacks": [callback]}).invoke(messages)
        else:
            # 通常の呼び出し
            response = self.llm.invoke(messages)
            
        proposal_text = response.content
        
        # Extract summary information for single node proposal
        summary = self.extract_single_node_summary(proposal_text, target_node)
        
        return proposal_text, summary
    
    def generate_proposal_streaming(self, challenge_tree_mermaid: str, proposal_guidance: str = "", priority_nodes: str = "", callback=None) -> Tuple[str, ProposalResult]:
        """
        Generate a proposal with streaming output
        
        Args:
            challenge_tree_mermaid: Mermaid diagram of the challenge tree
            proposal_guidance: Optional guidance for proposal generation
            priority_nodes: Optional prioritized nodes to focus on
            callback: Optional streaming callback
            
        Returns:
            Tuple of (proposal text, proposal summary)
        """
        # Generate proposal using LLM with streaming
        messages = self.proposal_prompt.format_messages(
            mermaid_diagram=challenge_tree_mermaid,
            priority_nodes=priority_nodes,
            proposal_guidance=proposal_guidance
        )
        
        if callback:
            # ストリーミングコールバックを使用
            response = self.llm.with_config({"callbacks": [callback]}).invoke(messages)
        else:
            # 通常の呼び出し
            response = self.llm.invoke(messages)
            
        proposal_text = response.content
        
        # Extract summary information
        summary = self.extract_summary(proposal_text)
        
        return proposal_text, summary
    
    def prioritize_nodes(self, challenge_tree_mermaid: str, leaf_nodes_text: str) -> Dict[str, Any]:
        """
        Prioritize leaf nodes for solution focus
        
        Args:
            challenge_tree_mermaid: Mermaid diagram of the challenge tree
            leaf_nodes_text: Text description of leaf nodes
            
        Returns:
            Prioritization recommendations
        """
        # Prioritize nodes using LLM
        messages = self.prioritization_prompt.format_messages(
            mermaid_diagram=challenge_tree_mermaid,
            leaf_nodes=leaf_nodes_text
        )
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        prioritization_json = self._extract_json(response.content)
        
        # DXツールの推奨がない場合はデフォルトを追加
        if "prioritized_nodes" in prioritization_json:
            for node in prioritization_json["prioritized_nodes"]:
                if "suggested_dx_tools" not in node:
                    node["suggested_dx_tools"] = ["RPAツール", "クラウドERP", "AIチャットボット"]
        
        return prioritization_json
    
    def extract_summary(self, proposal_text: str) -> ProposalResult:
        """
        Extract summary information from proposal text
        
        Args:
            proposal_text: Text of the proposal
            
        Returns:
            Summary of the proposal
        """
        # Extract summary using LLM
        messages = self.summary_prompt.format_messages(
            proposal_text=proposal_text
        )
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        summary_json = self._extract_json(response.content)
        
        try:
            # 単年ROI用の情報を追加
            if "roi_percentage" in summary_json:
                summary_json["single_year_roi"] = {
                    "roi": summary_json["roi_percentage"],
                    "investment": summary_json["total_investment"],
                    "benefit": summary_json["total_benefit"]
                }
                
            return ProposalResult(**summary_json)
        except Exception as e:
            print(f"Error parsing proposal summary: {str(e)}")
            # Return default summary if parsing fails
            return ProposalResult(
                total_investment=0,
                total_benefit=0,
                roi_percentage=0,
                implementation_timeframe="不明",
                key_recommendations=["情報抽出に失敗しました"],
                single_year_roi={"roi": 0, "investment": 0, "benefit": 0}
            )
    
    def extract_single_node_summary(self, proposal_text: str, target_node: str) -> ProposalResult:
        """
        Extract summary information from single node proposal text
        
        Args:
            proposal_text: Text of the proposal
            target_node: The target node for the proposal
            
        Returns:
            Summary of the proposal
        """
        # Extract summary using LLM
        messages = self.single_node_summary_prompt.format_messages(
            target_node=target_node,
            proposal_text=proposal_text
        )
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        summary_json = self._extract_json(response.content)
        
        try:
            # 単一ノード提案の要約結果をProposalResultに変換
            # (ここでは構造が異なるので変換処理が必要)
            best_proposal = summary_json.get("best_proposal", "")
            
            # ProposalResult形式に変換
            result = ProposalResult(
                total_investment=summary_json.get("investment", 0),
                total_benefit=summary_json.get("benefit", 0),
                roi_percentage=summary_json.get("roi_percentage", 0),
                implementation_timeframe=summary_json.get("implementation_timeframe", "不明"),
                key_recommendations=[best_proposal] + summary_json.get("key_points", []),
                priority_ranking=[
                    {"proposal": prop.get("name", ""), 
                     "priority": i+1, 
                     "cost": prop.get("cost", 0), 
                     "benefit": prop.get("benefit", 0),
                     "roi": prop.get("roi", 0),
                     "dx_tools": prop.get("dx_tools", [])
                    } 
                    for i, prop in enumerate(summary_json.get("alternative_proposals", []))
                ],
                single_year_roi={
                    "roi": summary_json.get("roi_percentage", 0),
                    "investment": summary_json.get("investment", 0),
                    "benefit": summary_json.get("benefit", 0)
                }
            )
            
            # 元のJSONもそのまま保持
            result.original_summary = summary_json
            result.target_node = summary_json.get("target_node", target_node)
            
            return result
        except Exception as e:
            print(f"Error parsing single node proposal summary: {str(e)}")
            # Return default summary if parsing fails
            return ProposalResult(
                total_investment=0,
                total_benefit=0,
                roi_percentage=0,
                implementation_timeframe="不明",
                key_recommendations=["情報抽出に失敗しました"],
                single_year_roi={"roi": 0, "investment": 0, "benefit": 0}
            )
    
    def calculate_single_year_roi(self, investment: float, benefit: float) -> Dict[str, Any]:
        """
        Calculate single year ROI from investment and benefit
        
        Args:
            investment: Investment amount
            benefit: Benefit amount
            
        Returns:
            Dictionary with ROI calculation results
        """
        if investment <= 0:
            return {
                "roi": 0, 
                "investment": investment, 
                "benefit": benefit,
                "is_valid": False,
                "message": "投資額がゼロ以下のためROIを計算できません"
            }
        
        roi = (benefit - investment) / investment * 100
        
        return {
            "roi": roi,
            "investment": investment,
            "benefit": benefit,
            "is_valid": True,
            "payback_period": investment / benefit if benefit > 0 else float('inf')
        }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text"""
        try:
            # Try to find JSON in a code block
            import re
            json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no code block, try to find JSON-like structure
            json_match = re.search(r'({.*})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Default empty response
            return {}
        except Exception as e:
            print(f"Error extracting JSON: {str(e)}")
            return {}