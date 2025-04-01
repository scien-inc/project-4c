"""
agents/proposal_agent.py
Simplified Proposal Agent for ROI calculation and solution recommendation
"""
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import ProposalResult
from domain.roitree import ROINode, get_leaf_nodes, mermaid_to_roi_tree


# Proposal agent system prompt
PROPOSAL_SYSTEM_PROMPT = """# ROI Proposal Expert

あなたはビジネス提案の専門エージェントです。
ユーザーから与えられた「課題ツリー」(Mermaid記法)をもとに、それぞれの末端ノードに対して1対1の提案を生成してください。

## 提案生成の条件:
- 課題の末端ノードそれぞれに対し、具体的な打ち手・ソリューションを1つずつ提案する
- 提案の粒度は1ノード＝1ソリューションで、現実的かつ具体的な内容とする
- 各提案には予想される実装コストを設定する
- 最後に「投資コスト」と「期待効果」を合計し、最終的なROIを試算する
- ROIは「(期待効果-投資コスト)/投資コスト×100%」の式で計算する

## 出力形式:
- 提案ツリーはMermaid記法で表現し、必ず「flowchart BT」で始めること
- 各ノード定義は独立した行に記述すること
- 各エッジ（接続）も独立した行に記述すること
- 具体的な書式例:
```
flowchart BT
    P_A["提案A (500万円)"]
    P_B["提案B (300万円)"]
    P_ROI["ROI: 120%"]
    P_A --> P_ROI
    P_B --> P_ROI
```
- ROIを頂点ノードに配置し、その下に提案ノードを配置する
- 各提案ノードには、具体的な名前とコスト情報を含める
- 末端ノードの提案は課題の末端ノードと1対1対応になるようにする
"""


class ProposalAgent:
    """Simplified agent for generating proposals based on ROI trees"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the proposal agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True  # ストリーミングを有効化
        )
        
        # Create prompt templates
        self.proposal_prompt = ChatPromptTemplate.from_messages([
            ("system", PROPOSAL_SYSTEM_PROMPT),
            ("human", """以下のMermaid記法で表された課題ツリーをもとに、提案ツリーを生成してください。
末端ノードと対応する提案は1対1になるようにし、下から上に生えるツリー（flowchart BT）として表現してください。
最終的なROIを計算して頂点ノードに配置してください。

【課題ツリー】
{mermaid_diagram}

【提案の方向性（任意）】
{proposal_guidance}

提案ツリーのMermaid記法と、最終的なROI試算を出力してください。
正しいMermaid構文に従って、各ノード定義とエッジ定義を別々の行に記述してください。
""")
        ])
        
        # Prompt for extracting summary information
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI提案の分析専門家です。提案ツリーとROI試算結果から、主要な情報を抽出してください。

以下の情報を抽出し、JSON形式で返してください:
1. 総投資額（円）
2. 総期待効果額（円）
3. ROI率（%）
4. 実装期間の目安
5. 主要な提案ポイント（最大5つ）

レスポンスは以下のJSON形式で返してください:
```json
{{
  "total_investment": 数値,
  "total_benefit": 数値,
  "roi_percentage": 数値,
  "implementation_timeframe": "期間の説明",
  "key_recommendations": ["提案1", "提案2", "提案3"]
}}
```"""),
            ("human", """以下の提案ツリーとROI試算から、主要情報を抽出してください：

【提案ツリーと試算】
{proposal_text}
""")
        ])
    
    def generate_proposal(self, challenge_tree_mermaid: str, proposal_guidance: str = "") -> Tuple[str, ProposalResult]:
        """
        Generate a proposal based on a challenge tree
        
        Args:
            challenge_tree_mermaid: Mermaid diagram of the challenge tree
            proposal_guidance: Optional guidance for proposal generation
            
        Returns:
            Tuple of (proposal text, proposal summary)
        """
        # Generate proposal using LLM
        messages = self.proposal_prompt.format_messages(
            mermaid_diagram=challenge_tree_mermaid,
            proposal_guidance=proposal_guidance
        )
        
        response = self.llm.invoke(messages)
        proposal_text = response.content
        
        # Extract summary information
        summary = self.extract_summary(proposal_text)
        
        return proposal_text, summary
    
    def generate_proposal_streaming(self, challenge_tree_mermaid: str, proposal_guidance: str = "", callback=None) -> Tuple[str, ProposalResult]:
        """
        Generate a proposal with streaming output
        
        Args:
            challenge_tree_mermaid: Mermaid diagram of the challenge tree
            proposal_guidance: Optional guidance for proposal generation
            callback: Optional streaming callback
            
        Returns:
            Tuple of (proposal text, proposal summary)
        """
        # Generate proposal using LLM with streaming
        messages = self.proposal_prompt.format_messages(
            mermaid_diagram=challenge_tree_mermaid,
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
            return ProposalResult(**summary_json)
        except Exception as e:
            print(f"Error parsing proposal summary: {str(e)}")
            # Return default summary if parsing fails
            return ProposalResult(
                total_investment=0,
                total_benefit=0,
                roi_percentage=0,
                implementation_timeframe="不明",
                key_recommendations=["情報抽出に失敗しました"]
            )
    
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