"""
agents/deepdive_agent.py
Simplified Challenge Agent for ROI tree exploration
"""
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import LeafNodeAnalysis, NumericalValue
from domain.roitree import ROINode, create_default_roi_tree, get_leaf_nodes, mermaid_to_roi_tree


# Challenge agent system prompt
CHALLENGE_SYSTEM_PROMPT = """# ROI Tree Analysis Expert

あなたはビジネス課題をROIの観点から構造化する専門エージェントです。
ユーザーから与えられる事業・組織の課題文を解析し、ROIを「コスト削減」と「売上拡大」の2つに分岐してツリー構造を生成してください。

## ROIツリー作成の条件:
- 課題を「コスト削減」と「売上拡大」の2つの主要カテゴリに分解する
- 各カテゴリの下に、より具体的なサブカテゴリを特定する
- サブカテゴリの下に、具体的な項目を特定する
- 文章中に明示的に数値目標が記載されている場合のみ、その値をノードに含める
- 数値目標が不明確な場合は、数値を含めずにノードを作成する
- 「課題要因」という抽象的なノードは含めず、具体的なタスクや目的をノードとする
- 末端ノードについては、投資対効果(ROI)計算に必要な数値情報の種類を識別する（金額、時間、件数、人数など）

## 出力:
- ROIツリーはMermaid記法で表現し、graph TDで開始すること
- 数値目標が明確な場合のみ、ノードにその値を含める（例: `node1["製造効率向上 (1億円削減)"]`）
- 数値目標が不明確な場合はノードに値を含めない（例: `node1["製造効率向上"]`）
- 勝手に数値を追加しないこと。入力文から明示的に読み取れる数値のみを使用すること。
"""


class ChallengeAgent:
    """Simplified agent for exploring business challenges and creating ROI trees"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the challenge agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True  # ストリーミングを有効化
        )
        
        self.reflection_llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            streaming=False  # 分析には高速レスポンスが必要
        )
        
        # Create prompt templates
        self.challenge_prompt = ChatPromptTemplate.from_messages([
            ("system", CHALLENGE_SYSTEM_PROMPT),
            ("human", """以下のビジネス課題を解析し、ROIツリーを作成してください。
明示的に数値目標が示されている場合のみ、その値をノードに含めてください。数値が不明確な場合は、値を含めずにノードを作成してください。

ビジネス課題:
{challenge_text}

Mermaid記法でROIツリーを表現してください。数値目標がわかる場合のみ、それをノードに含めてください。
""")
        ])
        
        # Reflection prompt for analyzing leaf nodes
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリーの分析専門家です。末端ノード（子を持たないノード）の分析をして、
数値目標が適切に設定されているかを判断してください。

分析では以下を判定してください:
1. すべての末端ノードに数値目標が設定されているか
2. 数値目標が設定されていないノードはどれか
3. 各末端ノードにどのような種類の数値情報が必要か（金額、時間、件数、人数など）
4. ツリー全体の完成度（パーセンテージ）

レスポンスは以下のJSON形式で返してください:
```json
{{
  "has_numerical_data": true/false,
  "missing_nodes": ["ノード名1", "ノード名2"],
  "incomplete_nodes": [
    {{"name": "ノード名", "issue": "問題の説明", "suggestion": "改善提案", "value_type": "数値の種類（金額/時間/件数/人数など）", "value_unit": "単位（円/時間/件/人など）"}}
  ],
  "completion_percentage": 0-100
}}
```"""),
            ("human", "以下のROIツリーの末端ノードを分析してください：\n\n{mermaid_diagram}\n\n末端ノードのリスト：\n{leaf_nodes}")
        ])
        
        # Solution suggestion prompt
        self.solution_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはビジネス課題に対して最適なソリューションを提案する専門家です。
ROIツリーの末端ノードを分析し、各ノードに適した解決策の種類と必要な情報を特定してください。

各末端ノードに対して、以下の情報を提供してください:
1. 想定される解決策のカテゴリ（例: システム導入、プロセス改善、人材育成など）
2. ROI計算に必要な追加情報（例: 現在の工数、対象人数、発生頻度など）
3. 優先度の判断基準（実装難易度、期待効果、緊急性など）

レスポンスは以下のJSON形式で返してください:
```json
{{
  "solution_suggestions": [
    {{
      "node_name": "ノード名",
      "solution_categories": ["カテゴリ1", "カテゴリ2"],
      "required_information": ["必要情報1", "必要情報2"],
      "priority_criteria": ["基準1", "基準2"]
    }}
  ]
}}
```"""),
            ("human", "以下のROIツリーの末端ノードに対する解決策の種類と必要な情報を特定してください：\n\n{mermaid_diagram}\n\n末端ノードのリスト：\n{leaf_nodes}")
        ])
    
    def create_roi_tree(self, challenge_text: str) -> Tuple[ROINode, str]:
        """
        Create an ROI tree from a challenge description
        
        Args:
            challenge_text: Text describing the business challenge
            
        Returns:
            Tuple of (ROI tree root node, Mermaid diagram)
        """
        # Generate ROI tree using LLM
        messages = self.challenge_prompt.format_messages(
            challenge_text=challenge_text
        )
        
        response = self.llm.invoke(messages)
        
        # Extract Mermaid diagram from response
        mermaid_diagram = self._extract_mermaid(response.content)
        
        # Convert Mermaid diagram to ROI tree
        root_node = mermaid_to_roi_tree(mermaid_diagram)
        
        # If parsing failed, create a default tree
        if not root_node:
            root_node = create_default_roi_tree()
            mermaid_diagram = root_node.get_full_mermaid()
        
        return root_node, mermaid_diagram
    
    def analyze_leaf_nodes(self, root_node: ROINode, mermaid_diagram: str) -> LeafNodeAnalysis:
        """
        Analyze leaf nodes to check if they have appropriate numerical targets
        
        Args:
            root_node: ROI tree root node
            mermaid_diagram: Mermaid diagram of the ROI tree
            
        Returns:
            Analysis of leaf nodes
        """
        # Get all leaf nodes
        leaf_nodes = get_leaf_nodes(root_node)
        
        # Format leaf nodes for the prompt
        leaf_nodes_text = "\n".join([
            f"- {node.name}" + (f" (値: {node.value})" if node.value is not None else " (値なし)")
            for node in leaf_nodes
        ])
        
        # Analyze leaf nodes using LLM
        messages = self.reflection_prompt.format_messages(
            mermaid_diagram=mermaid_diagram,
            leaf_nodes=leaf_nodes_text
        )
        
        response = self.reflection_llm.invoke(messages)
        
        # Extract JSON from response
        analysis_json = self._extract_json(response.content)
        
        try:
            return LeafNodeAnalysis(**analysis_json)
        except Exception as e:
            print(f"Error parsing leaf node analysis: {str(e)}")
            # Return default analysis if parsing fails
            return LeafNodeAnalysis(
                has_numerical_data=any(node.value is not None for node in leaf_nodes),
                missing_nodes=[node.name for node in leaf_nodes if node.value is None],
                incomplete_nodes=[],
                completion_percentage=50.0
            )
    
    def suggest_solutions(self, root_node: ROINode, mermaid_diagram: str) -> Dict[str, Any]:
        """
        Suggest solutions for leaf nodes
        
        Args:
            root_node: ROI tree root node
            mermaid_diagram: Mermaid diagram of the ROI tree
            
        Returns:
            Solution suggestions for leaf nodes
        """
        # Get all leaf nodes
        leaf_nodes = get_leaf_nodes(root_node)
        
        # Format leaf nodes for the prompt
        leaf_nodes_text = "\n".join([
            f"- {node.name}" + (f" (値: {node.value})" if node.value is not None else " (値なし)")
            for node in leaf_nodes
        ])
        
        # Suggest solutions using LLM
        messages = self.solution_prompt.format_messages(
            mermaid_diagram=mermaid_diagram,
            leaf_nodes=leaf_nodes_text
        )
        
        response = self.reflection_llm.invoke(messages)
        
        # Extract JSON from response
        solution_json = self._extract_json(response.content)
        
        return solution_json
    
    def _extract_mermaid(self, text: str) -> str:
        """Extract Mermaid diagram from text"""
        lines = text.split('\n')
        mermaid_lines = []
        in_mermaid = False
        
        for line in lines:
            if line.strip() == '```mermaid' or line.strip() == '```':
                in_mermaid = not in_mermaid
                continue
                
            if in_mermaid:
                mermaid_lines.append(line)
        
        # If no code block was found, look for graph TD
        if not mermaid_lines:
            started = False
            for line in lines:
                if line.strip().startswith('graph TD'):
                    started = True
                    mermaid_lines.append(line.strip())
                elif started:
                    mermaid_lines.append(line.strip())
        
        return '\n'.join(mermaid_lines)
    
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