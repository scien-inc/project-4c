"""
agents/node_prioritization_agent.py
Node prioritization agent for ROI tree analysis
"""
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import PrioritizationResult, MappingResult


class NodePrioritizationAgent:
    """Agent for prioritizing and mapping ROI tree nodes"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the node prioritization agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            #temperature=0.2,
            streaming=True
        )
        
        # Prompt for prioritizing nodes
        self.prioritization_prompt = ChatPromptTemplate.from_messages([
            ("system", """# ノード優先順位付け専門家

あなたはROIツリーの末端ノードを優先順位付けする専門家です。
ROIツリーの末端ノードを分析し、投資対効果や実現可能性などの観点から
どのノードから対応すべきかを提案します。

## 優先順位付けの条件:
- ROI（投資対効果）の大きさ
- 実装の難易度
- 実装までの期間
- ユーザーの優先事項
- ビジネスインパクト

## 出力形式:
優先すべき末端ノードをJSONで順序付けて返してください。
"""),
            ("human", """以下のROIツリーの末端ノードについて、優先順位付けを行ってください。

末端ノード:
{leaf_nodes}

ユーザーの優先事項:
{user_priorities}

以下のJSON形式で出力してください:
```json
{{
  "prioritized_nodes": [
    {{
      "node_id": "ノードID",
      "name": "ノード名",
      "priority_score": 0-100,
      "rationale": "優先理由の説明"
    }}
  ],
  "overall_strategy": "全体的な優先戦略の説明"
}}
```
""")
        ])
        
        # Prompt for 2D mapping
        self.mapping_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 2次元マッピング専門家

あなたはROIツリーのノードを2次元マップにマッピングする専門家です。
ROIツリーのノードを分析し、指定された2軸上に適切に配置します。

## マッピングの条件:
各ノードを0～10の値でX軸とY軸上にマッピングし、配置の理由を説明してください。
"""),
            ("human", """以下のROIツリーのノードを指定された2軸上にマッピングしてください。

ノード:
{nodes}

X軸（0-10）: {x_axis}
Y軸（0-10）: {y_axis}

以下のJSON形式で出力してください:
```json
{{
  "mapping": [
    {{
      "node_id": "ノードID",
      "name": "ノード名",
      "x_value": X軸の値（0-10）,
      "y_value": Y軸の値（0-10）,
      "explanation": "この位置に配置した理由"
    }}
  ],
  "quadrant_analysis": [
    {{
      "quadrant": "右上/左上/右下/左下",
      "description": "この象限の特徴",
      "recommendation": "この象限のノードへの対応方針"
    }}
  ]
}}
```
""")
        ])
    
    def prioritize_nodes(self, leaf_nodes: List[Dict], user_priorities: str = "") -> PrioritizationResult:
        """
        Prioritize leaf nodes based on ROI and user priorities
        
        Args:
            leaf_nodes: List of leaf node dictionaries
            user_priorities: Optional user priorities as text
            
        Returns:
            PrioritizationResult object with prioritized nodes and overall strategy
        """
        # Format node list for prompt
        nodes_text = "\n".join([
            f"- ID: {node['node_id']}, 名前: {node['name']}, 値: {node.get('value', '不明')}"
            for node in leaf_nodes
        ])
        
        # Get prioritization from LLM
        response = self.llm.invoke(
            self.prioritization_prompt.format(
                leaf_nodes=nodes_text,
                user_priorities=user_priorities
            )
        )
        
        # Extract JSON from response
        result_json = self._extract_json(response.content)
        
        try:
            return PrioritizationResult(**result_json)
        except Exception as e:
            print(f"Error creating PrioritizationResult: {str(e)}")
            # Return default prioritization result if parsing fails
            return PrioritizationResult(
                prioritized_nodes=[],
                overall_strategy="優先順位付けの処理中にエラーが発生しました"
            )
    
    def create_2d_mapping(self, nodes: List[Dict], x_axis: str, y_axis: str) -> MappingResult:
        """
        Map nodes to a 2D coordinate system based on specified axes
        
        Args:
            nodes: List of node dictionaries
            x_axis: Description of X axis
            y_axis: Description of Y axis
            
        Returns:
            MappingResult object with mapped nodes and quadrant analysis
        """
        # Format node list for prompt
        nodes_text = "\n".join([
            f"- ID: {node['node_id']}, 名前: {node['name']}, 値: {node.get('value', '不明')}"
            for node in nodes
        ])
        
        # Get mapping from LLM
        response = self.llm.invoke(
            self.mapping_prompt.format(
                nodes=nodes_text,
                x_axis=x_axis,
                y_axis=y_axis
            )
        )
        
        # Extract JSON from response
        result_json = self._extract_json(response.content)
        
        try:
            return MappingResult(**result_json)
        except Exception as e:
            print(f"Error creating MappingResult: {str(e)}")
            # Return default mapping result if parsing fails
            return MappingResult(
                mapping=[],
                quadrant_analysis=[]
            )
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from text"""
        try:
            # Try to find JSON in a code block
            json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # If no code block, try to find JSON-like structure
            json_match = re.search(r'({.*})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            return {}
        except Exception as e:
            print(f"JSON extraction error: {str(e)}")
            return {}