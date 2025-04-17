"""
agents/solution_agent.py
Solution recommendation agent for ROI calculations
"""
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import SolutionRecommendation


class SolutionAgent:
    """
    Agent for recommending solutions based on ROI leaf nodes
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the solution agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            #temperature=0.2,
            streaming=True  # ストリーミングを有効化
        )
        
        # Create prompt templates
        self.solution_prompt = ChatPromptTemplate.from_messages([
            ("system", """# ソリューション推薦エキスパート

あなたはビジネス課題に最適なソリューションを推薦する専門家です。
ROIツリーの末端ノードに対して、具体的で実用的なソリューションを提案してください。

## 推薦プロセス:
1. 末端ノードの課題を詳細に理解する
2. 課題に適した具体的なソリューションを検討する
3. 各ソリューションの実装コスト、期待効果、実装期間を見積もる
4. 優先度を付け、投資対効果（ROI）を計算する

## 推薦条件:
- ソリューションは実用的かつ具体的であること
- DXソリューションを中心に、テクノロジーを活用した解決策を優先
- 導入コストと期待効果のバランスを考慮すること
- 実装のしやすさや組織への適合性を考慮すること
- 類似事例やベストプラクティスを参照すること

## 出力形式:
以下のJSON形式で回答してください:
```json
{{
  "solution_recommendations": [
    {{
      "leaf_node_id": "ノードID",
      "leaf_node_name": "ノード名",
      "solution_name": "ソリューション名",
      "solution_description": "ソリューションの詳細説明",
      "estimated_cost": {{"value": 数値, "unit": "単位", "type": "費用の種類"}},
      "estimated_benefit": {{"value": 数値, "unit": "単位", "type": "効果の種類"}},
      "implementation_timeframe": "実装期間の目安",
      "priority": 優先度
    }}
  ]
}}
```"""),
            ("human", """以下のROIツリーの末端ノードに対して、具体的なソリューションを推薦してください。

【ROIツリーの末端ノード】
{leaf_nodes}

【優先ノード】
{priority_nodes}

【提案の方向性】
{proposal_guidance}

各末端ノードに対して1つのソリューションを推薦し、その実装コスト、期待効果、実装期間、優先度を見積もってください。
可能な限り現実的かつ具体的なソリューションを提案してください。

JSON形式で回答してください。
""")
        ])
        
        # Prompt for detailed solution description
        self.detail_prompt = ChatPromptTemplate.from_messages([
            ("system", """# ソリューション詳細解説エキスパート

あなたはDXソリューションの専門家です。
特定のビジネス課題に対して推薦されたソリューションの詳細を説明してください。

## 説明内容:
1. ソリューションの具体的な実装方法
2. 必要なテクノロジーや製品・サービス
3. 実装プロセスのステップ
4. 導入による具体的なメリット
5. 考慮すべきリスクや課題
6. 同様のソリューションの導入事例（もしあれば）

説明は具体的かつ実用的で、実際のビジネス状況に適用できるものにしてください。
"""),
            ("human", """以下のソリューション推薦について、より詳細な説明を提供してください：

【ビジネス課題】
{challenge_description}

【末端ノード】
{leaf_node}

【推薦ソリューション】
{solution_name}

【基本情報】
- 実装コスト: {estimated_cost}
- 期待効果: {estimated_benefit}
- 実装期間: {implementation_timeframe}
- 優先度: {priority}

このソリューションの詳細な実装方法、必要なテクノロジー、導入プロセス、期待されるメリット、考慮すべきリスクについて説明してください。
可能であれば、類似の導入事例も紹介してください。
""")
        ])
    
    def recommend_solutions(self, leaf_nodes: str, priority_nodes: str = "", proposal_guidance: str = "") -> Dict[str, Any]:
        """
        Recommend solutions for leaf nodes
        
        Args:
            leaf_nodes: Text description of leaf nodes
            priority_nodes: Optional prioritized nodes
            proposal_guidance: Optional guidance for proposal generation
            
        Returns:
            Solution recommendations
        """
        # Generate solution recommendations using LLM
        messages = self.solution_prompt.format_messages(
            leaf_nodes=leaf_nodes,
            priority_nodes=priority_nodes,
            proposal_guidance=proposal_guidance
        )
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        recommendations_json = self._extract_json(response.content)
        
        return recommendations_json
    
    def get_solution_details(self, challenge_description: str, leaf_node: str, solution_name: str, 
                           estimated_cost: str, estimated_benefit: str, 
                           implementation_timeframe: str, priority: int) -> str:
        """
        Get detailed description of a solution
        
        Args:
            challenge_description: Description of the business challenge
            leaf_node: Leaf node for which the solution is recommended
            solution_name: Name of the recommended solution
            estimated_cost: Estimated cost of the solution
            estimated_benefit: Estimated benefit of the solution
            implementation_timeframe: Timeframe for implementing the solution
            priority: Priority of the solution
            
        Returns:
            Detailed description of the solution
        """
        # Generate solution details using LLM
        messages = self.detail_prompt.format_messages(
            challenge_description=challenge_description,
            leaf_node=leaf_node,
            solution_name=solution_name,
            estimated_cost=estimated_cost,
            estimated_benefit=estimated_benefit,
            implementation_timeframe=implementation_timeframe,
            priority=priority
        )
        
        response = self.llm.invoke(messages)
        
        return response.content
    
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