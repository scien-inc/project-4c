"""
agents/unit_conversion_agent.py
Unit conversion agent for ROI tree analysis
"""
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import ConversionResult, NodeConversion


class UnitConversionAgent:
    """Agent for converting different units in ROI tree nodes to a common unit (Japanese Yen)"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the unit conversion agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            streaming=True
        )
        
        # Prompt for analyzing units and identifying needed conversions
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 単位変換専門家

あなたはROI分析のための単位変換専門エージェントです。
ROIツリーの各ノードの値や単位を確認し、必要に応じて追加情報をユーザーに質問し、
すべての値を共通の単位（通常は日本円）に変換します。

## 対応すべき単位:
- 時間（時間、日、週、月、年）→ 円換算
- 人数（人、チーム）→ 円換算
- パーセンテージ（％）→ 基準値に応じた円換算
- 金額（円、万円、億円）→ 円に標準化

## 出力形式:
変換が必要な項目を特定し、必要な追加情報を具体的に列挙し、JSONで返してください。
"""),
            ("human", """以下のROIツリーの末端ノードについて、すべての値を共通の単位（日本円）に変換するための分析を行ってください。

ノードリスト:
{node_list}

以下のJSON形式で出力してください:
```json
{{
  "nodes_needing_conversion": [
    {{
      "node_id": "ノードID",
      "current_value": "現在の値と単位",
      "required_info": ["必要な追加情報1", "必要な追加情報2"]
    }}
  ],
  "conversion_explanation": "変換全体の説明"
}}
```
""")
        ])
        
        # Prompt for generating questions to gather additional information
        self.question_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 単位変換質問生成

単位変換に必要な追加情報を収集するための質問を生成してください。
質問は簡潔で明確にし、一度に1つの情報だけを尋ねるようにしてください。
"""),
            ("human", """変換が必要なノードとこれまでの会話履歴をもとに、次に尋ねるべき質問を生成してください。

ノード情報: {node_info}
必要な情報: {required_info}
これまでの会話:
{conversation_history}

次の質問を1つだけ生成してください。
""")
        ])
        
        # Prompt for final conversion based on collected information
        self.conversion_prompt = ChatPromptTemplate.from_messages([
            ("system", """# 単位変換実行

収集した情報をもとに、すべてのノードの値を日本円に変換してください。
変換の過程と仮定を明示し、結果をJSON形式で出力してください。
"""),
            ("human", """以下の情報をもとに、ノードの値を日本円に変換してください。

ノード情報:
{node_info}

収集した追加情報:
{collected_info}

以下のJSON形式で出力してください:
```json
{{
  "converted_nodes": [
    {{
      "node_id": "ノードID",
      "original_value": "元の値",
      "converted_value": 数値,
      "unit": "円",
      "conversion_process": "変換過程の説明"
    }}
  ],
  "assumptions": ["仮定1", "仮定2"]
}}
```
""")
        ])
    
    def analyze_nodes(self, nodes: List[Dict]) -> Dict:
        """
        Analyze nodes to identify which ones need conversion and what information is needed
        
        Args:
            nodes: List of node dictionaries with node_id, name, and value
            
        Returns:
            Dictionary with analysis results
        """
        # Format node list for prompt
        node_list = "\n".join([
            f"- ID: {node['node_id']}, 名前: {node['name']}, 値: {node.get('value', '不明')}"
            for node in nodes
        ])
        
        # Get analysis from LLM
        response = self.llm.invoke(
            self.analysis_prompt.format(
                node_list=node_list
            )
        )
        
        # Extract JSON from response
        return self._extract_json(response.content)
    
    def generate_question(self, node_info: Dict, required_info: List[str], conversation_history: List[Tuple[str, str]]) -> str:
        """
        Generate a question to gather additional information needed for conversion
        
        Args:
            node_info: Information about the node
            required_info: List of required additional information
            conversation_history: List of (role, content) tuples representing conversation history
            
        Returns:
            Question to ask the user
        """
        # Format conversation history
        history_text = "\n".join([
            f"{'ユーザー' if role == 'user' else 'システム'}: {content}"
            for role, content in conversation_history
        ])
        
        # Generate question using LLM
        response = self.llm.invoke(
            self.question_prompt.format(
                node_info=json.dumps(node_info, ensure_ascii=False),
                required_info=", ".join(required_info),
                conversation_history=history_text
            )
        )
        
        return response.content
    
    def convert_nodes(self, nodes: List[Dict], collected_info: Dict) -> ConversionResult:
        """
        Convert node values to Japanese Yen based on collected information
        
        Args:
            nodes: List of node dictionaries
            collected_info: Dictionary of collected additional information
            
        Returns:
            ConversionResult object with converted nodes and assumptions
        """
        # Perform conversion using LLM
        response = self.llm.invoke(
            self.conversion_prompt.format(
                node_info=json.dumps(nodes, ensure_ascii=False),
                collected_info=json.dumps(collected_info, ensure_ascii=False)
            )
        )
        
        # Extract JSON from response
        result_json = self._extract_json(response.content)
        
        try:
            return ConversionResult(**result_json)
        except Exception as e:
            print(f"Error creating ConversionResult: {str(e)}")
            # Return default conversion result if parsing fails
            return ConversionResult(
                converted_nodes=[],
                assumptions=["変換処理中にエラーが発生しました"]
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