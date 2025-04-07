"""
agents/conversion_agent.py
Unit conversion agent for ROI calculations with self-reflection
"""
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import NumericalValue, UnitConversion


class ConversionAgent:
    """Agent for converting between different units for ROI calculation with self-reflection capabilities"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the conversion agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            streaming=False  # 高速レスポンスが必要
        )
        
        # Create prompt templates
        self.conversion_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは単位変換の専門家です。
異なる単位を持つ数値情報をROI計算に適した共通単位（通常は金額）に変換します。

あなたの役割:
1. 提供された数値の単位を判断する
2. 金額や経済的価値への変換が必要かを判断する
3. 変換のために必要な追加情報を特定する
4. 変換係数と計算式を提供する

レスポンスは以下のJSON形式で返してください:
```json
{{
  "can_convert": true/false,
  "needs_additional_info": true/false,
  "required_info": ["必要な情報1", "必要な情報2"],
  "suggested_questions": ["質問1", "質問2"],
  "conversion_factor": 数値 または null,
  "conversion_formula": "計算式",
  "explanation": "変換の説明"
}}
```"""),
            ("human", """以下の数値情報を分析し、ROI計算のために金額（円）に変換するために必要な情報と変換方法を提案してください。

数値: {{value}}
単位: {{unit}}
説明: {{description}}

すでに分かっている追加情報:
{{additional_info}}

JSON形式で回答してください。
""")
        ])
        
        # Prompt for actual conversion
        self.perform_conversion_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは単位変換の専門家です。
提供された数値と追加情報を使って、適切な単位変換を実行してください。

レスポンスは以下のJSON形式で返してください:
```json
{{
  "original_value": {数値},
  "original_unit": "元の単位",
  "converted_value": {変換後の数値},
  "converted_unit": "変換後の単位",
  "conversion_factor": {変換係数},
  "calculation": "計算式と計算過程",
  "explanation": "変換の説明"
}}
```"""),
            ("human", """以下の数値を変換してください。

元の数値: {{value}}
元の単位: {{unit}}
説明: {{description}}

追加情報:
{{additional_info}}

変換係数や計算式: {{conversion_info}}

この情報をもとに、数値を金額（円）に変換してください。
""")
        ])
        
        # 新規: セルフリフレクション用のプロンプトを追加
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは単位変換の専門家であり、自己評価能力に優れています。
実施した単位変換の品質と正確性を評価し、必要に応じて改善点を提案してください。

以下の点について評価してください:
1. 変換の正確性 - 数学的に正しいか
2. 前提条件の妥当性 - 仮定が合理的か
3. 代替手法の有無 - より良い変換方法はあるか
4. 不確実性の度合い - 変換にどの程度の確信があるか
5. エッジケースの考慮 - 特殊なケースへの対応

レスポンスは以下のJSON形式で返してください:
```json
{{
  "conversion_quality": 0-100,
  "confidence": 0-100,
  "strengths": ["強み1", "強み2"],
  "weaknesses": ["弱点1", "弱点2"],
  "improvement_suggestions": ["改善案1", "改善案2"],
  "needs_reconsideration": true/false,
  "alternative_conversion": {{
    "factor": 数値 または null,
    "formula": "代替計算式",
    "explanation": "代替案の説明"
  }}
}}
```"""),
            ("human", """以下の単位変換結果を評価してください：

元の数値: {{value}}
元の単位: {{unit}}
説明: {{description}}

追加情報:
{{additional_info}}

変換結果:
{{conversion_result}}

この変換の品質、正確性、改善点を評価してJSON形式で回答してください。
""")
        ])
    
    def analyze_conversion_needs(self, value: float, unit: str, description: str = "", additional_info: str = "") -> Dict[str, Any]:
        """
        Analyze what information is needed to convert a value to a common unit
        
        Args:
            value: Numerical value
            unit: Unit of the value
            description: Description of the value
            additional_info: Additional information already known
            
        Returns:
            Analysis of conversion needs
        """
        # Check if conversion is needed
        if unit == "円" or unit == "万円" or unit == "億円":
            # Already in monetary units, just need to standardize
            if unit == "万円":
                conversion_factor = 10000
                formula = f"{value} * 10000"
                converted_value = value * 10000
            elif unit == "億円":
                conversion_factor = 100000000
                formula = f"{value} * 100000000"
                converted_value = value * 100000000
            else:
                conversion_factor = 1
                formula = f"{value}"
                converted_value = value
                
            return {
                "can_convert": True,
                "needs_additional_info": False,
                "required_info": [],
                "suggested_questions": [],
                "conversion_factor": conversion_factor,
                "conversion_formula": formula,
                "converted_value": converted_value,
                "converted_unit": "円",
                "explanation": f"{unit}から円への変換"
            }
        
        # For other units, use LLM to analyze
        messages = self.conversion_prompt.format_messages(
            value=value,
            unit=unit,
            description=description,
            additional_info=additional_info
        )
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        analysis_json = self._extract_json(response.content)
        
        return analysis_json
    
    def perform_conversion(self, value: float, unit: str, description: str = "", additional_info: str = "", conversion_info: str = "") -> Dict[str, Any]:
        """
        Perform unit conversion based on provided information with self-reflection
        
        Args:
            value: Numerical value
            unit: Unit of the value
            description: Description of the value
            additional_info: Additional information for conversion
            conversion_info: Information about conversion factor or formula
            
        Returns:
            Conversion result with self-reflection
        """
        # For simple monetary conversions, handle directly
        if unit == "万円":
            conversion_result = {
                "original_value": value,
                "original_unit": unit,
                "converted_value": value * 10000,
                "converted_unit": "円",
                "conversion_factor": 10000,
                "calculation": f"{value} × 10000 = {value * 10000}",
                "explanation": "万円から円への変換"
            }
        elif unit == "億円":
            conversion_result = {
                "original_value": value,
                "original_unit": unit,
                "converted_value": value * 100000000,
                "converted_unit": "円",
                "conversion_factor": 100000000,
                "calculation": f"{value} × 100000000 = {value * 100000000}",
                "explanation": "億円から円への変換"
            }
        elif unit == "円":
            conversion_result = {
                "original_value": value,
                "original_unit": unit,
                "converted_value": value,
                "converted_unit": "円",
                "conversion_factor": 1,
                "calculation": f"{value}",
                "explanation": "すでに円単位のため変換不要"
            }
        else:
            # For other conversions, use LLM
            messages = self.perform_conversion_prompt.format_messages(
                value=value,
                unit=unit,
                description=description,
                additional_info=additional_info,
                conversion_info=conversion_info
            )
            
            response = self.llm.invoke(messages)
            
            # Extract JSON from response
            conversion_result = self._extract_json(response.content)
        
        # 新規: セルフリフレクションを実行
        reflection_result = self.self_reflect(value, unit, description, additional_info, conversion_result)
        
        # 新規: リフレクションの結果、再考が必要な場合は代替変換を採用
        if reflection_result.get("needs_reconsideration", False) and reflection_result.get("alternative_conversion"):
            alternative = reflection_result.get("alternative_conversion", {})
            
            if "factor" in alternative and alternative["factor"] is not None:
                # 代替変換を適用
                if unit == "%" and "割合" in description.lower():
                    # 特殊ケース: パーセンテージの変換の場合
                    try:
                        new_value = float(value) * alternative.get("factor", 1)
                        conversion_result["converted_value"] = new_value
                        conversion_result["conversion_factor"] = alternative.get("factor")
                        conversion_result["calculation"] = alternative.get("formula", f"{value} × {alternative.get('factor')} = {new_value}")
                        conversion_result["explanation"] = alternative.get("explanation", "リフレクションに基づく変換")
                    except (ValueError, TypeError):
                        # 数値変換エラーの場合は元の結果を使用
                        pass
                else:
                    # 通常の変換
                    try:
                        new_value = float(value) * alternative.get("factor", 1)
                        conversion_result["converted_value"] = new_value
                        conversion_result["conversion_factor"] = alternative.get("factor")
                        conversion_result["calculation"] = alternative.get("formula", f"{value} × {alternative.get('factor')} = {new_value}")
                        conversion_result["explanation"] = alternative.get("explanation", "リフレクションに基づく変換")
                    except (ValueError, TypeError):
                        # 数値変換エラーの場合は元の結果を使用
                        pass
        
        # リフレクション結果を追加
        conversion_result["reflection"] = {
            "quality": reflection_result.get("conversion_quality", 0),
            "confidence": reflection_result.get("confidence", 0),
            "strengths": reflection_result.get("strengths", []),
            "weaknesses": reflection_result.get("weaknesses", []),
            "improvement_suggestions": reflection_result.get("improvement_suggestions", [])
        }
        
        return conversion_result
    
    # 新規: セルフリフレクション機能
    def self_reflect(self, value: float, unit: str, description: str, additional_info: str, conversion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform self-reflection on a conversion result
        
        Args:
            value: Original numerical value
            unit: Unit of the original value
            description: Description of the value
            additional_info: Additional information used for conversion
            conversion_result: Result of the conversion
            
        Returns:
            Self-reflection on the conversion quality
        """
        # Simple unit conversions don't need complex reflection
        if unit in ["円", "万円", "億円"]:
            return {
                "conversion_quality": 100,
                "confidence": 100,
                "strengths": ["標準的な単位変換", "確実な変換係数"],
                "weaknesses": [],
                "improvement_suggestions": [],
                "needs_reconsideration": False
            }
        
        # For complex conversions, use LLM for reflection
        messages = self.reflection_prompt.format_messages(
            value=value,
            unit=unit,
            description=description,
            additional_info=additional_info,
            conversion_result=json.dumps(conversion_result, ensure_ascii=False, indent=2)
        )
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        reflection_json = self._extract_json(response.content)
        
        return reflection_json
    
    def get_conversion_questions(self, value: float, unit: str, description: str = "") -> List[str]:
        """
        Get questions to ask for converting a value
        
        Args:
            value: Numerical value
            unit: Unit of the value
            description: Description of the value
            
        Returns:
            List of questions to ask
        """
        # First analyze conversion needs
        analysis = self.analyze_conversion_needs(value, unit, description)
        
        # If additional info is needed, return suggested questions
        if analysis.get("needs_additional_info", False):
            return analysis.get("suggested_questions", [])
        
        # If no additional info is needed, return empty list
        return []
    
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