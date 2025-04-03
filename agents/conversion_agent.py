"""
agents/conversion_agent.py
Unit conversion agent for ROI calculations
"""
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import NumericalValue, UnitConversion


class ConversionAgent:
    """Agent for converting between different units for ROI calculation"""
    
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
{
  "can_convert": true/false,
  "needs_additional_info": true/false,
  "required_info": ["必要な情報1", "必要な情報2"],
  "suggested_questions": ["質問1", "質問2"],
  "conversion_factor": 数値 または null,
  "conversion_formula": "計算式",
  "explanation": "変換の説明"
}
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
        Perform unit conversion based on provided information
        
        Args:
            value: Numerical value
            unit: Unit of the value
            description: Description of the value
            additional_info: Additional information for conversion
            conversion_info: Information about conversion factor or formula
            
        Returns:
            Conversion result
        """
        # For simple monetary conversions, handle directly
        if unit == "万円":
            return {
                "original_value": value,
                "original_unit": unit,
                "converted_value": value * 10000,
                "converted_unit": "円",
                "conversion_factor": 10000,
                "calculation": f"{value} × 10000 = {value * 10000}",
                "explanation": "万円から円への変換"
            }
        elif unit == "億円":
            return {
                "original_value": value,
                "original_unit": unit,
                "converted_value": value * 100000000,
                "converted_unit": "円",
                "conversion_factor": 100000000,
                "calculation": f"{value} × 100000000 = {value * 100000000}",
                "explanation": "億円から円への変換"
            }
        elif unit == "円":
            return {
                "original_value": value,
                "original_unit": unit,
                "converted_value": value,
                "converted_unit": "円",
                "conversion_factor": 1,
                "calculation": f"{value}",
                "explanation": "すでに円単位のため変換不要"
            }
        
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
        conversion_json = self._extract_json(response.content)
        
        return conversion_json
    
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