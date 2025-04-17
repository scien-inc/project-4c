"""
agents/roi_chat_agent.py
Improved ROI Chat Agent for targeted data collection with contextual understanding
and efficient parameter collection
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
    ROIツリーを対話的に構築するチャットエージェント（コンテキスト理解とパラメータ収集を改善）
    """
    def __init__(self, model_name="o3-mini"):
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
        
        # ノード理解用のプロンプト（追加）
        self.node_understanding_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
ユーザーとの対話を通じて、特定のROIノードに関する理解を深め、背景情報を収集します。

以下のノードとビジネス課題から、このDX施策の背景情報を分析してください：
- ノード名: {node_name}
- ビジネス課題: {business_challenge}

ユーザーの回答から以下の情報を抽出してください：
1. DX施策の種類や目的
2. 解決すべき具体的な業務課題
3. 使用するDXツールや技術
4. 期待される効果
5. 懸念事項やリスク

また、ユーザーの回答から得られたコンテキストを要約し、次のステップ（ROI計算に必要なパラメータ収集）に関連付けてください。
会話は自然でフレンドリーにし、専門用語は必要に応じて簡潔に説明してください。"""),
            ("human", "{user_message}")
        ])
        
        # パラメータ最適化プロンプト（追加）
        self.parameter_optimization_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
特定のノードのコンテキストを理解した上で、ROI計算に必要な最適なパラメータセットを特定してください。

ノード名: {node_name}
ビジネス課題: {business_challenge}
ユーザーの説明: {node_context}

このノードのROI計算に必要な最適なパラメータセットを特定してください。
パラメータは必要最小限にし、重要度順に並べてください。各パラメータには質問文も設定してください。

以下のJSON形式で回答してください：
```json
{
  "parameters": [
    {
      "name": "パラメータ名",
      "question": "このパラメータを聞くための質問文（コンテキストを踏まえた自然な質問）",
      "unit": "単位（円、%、人など）",
      "default": デフォルト値,
      "importance": 1-5の重要度（5が最高）
    }
  ]
}
```"""),
            ("human", "このDX施策の説明を理解し、ROI計算に必要な最適なパラメータセットを特定してください。")
        ])
        
        # コンテキスト認識パラメータ収集プロンプト（追加）
        self.contextual_parameter_collection_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
ユーザーとの対話の流れを踏まえて、自然な会話でROI計算に必要なパラメータを収集してください。

ノード名: {node_name}
ビジネス課題: {business_challenge}
ノードのコンテキスト: {node_context}
現在のパラメータ: {current_parameter}
これまでの収集情報: {collected_parameters}

以下のポイントを守ってください：
1. ユーザーの前回の回答を肯定的に受け止め、理解したことを示す
2. 質問は1つずつ行い、簡潔で具体的にする
3. 必要に応じて例や選択肢を提示する
4. 会話の自然な流れを維持する
5. 前の回答から推測できる情報は再度聞かない
6. ユーザーの回答が曖昧な場合のみ、より具体的に再質問する"""),
            ("human", "{user_message}")
        ])
        
        # ROI計算結果表示用のプロンプト（改善）
        self.roi_calculation_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
収集したパラメータを基に、選択されたノードのROI（投資対効果）を計算し、ビジネス的な洞察も含めて説明してください。

ノード名: {node_name}
ビジネス課題: {business_challenge}
ノードのコンテキスト: {node_context}

【収集したパラメータ】
{collected_parameters}

以下のポイントを含めて説明してください:
1. 単年ROIの計算結果（投資対効果率）
2. 投資額と期待効果の内訳
3. 投資回収期間
4. ビジネス的な意思決定のためのポイント（このROIが高いのか低いのか、改善の余地はあるか）
5. 提案や注意点（他のDX施策との比較や段階的導入の可能性など）

専門用語は必要に応じて簡潔に説明し、経営者やビジネス部門の方にも理解しやすい表現を心がけてください。
この結果がROIツリーに反映されることも伝えてください。"""),
            ("human", "{user_message}")
        ])
    
    def get_response(self, user_message: str, conversation_history: List, mermaid_diagram: str, 
                     current_focus: str = "node_understanding", callback=None, **context):
        """ユーザーメッセージに対する応答を生成する（コンテキスト理解とパラメータ収集を効率化）"""
        # 現在の処理モードに応じたプロンプトを選択
        if current_focus == "node_understanding":
            # ノード理解モード: DX施策の背景や目的を理解する
            node_name = context.get("node_name", "選択されたノード")
            business_challenge = context.get("business_challenge", "")
            
            messages = self.node_understanding_prompt.format_messages(
                node_name=node_name,
                business_challenge=business_challenge,
                user_message=user_message
            )
        
        elif current_focus == "parameter_collection":
            # パラメータ収集モード: 必要な数値情報を収集する
            node_name = context.get("node_name", "選択されたノード")
            business_challenge = context.get("business_challenge", "")
            node_context = context.get("node_context", "")
            current_parameter = context.get("current_parameter", {})
            collected_parameters = context.get("collected_parameters", {})
            
            messages = self.contextual_parameter_collection_prompt.format_messages(
                node_name=node_name,
                business_challenge=business_challenge,
                node_context=node_context,
                current_parameter=current_parameter,
                collected_parameters=collected_parameters,
                user_message=user_message
            )
        
        elif current_focus == "roi_calculation":
            # ROI計算モード: 収集した情報からROIを計算し説明する
            node_name = context.get("node_name", "選択されたノード")
            business_challenge = context.get("business_challenge", "")
            node_context = context.get("node_context", "")
            collected_parameters = context.get("collected_parameters", {})
            
            messages = self.roi_calculation_prompt.format_messages(
                node_name=node_name,
                business_challenge=business_challenge,
                node_context=node_context,
                collected_parameters=collected_parameters,
                user_message=user_message
            )
        
        else:
            # デフォルトの応答（エラー処理）
            return "申し訳ありません。現在の処理モードに対応できませんでした。"
        
        # LLMで応答を生成 （コールバック付きの場合はストリーミング）
        if callback:
            response = self.llm.with_config({"callbacks": [callback]}).invoke(messages)
        else:
            response = self.llm.invoke(messages)
        
        return response.content
    
    def analyze_node_context(self, node_name: str, business_challenge: str, user_message: str) -> Dict[str, Any]:
        """
        ユーザーの説明からノードの背景や目的を分析する
        
        Args:
            node_name: ノード名
            business_challenge: ビジネス課題のテキスト
            user_message: ユーザーの説明
            
        Returns:
            ノードのコンテキスト情報を含む辞書
        """
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
ユーザーの説明から、特定のROIノードに関する背景や目的を分析してください。

以下のJSON形式で回答してください：
```json
{
  "dx_initiative": "DX施策の種類や目的",
  "business_problems": ["解決すべき具体的な業務課題1", "業務課題2"],
  "dx_tools": ["使用するDXツール/技術1", "ツール2"],
  "expected_benefits": ["期待される効果1", "効果2"],
  "concerns": ["懸念事項/リスク1", "リスク2"],
  "summary": "コンテキストの要約（100文字程度）"
}
```"""),
            ("human", f"""ノード名: {node_name}
ビジネス課題: {business_challenge}
ユーザーの説明: {user_message}

このノードに関するコンテキスト情報を分析してください。""")
        ])
        
        # 分析を実行
        analysis_messages = analysis_prompt.format_messages()
        analysis_result = self.analysis_llm.invoke(analysis_messages)
        
        # JSONを抽出
        analysis_json = self._extract_json(analysis_result.content)
        
        return analysis_json or {
            "dx_initiative": "不明",
            "business_problems": ["具体的な業務課題が不明確です"],
            "dx_tools": ["具体的なツールが特定できていません"],
            "expected_benefits": ["効果が不明確です"],
            "concerns": ["リスクが特定できていません"],
            "summary": "コンテキスト情報が不足しています"
        }
    
    def determine_optimal_parameters(self, node_name: str, business_challenge: str, node_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        ノードのコンテキストに基づいて最適なパラメータセットを決定する
        
        Args:
            node_name: ノード名
            business_challenge: ビジネス課題のテキスト
            node_context: ノードのコンテキスト情報
            
        Returns:
            最適なパラメータのリスト
        """
        # コンテキスト情報を文字列に変換
        context_str = json.dumps(node_context, ensure_ascii=False, indent=2)
        
        messages = self.parameter_optimization_prompt.format_messages(
            node_name=node_name,
            business_challenge=business_challenge,
            node_context=context_str
        )
        
        # 最適なパラメータを決定
        params_result = self.analysis_llm.invoke(messages)
        params_json = self._extract_json(params_result.content)
        
        if params_json and "parameters" in params_json:
            return params_json["parameters"]
        
        # デフォルトのパラメータセット
        return [
            {
                "name": "初期導入コスト",
                "question": f"{node_name}の初期導入コストはいくらくらいを想定していますか？",
                "unit": "円",
                "default": 1000000,
                "importance": 5
            },
            {
                "name": "月額運用コスト",
                "question": "導入後の月々のランニングコストはいくらくらいになりそうですか？",
                "unit": "円/月",
                "default": 50000,
                "importance": 4
            },
            {
                "name": "月間効果額",
                "question": "この施策により、月々どのくらいの金銭的効果が見込めますか？",
                "unit": "円/月",
                "default": 150000,
                "importance": 5
            },
            {
                "name": "実装期間",
                "question": "この施策の実装にはどのくらいの期間が必要ですか？",
                "unit": "ヶ月",
                "default": 3,
                "importance": 3
            }
        ]
    
    def extract_parameter_from_message(self, user_message: str, parameter: Dict[str, Any], 
                                      node_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        ユーザーメッセージから特定のパラメータ値を抽出する（コンテキスト活用版）
        
        Args:
            user_message: ユーザーのメッセージ
            parameter: 抽出対象のパラメータ情報
            node_context: ノードのコンテキスト情報
            
        Returns:
            抽出結果を含む辞書
        """
        # コンテキスト情報を文字列に変換
        context_str = json.dumps(node_context, ensure_ascii=False, indent=2)
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
ユーザーのメッセージから、特定のパラメータの値を抽出してください。
コンテキスト情報も考慮して、最も適切な値を判断してください。

以下のJSON形式で回答してください：
```json
{
  "extracted": true/false,
  "value": 抽出した値（数値）,
  "unit": "抽出した単位（円、%など）",
  "confidence": 0-100の確信度,
  "reasoning": "この値を選んだ理由の簡単な説明"
}
```

値が明示的に見つからない場合でも、コンテキストから推測できる場合は推測値を示し、
その場合はconfidenceを50以下にしてください。
値が全く推測できない場合はextracted=falseとしてください。"""),
            ("human", f"""ノードのコンテキスト情報:
{context_str}

抽出するパラメータ:
- 名前: {parameter["name"]}
- 質問: {parameter["question"]}
- 単位: {parameter["unit"]}
- デフォルト値: {parameter["default"]}

ユーザーのメッセージ:
"{user_message}"

このメッセージから「{parameter["name"]}」の値を抽出してください。""")
        ])
        
        # パラメータ抽出を実行
        extraction_messages = extraction_prompt.format_messages()
        extraction_result = self.analysis_llm.invoke(extraction_messages)
        
        # JSONを抽出
        extracted_json = self._extract_json(extraction_result.content)
        
        if extracted_json and "extracted" in extracted_json:
            return extracted_json
        
        # 抽出失敗時のデフォルト値
        return {
            "extracted": False,
            "value": parameter["default"],
            "unit": parameter["unit"],
            "confidence": 0,
            "reasoning": "値を抽出できませんでした"
        }
    
    def calculate_roi_with_insights(self, node_name: str, business_challenge: str, 
                                   node_context: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        収集したパラメータからROIを計算し、ビジネス的な洞察を提供する
        
        Args:
            node_name: ノード名
            business_challenge: ビジネス課題のテキスト
            node_context: ノードのコンテキスト情報
            parameters: 収集したパラメータ値
            
        Returns:
            ROI計算結果と洞察を含む辞書
        """
        # コンテキスト情報を文字列に変換
        context_str = json.dumps(node_context, ensure_ascii=False, indent=2)
        parameters_str = json.dumps(parameters, ensure_ascii=False, indent=2)
        
        calculation_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
収集したパラメータを基にROIを計算し、ビジネス的な洞察も提供してください。

以下のJSON形式で回答してください：
```json
{
  "calculation": {
    "initial_cost": 初期コスト,
    "monthly_cost": 月額コスト,
    "annual_cost": 年間コスト,
    "monthly_benefit": 月間効果,
    "annual_benefit": 年間効果,
    "roi_percentage": ROI比率,
    "payback_period_months": 投資回収期間(月)
  },
  "insights": {
    "roi_evaluation": "ROIの評価（高い/中程度/低いなど）",
    "key_points": ["ビジネス的な洞察1", "洞察2"],
    "recommendations": ["提案1", "提案2"],
    "risks": ["リスク1", "リスク2"]
  },
  "summary": "ROI結果の簡潔なまとめと次のステップの提案"
}
```

基本的なROI計算式:
- 年間コスト = 初期コスト ÷ 償却年数 (通常3-5年) + 月額コスト × 12
- 年間効果 = 月間効果 × 12
- ROI比率 = (年間効果 - 年間コスト) ÷ 年間コスト × 100%
- 投資回収期間 = 初期コスト ÷ (月間効果 - 月額コスト)"""),
            ("human", f"""ノード名: {node_name}
ビジネス課題: {business_challenge}
ノードのコンテキスト情報:
{context_str}

収集したパラメータ:
{parameters_str}

これらの情報からROIを計算し、ビジネス的な洞察を提供してください。""")
        ])
        
        # ROI計算と洞察生成を実行
        calculation_messages = calculation_prompt.format_messages()
        calculation_result = self.analysis_llm.invoke(calculation_messages)
        
        # JSONを抽出
        roi_json = self._extract_json(calculation_result.content)
        
        if roi_json and "calculation" in roi_json and "insights" in roi_json:
            return roi_json
        
        # 計算失敗時のデフォルト値
        return {
            "calculation": {
                "initial_cost": parameters.get("初期導入コスト", {}).get("value", 1000000),
                "monthly_cost": parameters.get("月額運用コスト", {}).get("value", 50000),
                "annual_cost": parameters.get("初期導入コスト", {}).get("value", 1000000) / 3 + 
                               parameters.get("月額運用コスト", {}).get("value", 50000) * 12,
                "monthly_benefit": parameters.get("月間効果額", {}).get("value", 150000),
                "annual_benefit": parameters.get("月間効果額", {}).get("value", 150000) * 12,
                "roi_percentage": 50.0,
                "payback_period_months": 24
            },
            "insights": {
                "roi_evaluation": "判断できません",
                "key_points": ["パラメータが不足しているため、正確な分析ができません"],
                "recommendations": ["より詳細なパラメータ情報を収集してください"],
                "risks": ["不正確なROI計算に基づく意思決定のリスク"]
            },
            "summary": "ROI計算に必要な情報が不足しています。再度パラメータを収集してください。"
        }
    
    def generate_roi_summary_message(self, node_name: str, roi_results: Dict[str, Any]) -> str:
        """
        ROI計算結果から自然な説明メッセージを生成する
        
        Args:
            node_name: ノード名
            roi_results: ROI計算結果と洞察
            
        Returns:
            説明メッセージ
        """
        # ROI結果情報を文字列に変換
        roi_str = json.dumps(roi_results, ensure_ascii=False, indent=2)
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはDXプロジェクトのROI分析の専門家です。
ROI計算結果を自然な言葉で説明するメッセージを作成してください。
専門用語は必要に応じて簡潔に説明し、経営者やビジネス部門の方にも理解しやすい表現を心がけてください。

以下のポイントを含めてください:
1. ROI計算結果（投資対効果率）と投資回収期間
2. 投資額と期待効果の内訳
3. ビジネス的な洞察や提案
4. リスクや注意点
5. 次のステップの提案

この結果がROIツリーに反映されることも伝えてください。"""),
            ("human", f"""ノード名: {node_name}
ROI計算結果:
{roi_str}

これらの情報から、自然な説明メッセージを作成してください。""")
        ])
        
        # 説明メッセージ生成を実行
        summary_messages = summary_prompt.format_messages()
        summary_result = self.analysis_llm.invoke(summary_messages)
        
        return summary_result.content
    
    def extract_leaf_nodes(self, mermaid_code):
        """
        Mermaidコードから末端ノード（他のノードの親になっていないノード）を抽出する
        """
        return self.nlu_agent.extract_leaf_nodes(mermaid_code)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """JSONを抽出する"""
        try:
            # JSONブロックを探す
            pattern = r'```(?:json)?\s*(.*?)\s*```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            
            # JSON形式の部分を探す
            pattern = r'{.*}'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            
            return {}
        except Exception as e:
            print(f"JSON解析エラー: {e}")
            return {}