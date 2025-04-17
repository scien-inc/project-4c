"""
agents/deepdive_agent.py
Improved Challenge Agent for ROI tree exploration with consistent granularity and DX tool focus
"""
from dotenv import load_dotenv
load_dotenv()        
from typing import Dict, List, Tuple, Optional, Any
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from domain.schemas import LeafNodeAnalysis, NumericalValue
from domain.roitree import ROINode, get_leaf_nodes, mermaid_to_roi_tree


# Revised Challenge agent system prompt to ensure diverse branching and avoid example copying
CHALLENGE_SYSTEM_PROMPT = """# ROI Tree Analysis Expert

あなたはビジネス課題をROIの観点から構造化する専門エージェントです。
ユーザーから与えられる事業・組織の課題文を解析し、ROIを「コスト削減」と「売上拡大」の2つに分岐してツリー構造を生成してください。

## ROIツリー作成の条件:
- 課題を「コスト削減」と「売上拡大」の2つの主要カテゴリに分解する
- 各カテゴリの下に、サブカテゴリが存在すれば具体的なサブカテゴリを作成する
- 各サブカテゴリの下にさらに具体的な項目やアプローチを特定する
- 最終的な末端ノードは、市場に出回っているDXツールで対応できる粒度にする
- 階層の深さはDXツールで対応できる粒度にまでにする、適切な分岐を持つツリー構造にする
- 文章中に明示的に数値目標が記載されている場合のみ、その値をノードに含める
- 数値目標が不明確な場合は、数値を含めずにノードを作成する
- 必ず具体的で現実的なビジネス状況に基づいたオリジナルのノード名を作成すること
- 下記の例はあくまで参考であり、そのまま使用しないこと

## 末端ノードの適切な粒度について:
以下は粒度の考え方であり、そのままコピーして使わないでください。課題に合わせた具体的なオリジナルの内容を考えてください。

- 適切な粒度の考え方：具体的なDXツールやソリューションで対応可能な業務課題を特定すること
  例えば「〇〇業務の△△プロセス自動化」「□□データの分析による××の予測」など
  
- 粒度が大きすぎる場合：抽象的で具体的なDXツールを特定できない
  例えば「業務効率化全般」「データ活用」などは粒度が大きすぎる

- 粒度が小さすぎる場合：特定のツールの細かい設定や機能に言及している
  例えば「ツールの特定機能の設定変更」などは粒度が小さすぎる

## 重要:
- 例として示した内容をそのままコピーしないこと
- 適切な数の分岐を必ず作成すること、結果的に一つの分岐になってもよい
- 現実的で具体的なビジネス課題に基づいたノード名を作成すること

## 出力:
- ROIツリーはMermaid記法で表現し、必ず "flowchart TD" または "graph TD" から始めること
- 数値目標が明確な場合のみ、ノードにその値を含める（例: `node1["製造効率向上 (1億円削減)"]`）
- 数値目標が不明確な場合はノードに値を含めない（例: `node1["製造効率向上"]`）
- DXツール名や必要パラメータをノードの中に直接記載しないこと
- 勝手に数値を追加しないこと。入力文から明示的に読み取れる数値のみを使用すること。
- 必ず ```mermaid と ``` で囲ってコードブロックとして返すこと
"""


class ChallengeAgent:
    """Improved agent for exploring business challenges and creating ROI trees with consistent granularity"""
    
    def __init__(self, model_name: str = "o3-mini"):
        """
        Initialize the challenge agent
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.llm = ChatOpenAI(
            model=model_name,
            #temperature=0.2,
            streaming=True  # ストリーミングを有効化
        )
        
        self.reflection_llm = ChatOpenAI(
            model=model_name,
            #temperature=0.1,
            streaming=False  # 分析には高速レスポンスが必要
        )
        
        # Create prompt templates
        self.challenge_prompt = ChatPromptTemplate.from_messages([
            ("system", CHALLENGE_SYSTEM_PROMPT),
            ("human", """以下のビジネス課題を解析し、ROIツリーを作成してください。
明示的に数値目標が示されている場合のみ、その値をノードに含めてください。数値が不明確な場合は、値を含めずにノードを作成してください。
末端ノードはDXツールで対応可能な具体的な粒度にしてください。

各カテゴリとサブカテゴリには、分岐が存在すれば分岐を作成し、例にとらわれない具体的でオリジナルなノード名を考えてください。

ビジネス課題:
{{challenge_text}}

Mermaid記法でROIツリーを表現してください。数値目標がわかる場合のみ、それをノードに含めてください。
各末端ノードには、ROI計算に必要なパラメータ情報を付記してください。

以下のフォーマットで返答してください:
```mermaid
flowchart TD
    root["ROIツリー"]
    ...（ここにツリーの内容）...
```
""")
        ])
        
        # その他のプロンプトテンプレートは変更なし（略）
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリーの分析専門家です。末端ノード（子を持たないノード）の分析をして、
数値目標が適切に設定されているかを判断してください。また、各ノードがDXツールで対応可能な適切な粒度かも評価してください。

分析では以下を判定してください:
1. すべての末端ノードに数値目標が設定されているか
2. 数値目標が設定されていないノードはどれか
3. 各末端ノードにどのような種類の数値情報が必要か（金額、時間、件数、人数など）
4. 各末端ノードは適切な粒度か（特定のDXツールで対応可能か）
5. ツリー全体の完成度（パーセンテージ）

レスポンスは以下のJSON形式で返してください:
```json
{{
  "has_numerical_data": true/false,
  "missing_nodes": ["ノード名1", "ノード名2"],
  "incomplete_nodes": [
    {{"name": "ノード名", "issue": "問題の説明", "suggestion": "改善提案", "value_type": "数値の種類（金額/時間/件数/人数など）", "value_unit": "単位（円/時間/件/人など）", "required_params": ["パラメータ1", "パラメータ2"]}}
  ],
  "granularity_issues": [
    {{"name": "ノード名", "issue": "粒度の問題点", "suggestion": "より適切な粒度の提案", "dx_tools": ["対応可能なDXツール1", "対応可能なDXツール2"]}}
  ],
  "branching_issues": [
    {{"name": "カテゴリ名", "current_branches": 数, "issue": "分岐数が少なすぎます", "suggestion": "さらに追加すべき分岐の例"}}
  ],
  "completion_percentage": 0-100
}}
```"""),
            ("human", "以下のROIツリーの末端ノードを分析してください：\n\n{mermaid_diagram}\n\n末端ノードのリスト：\n{leaf_nodes}")
        ])
        
        self.solution_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはビジネス課題に対して最適なDXソリューションを提案する専門家です。
ROIツリーの末端ノードを分析し、各ノードに適した具体的なDXツールやソリューションと必要な情報を特定してください。

各末端ノードに対して、以下の情報を提供してください:
1. 推奨する具体的なDXツールやソリューション（例: 特定のRPAツール、AIチャットボット製品、クラウドERPなど）
2. ROI計算に必要な追加情報（例: 導入コスト、月額費用、工数削減量、対象プロセス数など）
3. 優先度の判断基準（実装難易度、期待効果、緊急性など）

レスポンスは以下のJSON形式で返してください:
```json
{{
  "solution_suggestions": [
    {{
      "node_name": "ノード名",
      "dx_tools": ["具体的なDXツール1", "DXツール2"],
      "required_information": ["必要情報1", "必要情報2"],
      "common_parameters": ["共通パラメータ1", "共通パラメータ2"],
      "node_specific_parameters": ["ノード固有パラメータ1", "ノード固有パラメータ2"],
      "priority_criteria": ["基準1", "基準2"]
    }}
  ],
  "common_parameters": {{
    "人件費": "必要な人件費情報の説明",
    "工数": "必要な工数情報の説明",
    "その他共通パラメータ": "説明"
  }}
}}
```"""),
            ("human", "以下のROIツリーの末端ノードに対する解決策の種類と必要な情報を特定してください：\n\n{mermaid_diagram}\n\n末端ノードのリスト：\n{leaf_nodes}")
        ])
    
# ChallengeAgentのcreate_roi_treeメソッド内のプロンプト部分の修正

    def create_roi_tree(self, challenge_text: str) -> Tuple[ROINode, str]:
        """
        Create an ROI tree from a challenge description
        
        Args:
            challenge_text: Text describing the business challenge
            
        Returns:
            Tuple of (ROI tree root node, Mermaid diagram)
        """
        # 最大再試行回数
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # より具体的なプロンプトを用意
                explicit_prompt = f"""
    以下のビジネス課題を解析し、ROIツリーを作成してください。課題文を注意深く読み、ROIの観点から構造化してください。

    ビジネス課題:
    「{challenge_text}」

    この課題に基づいて、ROIツリーをMermaid記法で作成してください。
    以下の点に注意してください：
    1. 「コスト削減」と「売上拡大」の2つの主要カテゴリから始めること
    2. 各カテゴリには具体的なサブカテゴリが存在すれば作成すること
    3. 数値目標が明確な場合のみ、その値をノードに含めること
    4. ノードIDは英数字のみを使用すること (例: cost1, rev2 など)

    以下の形式で返答してください:
    ```mermaid
    flowchart TD
        root["ROI"]
        cost["コスト削減"]
        revenue["売上拡大"]
        
        root --> cost
        root --> revenue
        
        cost1["コスト削減の具体例1"]
        cost2["コスト削減の具体例2"]
        ...
    ```
                """
                
                # Generate ROI tree using LLM (直接メッセージを作成)
                messages = [
                    {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
                    {"role": "user", "content": explicit_prompt}
                ]
                
                print("Challenge text:", challenge_text)  # デバッグ出力
                
                response = self.llm.invoke(messages)
                
                # デバッグ: LLMのレスポンスを出力
                print("LLMレスポンス:")
                print("-" * 40)
                print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
                print("-" * 40)
                
                # Extract Mermaid diagram from response
                mermaid_diagram = self._extract_mermaid(response.content)
                
                # デバッグ: 抽出されたMermaidダイアグラムを出力
                print("抽出されたMermaidダイアグラム:")
                print("-" * 40)
                print(mermaid_diagram[:500] + "..." if mermaid_diagram and len(mermaid_diagram) > 500 else mermaid_diagram)
                print("-" * 40)
                
                # 以下は変更なし...
                
                # Check if mermaid diagram is valid
                if not mermaid_diagram or not mermaid_diagram.strip():
                    print(f"生成されたMermaidダイアグラムが空です。再試行します。({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    continue
                
                # Mermaidダイアグラムが flowchart TD または graph TD で始まることを確認
                if not mermaid_diagram.strip().startswith(("flowchart TD", "graph TD")):
                    # 先頭に追加
                    mermaid_diagram = "flowchart TD\n" + mermaid_diagram
                    print("Mermaidダイアグラムに先頭行を追加しました")
                
                # Convert Mermaid diagram to ROI tree
                root_node = mermaid_to_roi_tree(mermaid_diagram)
                
                # Check if parsing succeeded
                if root_node:
                    return root_node, mermaid_diagram
                
                print(f"Mermaidダイアグラムのパースに失敗しました。再試行します。({retry_count + 1}/{max_retries})")
                retry_count += 1
                
            except Exception as e:
                print(f"ROIツリー生成中にエラーが発生しました: {str(e)}. 再試行します。({retry_count + 1}/{max_retries})")
                retry_count += 1
        
        # すべての再試行が失敗した場合は、基本的なMermaidダイアグラムとROIツリーを作成して返す
        print("最大再試行回数に達しました。基本的なROIツリーを生成します。")
        basic_mermaid = """flowchart TD
    root["ROI"]
    cost["コスト削減"]
    revenue["売上拡大"]
    
    root --> cost
    root --> revenue
    
    cost1["業務効率化"]
    cost2["リソース最適化"]
    cost3["IT基盤最適化"]
    
    cost --> cost1
    cost --> cost2
    cost --> cost3
    
    rev1["顧客体験向上"]
    rev2["市場拡大"]
    rev3["商品・サービス革新"]
    
    revenue --> rev1
    revenue --> rev2
    revenue --> rev3
        """
        
        # 基本的なMermaidダイアグラムをパースしてROIノードを作成
        basic_root_node = mermaid_to_roi_tree(basic_mermaid)
        
        # パースが成功した場合はそれを返し、失敗した場合は例外を発生させる
        if basic_root_node:
            return basic_root_node, basic_mermaid
        else:
            raise ValueError("ROIツリーの生成に失敗しました。問題が解決しない場合は、別の表現で課題を記述してみてください。")
    
    def analyze_leaf_nodes(self, root_node: ROINode, mermaid_diagram: str) -> LeafNodeAnalysis:
        """
        Analyze leaf nodes to check if they have appropriate numerical targets and granularity
        
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
            # Add granularity_issues to the analysis if not present
            if "granularity_issues" not in analysis_json:
                analysis_json["granularity_issues"] = []
            
            # Add branching_issues to the analysis if not present
            if "branching_issues" not in analysis_json:
                analysis_json["branching_issues"] = []
                
            return LeafNodeAnalysis(**analysis_json)
        except Exception as e:
            print(f"Error parsing leaf node analysis: {str(e)}")
            # Return default analysis if parsing fails
            return LeafNodeAnalysis(
                has_numerical_data=any(node.value is not None for node in leaf_nodes),
                missing_nodes=[node.name for node in leaf_nodes if node.value is None],
                incomplete_nodes=[],
                granularity_issues=[],
                branching_issues=[],
                completion_percentage=50.0
            )
    
    def suggest_solutions(self, root_node: ROINode, mermaid_diagram: str) -> Dict[str, Any]:
        """
        Suggest DX tool solutions for leaf nodes
        
        Args:
            root_node: ROI tree root node
            mermaid_diagram: Mermaid diagram of the ROI tree
            
        Returns:
            Solution suggestions for leaf nodes with specific DX tools
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
        """Extract Mermaid diagram from text - 改善版"""
        
        # パターン1: ```mermaid...```形式のコードブロック
        mermaid_pattern1 = re.search(r'```mermaid\s*([\s\S]*?)```', text, re.DOTALL)
        if mermaid_pattern1:
            return mermaid_pattern1.group(1).strip()
        
        # パターン2: ```内にflowchart TDやgraph TD形式で記述されているコードブロック
        mermaid_pattern2 = re.search(r'```\s*((?:flowchart|graph)\s+TD[\s\S]*?)```', text, re.DOTALL)
        if mermaid_pattern2:
            return mermaid_pattern2.group(1).strip()
        
        # パターン3: コードブロック以外の場所に書かれているflowchart TDやgraph TD
        mermaid_pattern3 = re.search(r'((?:flowchart|graph)\s+TD[\s\S]*?)(?:\n\n|$)', text, re.DOTALL)
        if mermaid_pattern3:
            # 次の空行または文書の終わりまでを取得
            return mermaid_pattern3.group(1).strip()
        
        # パターン4: コードブロックはないが、明示的なノード定義があるもの
        node_pattern = re.search(r'([A-Za-z0-9_]+\s*\[\s*"[^"]*"\][\s\S]*?(?:\n\n|$))', text, re.DOTALL)
        if node_pattern:
            # ノード定義を見つけたので、マーメイドダイアグラムの形式に変換
            return "flowchart TD\n" + node_pattern.group(1).strip()
            
        # パターン5: フォールバック - 何も見つからない場合は最初の行から解析し、
        # ノード定義らしき行が見つかったらそれ以降を抽出
        lines = text.split('\n')
        mermaid_lines = []
        capture_started = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # ノード定義またはエッジ定義を検出
            if re.match(r'[A-Za-z0-9_]+\s*(\[|\(|--)', line) or capture_started:
                capture_started = True
                mermaid_lines.append(line)
        
        if mermaid_lines:
            return "flowchart TD\n" + "\n".join(mermaid_lines)
            
        # 何も見つからなかった場合は空文字列を返す
        return ""
    
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

# Extended LeafNodeAnalysis model
if "granularity_issues" not in dir(LeafNodeAnalysis):
    from typing import List, Dict
    from pydantic import Field
    
    # Extend the LeafNodeAnalysis model to include granularity issues and branching issues
    class ExtendedLeafNodeAnalysis(LeafNodeAnalysis):
        """Extended analysis of leaf nodes including granularity issues and branching issues"""
        granularity_issues: List[Dict[str, Any]] = Field(default_factory=list)
        branching_issues: List[Dict[str, Any]] = Field(default_factory=list)