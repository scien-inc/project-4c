"""
main.py
Improved Streamlit app for ROI Analysis with regeneration capabilities and enhanced chat-based data collection
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re
import json
import time
import random
from typing import List, Dict, Tuple, Optional, Any, Callable

from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.deepdive_agent import ChallengeAgent
from agents.proposal_agent import ProposalAgent
from agents.solution_agent import SolutionAgent
from agents.roi_chat_agent import ROIChatAgent
from agents.conversion_agent import ConversionAgent
from domain.roitree import ROINode, mermaid_to_roi_tree, get_leaf_nodes
from domain.schemas import NumericalValue
from agents.deepdive_agent import CHALLENGE_SYSTEM_PROMPT


# Streamlitストリーミング出力のためのハンドラー
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.message_placeholder = container.empty()
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """LLMから新しいトークンが生成されたときに呼ばれる"""
        self.text += token
        self.message_placeholder.markdown(self.text + "▌")
    
    def on_llm_end(self, response, **kwargs) -> None:
        """LLMの生成が終了したときに呼ばれる"""
        self.message_placeholder.markdown(self.text)


def render_mermaid(mermaid_syntax, height=500):
    """
    HTMLとJavaScriptを使用してStreamlit上でMermaidダイアグラムをレンダリングする
    """
    # Mermaid構文を清掃（先頭と末尾の不要な空白行を削除）
    mermaid_syntax = mermaid_syntax.strip()
    
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@9.3.0/dist/mermaid.min.js"></script>
    <div class="mermaid" style="width: 100%;">
    {mermaid_syntax}
    </div>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'default',
            flowchart: {{ 
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
    """
    components.html(html, height=height)


def extract_leaf_nodes(mermaid_code):
    """
    Mermaidコードから末端ノード（他のノードの親になっていないノード）を抽出する
    """
    # Noneや空文字列の場合は空のリストを返す
    if not mermaid_code:
        return []
        
    all_node_ids = set()
    parent_node_ids = set()
    node_definitions = {}
    lines = mermaid_code.strip().split('\n')
    
    # まずノードの定義をキャプチャ
    for line in lines:
        # ノード定義をキャプチャ（例: node1["テキスト"]）
        matches = re.findall(r'(\w+)\[\"([^\"]+)\"', line)
        for match in matches:
            node_id = match[0]
            node_label = match[1]
            all_node_ids.add(node_id)
            node_definitions[node_id] = node_label
    
    # 次に親ノードを見つける
    for line in lines:
        # エッジ定義をキャプチャ（例: node1 --> node2）
        edge_matches = re.findall(r'(\w+)\s*-->', line)
        for match in edge_matches:
            parent_node_ids.add(match)
    
    # 末端ノードは、親になっていないノード
    leaf_nodes = all_node_ids - parent_node_ids
    
    # 末端ノードとそのラベルを返す
    result = [(node_id, node_definitions.get(node_id, "")) for node_id in leaf_nodes]
    return result


def extract_non_leaf_nodes(mermaid_code):
    """
    Mermaidコードから非末端ノード（子ノードを持つノード）を抽出する
    """
    if not mermaid_code:
        return []
        
    all_node_ids = set()
    parent_node_ids = set()
    child_node_ids = set()
    node_definitions = {}
    lines = mermaid_code.strip().split('\n')
    
    # まずノードの定義をキャプチャ
    for line in lines:
        # ノード定義をキャプチャ（例: node1["テキスト"]）
        matches = re.findall(r'(\w+)\[\"([^\"]+)\"', line)
        for match in matches:
            node_id = match[0]
            node_label = match[1]
            all_node_ids.add(node_id)
            node_definitions[node_id] = node_label
    
    # 親子関係を抽出
    for line in lines:
        # エッジ定義をキャプチャ（例: node1 --> node2）
        edge_matches = re.findall(r'(\w+)\s*-->\s*(\w+)', line)
        for match in edge_matches:
            parent_id = match[0]
            child_id = match[1]
            parent_node_ids.add(parent_id)
            child_node_ids.add(child_id)
    
    # 非末端ノード（子を持つノード）のみを返す
    non_leaf_nodes = parent_node_ids
    
    # 非末端ノードとそのラベルを返す
    result = [(node_id, node_definitions.get(node_id, "")) for node_id in non_leaf_nodes]
    return result

def get_child_nodes(mermaid_code, parent_node_id):
    """
    特定の親ノードの子ノードを抽出する
    """
    if not mermaid_code or not parent_node_id:
        return []
    
    children = set()
    node_definitions = {}
    lines = mermaid_code.strip().split('\n')
    
    # ノード定義を取得
    for line in lines:
        matches = re.findall(r'(\w+)\[\"([^\"]+)\"', line)
        for match in matches:
            node_id = match[0]
            node_label = match[1]
            node_definitions[node_id] = node_label
    
    # 親ノードからの直接の子ノードを抽出
    for line in lines:
        edge_matches = re.findall(rf'{parent_node_id}\s*-->\s*(\w+)', line)
        for child_id in edge_matches:
            children.add(child_id)
    
    # 子ノードとそのラベルを返す
    result = [(node_id, node_definitions.get(node_id, "")) for node_id in children]
    return result


def has_numerical_value(node_label: str) -> bool:
    """
    ノードラベルに数値情報（金額、パーセンテージなど）が含まれているかをチェック
    """
    # 金額のパターン（例: 1億円, 500万円, 1,000円）
    amount_pattern = r'(\d+[,\.]?\d*\s*[億万千]?円)'
    # パーセンテージのパターン（例: 25%, 3.5%）
    percentage_pattern = r'(\d+[,\.]?\d*\s*%)'
    # 一般的な数値パターン（例: 100件, 50人）
    general_pattern = r'(\d+[,\.]?\d*\s*[件人時分秒])'
    
    return bool(re.search(amount_pattern, node_label)) or \
           bool(re.search(percentage_pattern, node_label)) or \
           bool(re.search(general_pattern, node_label))


def extract_nodes_without_values(mermaid_code):
    """
    数値情報を持たない末端ノードを抽出する
    """
    leaf_nodes = extract_leaf_nodes(mermaid_code)
    nodes_without_values = []
    
    for node_id, node_label in leaf_nodes:
        if not has_numerical_value(node_label):
            nodes_without_values.append((node_id, node_label))
    
    return nodes_without_values


def update_node_value_in_mermaid(mermaid_code, node_id, new_value):
    """
    指定したノードIDのノードに数値を追加または更新する
    """
    lines = mermaid_code.strip().split('\n')
    updated_lines = []
    
    for line in lines:
        # ノード定義を検索（例: node1["テキスト"]）
        node_match = re.search(rf'({node_id}\[\")([^\"]+)(\"\])', line)
        if node_match:
            # 既存の値の有無をチェック
            current_label = node_match.group(2)
            value_pattern = r'\(([^)]+)\)'
            value_match = re.search(value_pattern, current_label)
            
            if value_match:
                # 既存の値を更新
                updated_label = re.sub(value_pattern, f'({new_value})', current_label)
            else:
                # 新しい値を追加
                updated_label = f"{current_label} ({new_value})"
            
            # 行を更新
            updated_line = f'{node_match.group(1)}{updated_label}{node_match.group(3)}'
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


# ROI情報を含むノード値更新関数
def update_node_value_with_roi_data(mermaid_code, node_id, value, cost=None, benefit=None, implementation_period=None, priority=None):
    """
    指定したノードIDのノードに数値とROI情報を追加または更新する
    """
    lines = mermaid_code.strip().split('\n')
    updated_lines = []
    
    for line in lines:
        # ノード定義を検索（例: node1["テキスト"]）
        node_match = re.search(rf'({node_id}\[\")([^\"]+)(\"\])', line)
        if node_match:
            # 既存の値の有無をチェック
            current_label = node_match.group(2)
            value_pattern = r'\(([^)]+)\)'
            value_match = re.search(value_pattern, current_label)
            
            # ROI情報を構築
            roi_info = []
            
            # コスト情報が提供されている場合
            if cost is not None:
                if cost >= 100000000:  # 1億円以上
                    roi_info.append(f"コスト:{cost/100000000:.1f}億円")
                elif cost >= 10000:  # 1万円以上
                    roi_info.append(f"コスト:{cost/10000:.1f}万円")
                else:
                    roi_info.append(f"コスト:{cost:.0f}円")
            
            # ベネフィット情報が提供されている場合
            if benefit is not None:
                if benefit >= 100000000:  # 1億円以上
                    roi_info.append(f"効果:{benefit/100000000:.1f}億円")
                elif benefit >= 10000:  # 1万円以上
                    roi_info.append(f"効果:{benefit/10000:.1f}万円")
                else:
                    roi_info.append(f"効果:{benefit:.0f}円")
            
            # ROIの計算（コストと利益の両方が提供されている場合）
            if cost is not None and benefit is not None and cost > 0:
                roi = (benefit - cost) / cost * 100
                roi_info.append(f"ROI:{roi:.1f}%")
            
            # 実装期間が提供されている場合
            if implementation_period:
                roi_info.append(f"期間:{implementation_period}")
                
            # 優先度が提供されている場合
            if priority:
                roi_info.append(f"優先度:{priority}")
            
            # 更新するノード値を構築
            if value_match:
                # 既存の値を更新
                updated_label = re.sub(value_pattern, f'({value})', current_label)
            else:
                # 新しい値を追加
                updated_label = f"{current_label} ({value})"
            
            # ROI情報を追加（あれば）
            if roi_info:
                roi_text = "/".join(roi_info)
                # すでにROI情報があるかチェック
                roi_pattern = r'\[([^\]]+)\]'
                roi_match = re.search(roi_pattern, updated_label)
                
                if roi_match:
                    # 既存のROI情報を更新
                    updated_label = re.sub(roi_pattern, f'[{roi_text}]', updated_label)
                else:
                    # 新しいROI情報を追加
                    updated_label = f"{updated_label} [{roi_text}]"
            
            # 行を更新
            updated_line = f'{node_match.group(1)}{updated_label}{node_match.group(3)}'
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


# 数値から金額を抽出する関数
def extract_numerical_value(value_str):
    """
    文字列から数値を抽出して円単位に変換する
    
    Args:
        value_str: 数値を含む文字列（単位付き）
        
    Returns:
        円単位の数値
    """
    # カンマを削除
    clean_str = value_str.replace(',', '')
    
    # 数値と単位を抽出
    number_match = re.search(r'(\d+\.?\d*)', clean_str)
    if not number_match:
        return None
        
    number = float(number_match.group(1))
    
    # 単位による変換
    if '億円' in clean_str:
        return number * 100000000
    elif '万円' in clean_str:
        return number * 10000
    elif '千円' in clean_str:
        return number * 1000
    elif '%' in clean_str:
        return number  # パーセンテージはそのまま返す
    elif '円' in clean_str:
        return number
    else:
        return number  # 単位が不明な場合はそのまま返す
# Regeneration functions that use CHALLENGE_SYSTEM_PROMPT from deepdive_agent.py
def regenerate_roi_tree(agent, challenge_text, current_mermaid):
    """
    現在のROIツリーを参考にして新しいROIツリーを生成する
    
    Args:
        agent: ChallengeAgent のインスタンス
        challenge_text: ビジネス課題テキスト
        current_mermaid: 現在のMermaidダイアグラム
        
    Returns:
        生成されたROIツリーのルートノードとMermaidダイアグラム
    """
    # カスタムプロンプトを作成（現在のツリーをfew-shotとして含める）
    custom_prompt = f"""
以下のビジネス課題を解析し、ROIツリーを再生成してください。

ビジネス課題:
「{challenge_text}」

以下が現在のROIツリーです。これを参考にしつつ、より適切なROIツリーを生成してください:

```mermaid
{current_mermaid}
```

現在のツリーを踏まえて改善点を考慮し、より適切な粒度と構造を持つROIツリーを生成してください。
末端ノードは特定のDXツールで対応可能な具体的な粒度にしてください。
必要に応じて新しいノードを追加したり、不適切なノードを削除・修正したりしても構いません。

以下の形式で返答してください:
```mermaid
flowchart TD
    root["ROI"]
    cost["コスト削減"]
    revenue["売上拡大"]
    ...
```
"""
    
    # LLMを使って新しいツリーを生成
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
    )
    
    # deepdive_agent.pyのCHALLENGE_SYSTEM_PROMPTを使用
    messages = [
        {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
        {"role": "user", "content": custom_prompt}
    ]
    
    response = llm.invoke(messages)
    
    # Mermaidダイアグラムを抽出
    mermaid_diagram = agent._extract_mermaid(response.content)
    
    # Mermaidダイアグラムの確認
    if not mermaid_diagram or not mermaid_diagram.strip():
        raise ValueError("生成されたMermaidダイアグラムが空です。")
    
    # Mermaidダイアグラムがflowchart TDまたはgraph TDで始まることを確認
    if not mermaid_diagram.strip().startswith(("flowchart TD", "graph TD")):
        mermaid_diagram = "flowchart TD\n" + mermaid_diagram
    
    # ROIツリーに変換
    root_node = mermaid_to_roi_tree(mermaid_diagram)
    
    return root_node, mermaid_diagram




def regenerate_child_nodes(agent, challenge_text, mermaid_diagram, parent_node_id):
    """
    特定の親ノードの子ノードを再生成する
    
    Args:
        agent: ChallengeAgent のインスタンス
        challenge_text: ビジネス課題テキスト
        mermaid_diagram: 現在のMermaidダイアグラム
        parent_node_id: 子ノードを再生成する親ノードのID
        
    Returns:
        更新されたMermaidダイアグラム
    """
    # 親ノードのラベルを取得
    parent_label = ""
    lines = mermaid_diagram.strip().split('\n')
    for line in lines:
        parent_match = re.search(rf'{parent_node_id}\[\s*"([^"]+)"\s*\]', line)
        if parent_match:
            parent_label = parent_match.group(1)
            break
    
    if not parent_label:
        raise ValueError(f"親ノード {parent_node_id} が見つかりません。")
    
    # 現在の子ノードを取得
    child_nodes = get_child_nodes(mermaid_diagram, parent_node_id)
    child_ids = [node_id for node_id, _ in child_nodes]
    
    # カスタムプロンプトを作成
    custom_prompt = f"""
以下のビジネス課題とROIツリーを参考に、特定の親ノードの子ノードを再生成してください。

ビジネス課題:
「{challenge_text}」

現在のROIツリー全体:
```mermaid
{mermaid_diagram}
```

再生成する対象の親ノード: {parent_node_id}["{parent_label}"]

この親ノードに対して、より適切な子ノードを生成してください。
子ノードは特定のDXツールで対応可能な具体的な粒度にしてください。
ツリー全体の構造を考慮して、バランスの取れた分岐と適切な粒度の子ノードを生成してください。

以下の形式で、新しい子ノードのみを返答してください:
```mermaid
{parent_node_id} --> new_child1["新しい子ノード1"]
{parent_node_id} --> new_child2["新しい子ノード2"]
...
```
"""
    
    # LLMを使って新しい子ノードを生成
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
    )
    
    # deepdive_agent.pyのCHALLENGE_SYSTEM_PROMPTを使用
    messages = [
        {"role": "system", "content": CHALLENGE_SYSTEM_PROMPT},
        {"role": "user", "content": custom_prompt}
    ]
    
    response = llm.invoke(messages)
    
    # 新しい子ノードの定義を抽出
    new_children_mermaid = agent._extract_mermaid(response.content)
    
    if not new_children_mermaid:
        # Mermaidブロックが見つからない場合、テキスト全体から親ノードの子ノード定義を抽出
        lines = response.content.strip().split('\n')
        new_children_lines = []
        
        for line in lines:
            if re.search(rf'{parent_node_id}\s*-->', line):
                new_children_lines.append(line)
        
        new_children_mermaid = '\n'.join(new_children_lines)
    
    if not new_children_mermaid:
        raise ValueError("新しい子ノードの定義が生成されませんでした。")
    
    # 現在のMermaidダイアグラムから既存の子ノード接続を削除
    updated_lines = []
    for line in lines:
        # 親ノードから子ノードへの接続を除外
        if not any(re.search(rf'{parent_node_id}\s*-->\s*{child_id}', line) for child_id in child_ids):
            updated_lines.append(line)
    
    # 子ノード自体の定義も削除
    final_lines = []
    for line in updated_lines:
        if not any(re.search(rf'{child_id}\[\s*"[^"]+"\s*\]', line) for child_id in child_ids):
            final_lines.append(line)
    
    # 新しい子ノード定義を追加
    # 通常、最後の行（クラス定義など）の前に挿入
    if final_lines and final_lines[-1].strip().startswith("classDef"):
        final_lines = final_lines[:-1] + new_children_mermaid.strip().split('\n') + [final_lines[-1]]
    else:
        final_lines.extend(new_children_mermaid.strip().split('\n'))
    
    return '\n'.join(final_lines)


def determine_required_parameters(node_label, challenge_text):
    """
    ノードの内容とビジネス課題から必要なパラメータを判断する
    JSON解析エラーに対する堅牢性を向上
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    parameter_prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはDXプロジェクトのROI分析専門家です。
与えられた末端ノードと課題文から、ROI計算に必要なパラメータを判断してください。

ROI計算には通常、以下のような項目が必要です：
1. 初期導入コスト（初期投資額）
2. 運用コスト（月額または年額）
3. 効果金額（削減額や増収額）
4. 実装期間
5. 対象範囲（対象プロセス数、対象ユーザー数など）

ノードの内容と課題文を分析し、必要なパラメータと具体的な質問文を提案してください。

以下のJSON形式で回答してください：
```json
{{
  "parameters": [
    {{
      "name": "パラメータ名",
      "question": "このパラメータを聞くための質問文",
      "unit": "単位（円、%、人など）",
      "default": デフォルト値,
      "importance": 1-5の重要度（5が最高）
    }}
  ]
}}
```"""),
        ("human", f"""以下の情報から必要なパラメータを判断してください：

末端ノード：「{node_label}」

ビジネス課題：
{challenge_text}

このノードのROI計算に必要なパラメータと質問内容を提案してください。""")
    ])
    
    # 正しくメッセージを生成
    messages = parameter_prompt.format_messages()
    response = llm.invoke(messages)
    
    # JSONを抽出 - より堅牢な処理
    try:
        # JSONブロックを探す - パターンを最適化
        pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(pattern, response.content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            # 余分な空白や改行を削除
            json_str = re.sub(r'\s+', ' ', json_str)
            return json.loads(json_str)
        
        # JSON形式の部分を探す - より柔軟なパターン
        pattern = r'{.*}'
        match = re.search(pattern, response.content, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
            return json.loads(json_str)
        
        # デフォルト値を返す
        print("JSONパターンが見つかりませんでした。デフォルト値を使用します。")
        return {{"parameters": [
            {
                "name": "初期導入コスト",
                "question": "このDX施策の初期導入コスト（初期投資額）はいくらくらいを想定していますか？",
                "unit": "円",
                "default": 1000000,
                "importance": 5
            },
            {
                "name": "月額運用コスト",
                "question": "導入後の月々の運用コストはいくらくらいになりそうですか？",
                "unit": "円/月",
                "default": 50000,
                "importance": 4
            },
            {
                "name": "月間効果額",
                "question": "この施策を導入することで、月々どのくらいの効果（コスト削減や増収）が見込めますか？",
                "unit": "円/月",
                "default": 150000,
                "importance": 5
            },
            {
                "name": "実装期間",
                "question": "この施策を実装するのにどのくらいの期間が必要ですか？",
                "unit": "ヶ月",
                "default": 3,
                "importance": 3
            }
        ]}}
    except Exception as e:
        print(f"JSON解析エラー: {e}")
        # エラーが発生した場合はデフォルト値を返す
        return {{"parameters": [
            {
                "name": "初期導入コスト",
                "question": "このDX施策の初期導入コスト（初期投資額）はいくらくらいを想定していますか？",
                "unit": "円",
                "default": 1000000,
                "importance": 5
            },
            {
                "name": "月額運用コスト",
                "question": "導入後の月々の運用コストはいくらくらいになりそうですか？",
                "unit": "円/月",
                "default": 50000,
                "importance": 4
            },
            {
                "name": "月間効果額",
                "question": "この施策を導入することで、月々どのくらいの効果（コスト削減や増収）が見込めますか？",
                "unit": "円/月",
                "default": 150000,
                "importance": 5
            },
            {
                "name": "実装期間",
                "question": "この施策を実装するのにどのくらいの期間が必要ですか？",
                "unit": "ヶ月",
                "default": 3,
                "importance": 3
            }
        ]}}


# 値の抽出関数（新規追加）
def extract_value_from_message(message, parameter):
    """
    ユーザーメッセージからパラメータ値を抽出する
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """ユーザーメッセージから特定のパラメータの値を抽出してください。
可能な限り正確な値と単位を抽出します。

以下のJSON形式で回答してください：
```json
{{
  "extracted": true/false,
  "value": 抽出した値（数値）,
  "unit": "抽出した単位（円、%など）",
  "confidence": 0-100の確信度
}}
```
値が見つからない場合はextracted=falseとし、valueは0、unitは空文字としてください。"""),
        ("human", f"""以下のメッセージから「{parameter['name']}」の値を抽出してください：

ユーザーメッセージ: {message}

単位は「{parameter['unit']}」を想定しています。""")
    ])
    
    # 正しくメッセージを生成
    messages = extraction_prompt.format_messages()
    response = llm.invoke(messages)
    
    # JSONを抽出
    try:
        # JSONブロックを探す
        pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(pattern, response.content, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        # JSON形式の部分を探す
        pattern = r'{.*}'
        match = re.search(pattern, response.content, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        
        return {"extracted": False, "value": 0, "unit": "", "confidence": 0}
    except Exception as e:
        print(f"JSON解析エラー: {e}")
        return {"extracted": False, "value": 0, "unit": "", "confidence": 0}


def main():
    """Improved Streamlit application for ROI analysis"""
    # アプリのタイトルとヘッダー
    st.set_page_config(
        page_title="ROI分析ツール",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 ROIツリー分析と提案ソリューション")
    st.markdown("""
    このツールは、ビジネス課題をROIの観点から構造化し、分析・提案するためのツールです。
    テキストボックスにビジネス課題を入力して「分析開始」ボタンをクリックしてください。
    """)
    
    # サイドバーの設定
    with st.sidebar:
        st.title("設定")
        model_name = st.selectbox(
            "使用するモデル",
            ["gpt-4o", "gpt-3.5-turbo","o1","o3-mini"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 分析例")
        st.markdown("""
        例: ゴム製品メーカーでDX推進を担当しています。製造ラインの自動化により生産効率25%向上を目指し、年間1億円のコスト削減を狙っていますが、初期投資が大きいです。また、品質管理の自動化でクレームを減らして売上を伸ばしたいのですが、現場がAIに不安を抱えています。教育コストもかかりそうです。
        """)
        
        if st.button("サンプル課題を入力"):
            st.session_state.challenge_text = """ゴム製品メーカーでDX推進を担当しています。製造ラインの自動化により生産効率25%向上を目指し、年間1億円のコスト削減を狙っていますが、初期投資が大きいです。また、品質管理の自動化でクレームを減らして売上を伸ばしたいのですが、現場がAIに不安を抱えています。教育コストもかかりそうです。"""
    
    # タブで「課題分析」と「提案生成」を分ける
    tab1, tab2, tab3 = st.tabs(["1. 課題分析", "2. 提案生成", "3. ROI計算"])
    
    # タブ1: 課題分析
    with tab1:
        st.header("課題分析")
        st.markdown("ビジネス課題をROIツリーとして構造化します。各末端ノードは具体的なDXツールで対応できる粒度になります。")
        
        # テキスト入力欄の設定（セッション状態を使用）
        if "challenge_text" not in st.session_state:
            st.session_state.challenge_text = ""
        
        # セッション状態の初期化
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "mermaid_history" not in st.session_state:
            st.session_state.mermaid_history = []
        
        if "chat_active" not in st.session_state:
            st.session_state.chat_active = False
        
        if "roi_chat_agent" not in st.session_state:
            st.session_state.roi_chat_agent = None
        
        # 追加キーの初期化
        if "mermaid_diagram" not in st.session_state:
            st.session_state.mermaid_diagram = None

        if "root_node" not in st.session_state:
            st.session_state.root_node = None

        if "challenge_agent" not in st.session_state:
            st.session_state.challenge_agent = None

        if "analysis" not in st.session_state:
            st.session_state.analysis = None

        if "solution_suggestions" not in st.session_state:
            st.session_state.solution_suggestions = None
        
        # 新しい初期化: 選択された末端ノード
        if "selected_leaf_nodes" not in st.session_state:
            st.session_state.selected_leaf_nodes = {}
            
        # 新しい初期化: 末端ノードのパラメータ
        if "node_parameters" not in st.session_state:
            st.session_state.node_parameters = {}
        
        # 新しい初期化: チャットモード関連
        if "chat_mode" not in st.session_state:
            st.session_state.chat_mode = "inactive"  # inactive, node_selection, parameter_collection
        
        if "current_node_id" not in st.session_state:
            st.session_state.current_node_id = None
            
        if "current_node_label" not in st.session_state:
            st.session_state.current_node_label = None
            
        if "required_parameters" not in st.session_state:
            st.session_state.required_parameters = []
            
        if "current_parameter_index" not in st.session_state:
            st.session_state.current_parameter_index = 0
            
        if "collected_values" not in st.session_state:
            st.session_state.collected_values = {}
        
        if "processing_queue" not in st.session_state:
            st.session_state.processing_queue = []
            
        # 新しい初期化: 選択された親ノード（子ノード再生成用）
        if "selected_parent_node" not in st.session_state:
            st.session_state.selected_parent_node = None
                
        challenge_text = st.text_area(
            "ビジネス課題を入力してください",
            value=st.session_state.challenge_text,
            height=150,
            placeholder="例: ゴム製品メーカーでDX推進を担当しています。製造ラインの自動化により..."
        )
        

        # 分析実行ボタン
        if st.button("分析開始", type="primary", disabled=not challenge_text):
            with st.spinner("ROIツリーを生成中..."):
                try:
                    # デバッグ用: 入力されたテキストを確認
                    print(f"入力されたビジネス課題: {challenge_text}")
                    print(f"長さ: {len(challenge_text)} 文字")
                    
                    # ChallengeAgentを初期化して実行
                    agent = ChallengeAgent(model_name=model_name)
                    
                    # テキストが空でないことを確認
                    if not challenge_text or len(challenge_text.strip()) < 10:
                        st.error("ビジネス課題の入力が不足しています。より詳細な課題を入力してください。")
                        return
                        
                    # 課題テキストを明示的に渡す
                    root_node, mermaid_diagram = agent.create_roi_tree(challenge_text.strip())
                    
                    # 結果を保存 - デフォルトには戻さず、生成されたものをそのまま使用
                    st.session_state.mermaid_diagram = mermaid_diagram
                    st.session_state.root_node = root_node
                    st.session_state.challenge_agent = agent
                    st.session_state.challenge_text = challenge_text
                    
                    # ROIチャットエージェントを初期化
                    st.session_state.roi_chat_agent = ROIChatAgent(model_name=model_name)
                    
                    # 末端ノードの分析 - 生成されたツリーを使用
                    analysis = agent.analyze_leaf_nodes(root_node, mermaid_diagram)
                    st.session_state.analysis = analysis
                    
                    # ソリューションの提案も準備 - 生成されたツリーを使用
                    solution_suggestions = agent.suggest_solutions(root_node, mermaid_diagram)
                    st.session_state.solution_suggestions = solution_suggestions
                    
                    # チャット履歴をリセット
                    st.session_state.chat_history = []
                    st.session_state.mermaid_history = [mermaid_diagram]
                    st.session_state.chat_active = False
                    
                    # 選択された末端ノードとパラメータもリセット
                    st.session_state.selected_leaf_nodes = {}
                    st.session_state.node_parameters = {}
                    
                    # チャットモード関連の初期化
                    st.session_state.chat_mode = "inactive"
                    st.session_state.current_node_id = None
                    st.session_state.current_node_label = None
                    st.session_state.required_parameters = []
                    st.session_state.current_parameter_index = 0
                    st.session_state.collected_values = {}
                    st.session_state.processing_queue = []
                except Exception as e:
                    st.error(f"ROIツリーの生成中にエラーが発生しました: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if "mermaid_diagram" in st.session_state and st.session_state.mermaid_diagram:
            st.markdown("---")
            
            # ツリーと分析結果を表示するためのカラムを作成
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("ROIツリー図")
                # 最新のMermaidダイアグラムを表示
                if st.session_state.mermaid_history:
                    render_mermaid(st.session_state.mermaid_history[-1])
                else:
                    render_mermaid(st.session_state.mermaid_diagram)
                
                # ROIツリーの再生成ボタン
                if st.button("ROIツリーを再生成", help="現在のツリーを参考にして新しいROIツリーを生成します"):
                    with st.spinner("ROIツリーを再生成中..."):
                        try:
                            current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                            agent = st.session_state.challenge_agent
                            
                            # deepdive_agent.pyのCHALLENGE_SYSTEM_PROMPTを参照する再生成関数を使用
                            new_root_node, new_mermaid_diagram = regenerate_roi_tree(
                                agent, 
                                st.session_state.challenge_text, 
                                current_mermaid
                            )
                            
                            # 結果を保存
                            st.session_state.root_node = new_root_node
                            st.session_state.mermaid_history.append(new_mermaid_diagram)
                            
                            # 末端ノードの分析を更新
                            st.session_state.analysis = agent.analyze_leaf_nodes(new_root_node, new_mermaid_diagram)
                            
                            # ソリューションの提案も更新
                            st.session_state.solution_suggestions = agent.suggest_solutions(new_root_node, new_mermaid_diagram)
                            
                            # 選択された末端ノードをリセット
                            st.session_state.selected_leaf_nodes = {}
                            
                            # 再読み込み
                            st.rerun()
                        except Exception as e:
                            st.error(f"ROIツリーの再生成中にエラーが発生しました: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                # 子ノード再生成セクション
                st.subheader("特定ノードの子ノードを再生成")
                
                # 親ノード（非末端ノード）のリストを取得
                current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                parent_nodes = extract_non_leaf_nodes(current_mermaid)
                
                if parent_nodes:
                    # 親ノードの選択ドロップダウン
                    parent_options = [f"{node_id}: {node_label}" for node_id, node_label in parent_nodes]
                    selected_parent = st.selectbox(
                        "子ノードを再生成する親ノードを選択",
                        options=parent_options,
                        index=0 if parent_options else None
                    )
                    
                    if selected_parent and st.button("選択したノードの子ノードを再生成", help="選択した親ノードの子ノードを再生成します"):
                        with st.spinner("子ノードを再生成中..."):
                            try:
                                # 選択された親ノードIDを取得
                                parent_node_id = selected_parent.split(":")[0].strip()
                                agent = st.session_state.challenge_agent
                                
                                # deepdive_agent.pyのCHALLENGE_SYSTEM_PROMPTを参照する子ノード再生成関数を使用
                                updated_mermaid = regenerate_child_nodes(
                                    agent,
                                    st.session_state.challenge_text,
                                    current_mermaid,
                                    parent_node_id
                                )
                                
                                # 更新されたMermaidダイアグラムをROIツリーに変換
                                updated_root_node = mermaid_to_roi_tree(updated_mermaid)
                                
                                # 結果を保存
                                st.session_state.root_node = updated_root_node
                                st.session_state.mermaid_history.append(updated_mermaid)
                                
                                # 末端ノードの分析を更新
                                st.session_state.analysis = agent.analyze_leaf_nodes(updated_root_node, updated_mermaid)
                                
                                # ソリューションの提案も更新
                                st.session_state.solution_suggestions = agent.suggest_solutions(updated_root_node, updated_mermaid)
                                
                                # 選択された末端ノードをリセット
                                st.session_state.selected_leaf_nodes = {}
                                
                                # 再読み込み
                                st.rerun()
                            except Exception as e:
                                st.error(f"子ノードの再生成中にエラーが発生しました: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                else:
                    st.info("親ノード（子ノードを持つノード）が見つかりません。")
 
                
                # コードとしても表示
                with st.expander("Mermaidコード"):
                    if st.session_state.mermaid_history:
                        st.code(st.session_state.mermaid_history[-1], language="text")
                    else:
                        st.code(st.session_state.mermaid_diagram, language="text")
            
            with col2:
                # 分析データがあるかチェック
                if "analysis" in st.session_state and st.session_state.analysis is not None:
                    analysis = st.session_state.analysis
                    
                    st.subheader("末端ノード分析")
                    st.info(f"**完成度**: {analysis.completion_percentage:.1f}%")
                    
                    if analysis.has_numerical_data:
                        st.success("✅ すべての末端ノードには数値目標が設定されています")
                    else:
                        st.warning("⚠️ 一部の末端ノードに数値目標が設定されていません")
                    
                    if analysis.missing_nodes:
                        st.subheader("数値目標がないノード")
                        for node in analysis.missing_nodes:
                            st.markdown(f"- {node}")
                    
                    if analysis.incomplete_nodes:
                        st.subheader("改善が必要なノード")
                        for node in analysis.incomplete_nodes:
                            st.markdown(f"- **{node['name']}**: {node['issue']}")
                            st.markdown(f"  提案: {node['suggestion']}")
                    
                    # 粒度の問題があるか確認
                    if hasattr(analysis, 'granularity_issues') and analysis.granularity_issues:
                        st.subheader("粒度の問題があるノード")
                        for node in analysis.granularity_issues:
                            st.markdown(f"- **{node['name']}**: {node['issue']}")
                            st.markdown(f"  提案: {node['suggestion']}")
                            if 'dx_tools' in node and node['dx_tools']:
                                st.markdown(f"  対応可能なDXツール: {', '.join(node['dx_tools'])}")
                    
                    # 分岐の問題があるか確認
                    if hasattr(analysis, 'branching_issues') and analysis.branching_issues:
                        st.subheader("分岐の問題があるカテゴリ")
                        for node in analysis.branching_issues:
                            st.markdown(f"- **{node['name']}**: {node['issue']}")
                            st.markdown(f"  現在の分岐数: {node['current_branches']}")
                            st.markdown(f"  提案: {node['suggestion']}")
                    
                    # 数値サマリーを表示
                    st.subheader("数値サマリー")
                    
                    # 末端ノードの値を抽出する関数
                    def get_node_values(mermaid_diagram):
                        leaf_nodes = extract_leaf_nodes(mermaid_diagram)
                        nodes_with_values = []
                        
                        for node_id, node_label in leaf_nodes:
                            # 金額のパターン（例: 1億円, 500万円）
                            amount_pattern = r'(\d+[,\.]?\d*\s*[億万千]?円)'
                            amount_match = re.search(amount_pattern, node_label)
                            
                            # パーセンテージのパターン
                            percentage_pattern = r'(\d+[,\.]?\d*\s*%)'
                            percentage_match = re.search(percentage_pattern, node_label)
                            
                            # 一般的な数値パターン
                            general_pattern = r'(\d+[,\.]?\d*\s*[件人時分秒])'
                            general_match = re.search(general_pattern, node_label)
                            
                            value = None
                            if amount_match:
                                value = amount_match.group(1)
                            elif percentage_match:
                                value = percentage_match.group(1)
                            elif general_match:
                                value = general_match.group(1)
                                
                            if value:
                                nodes_with_values.append((node_id, node_label, value))
                        
                        return nodes_with_values
                    
                    # 現在の末端ノードの値を表示
                    current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                    nodes_with_values = get_node_values(current_mermaid)
                    
                    if nodes_with_values:
                        items = []
                        for _, node_label, value in nodes_with_values:
                            # ノード名からカッコの部分を削除
                            node_name = re.sub(r'\s*\([^)]+\)', '', node_label)
                            items.append({"ノード": node_name, "値": value})
                        
                        st.dataframe(pd.DataFrame(items), use_container_width=True)
                    else:
                        st.info("数値が設定されたノードがありません")
                else:
                    # 分析データがない場合の表示
                    st.warning("分析データがまだ生成されていません。分析中にエラーが発生した可能性があります。")
                    st.info("「分析開始」ボタンをクリックして再度分析を実行してください。")
            
            st.markdown("---")
            st.subheader("優先ノード選択とROI計算")
            
            # 現在のMermaidダイアグラムからすべての末端ノードを取得
            current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
            leaf_nodes = extract_leaf_nodes(current_mermaid)
            
            # チェックボックスUIによるノード選択
            st.markdown("### 対応するノードを選択")
            
            # 末端ノードの選択UIの構築
            selected_nodes = []
            
            for node_id, node_label in leaf_nodes:
                # ノード名から括弧の部分を削除して表示を簡潔に
                clean_label = re.sub(r'\s*\([^)]+\)', '', node_label)
                
                # セッションステートの初期化
                if node_id not in st.session_state.selected_leaf_nodes:
                    st.session_state.selected_leaf_nodes[node_id] = False
                
                # チェックボックスの作成
                is_selected = st.checkbox(
                    clean_label, 
                    value=st.session_state.selected_leaf_nodes[node_id],
                    key=f"node_{node_id}"
                )
                
                # 選択状態を保存
                st.session_state.selected_leaf_nodes[node_id] = is_selected
                
                # 選択されたノードを記録
                if is_selected:
                    selected_nodes.append((node_id, node_label, clean_label))
            
            # パラメータ収集のためのチャットインターフェースを開始
            if selected_nodes:
                if st.button("選択したノードのROI計算を開始", type="primary"):
                    # チャットモードをアクティブに設定
                    st.session_state.chat_mode = "node_preparation"
                    
                    # 処理キューに選択されたノードを追加
                    st.session_state.processing_queue = selected_nodes
                    
                    # 最初のノードの処理を開始
                    if st.session_state.processing_queue:
                        next_node = st.session_state.processing_queue[0]
                        st.session_state.current_node_id = next_node[0]
                        st.session_state.current_node_label = next_node[2]  # clean_label
                        
                        # 必要なパラメータを判断
                        params_result = determine_required_parameters(next_node[2], st.session_state.challenge_text)
                        st.session_state.required_parameters = params_result.get("parameters", [])
                        st.session_state.current_parameter_index = 0
                        st.session_state.collected_values = {}
                        
                        # チャット履歴を初期化して、最初のメッセージを追加
                        st.session_state.chat_history = []
                        
                        # 改善: より会話的な初期メッセージ
                        business_context = st.session_state.challenge_text
                        node_name = next_node[2]
                        
                        # より自然な対話のための初期メッセージを作成
                        initial_message = f"""「{node_name}」についてROI計算を行います。

この施策は「{business_context}」という課題に対応するものですね。
まず、簡単に「{node_name}」でどのようなDX施策を考えているか教えていただけますか？
具体的にどのような課題を解決したいのか、どんなソリューションを想定されているかを簡単に教えてください。"""
                        
                        st.session_state.chat_history.append(("assistant", initial_message))
                        st.session_state.chat_mode = "node_introduction"
                    
                    # 再ロードして状態を更新
                    st.rerun()
            
            # チャットインターフェースの表示
            if st.session_state.chat_mode != "inactive":
                st.markdown("---")
                st.subheader(f"「{st.session_state.current_node_label}」のROI計算")
                
                # チャット履歴の表示
                chat_container = st.container()
                with chat_container:
                    for role, text in st.session_state.chat_history:
                        if role == "user":
                            st.chat_message("user").write(text)
                        else:
                            st.chat_message("assistant").write(text)
                
                # チャット入力
                user_input = st.chat_input("メッセージを入力...")
                
                if user_input:
                    # ユーザー入力を表示・保存
                    st.chat_message("user").write(user_input)
                    st.session_state.chat_history.append(("user", user_input))
                    
                    # チャットモードに応じた処理
                    if st.session_state.chat_mode == "node_introduction":
                        # 改善: ユーザーの回答を要約して確認する
                        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
                        
                        summary_prompt = ChatPromptTemplate.from_messages([
                            ("system", """ユーザーの回答を簡潔に要約し、理解したことを確認するメッセージを作成してください。
その後、必要なパラメータ情報の収集に進んでください。会話的で自然な応答を心がけてください。"""),
                            ("human", f"""ノード: {st.session_state.current_node_label}
ユーザーの回答: {user_input}

ユーザーの回答を簡潔に要約し、次に必要なパラメータを収集する会話を作成してください。""")
                        ])
                        
                        # 応答を生成
                        summary_messages = summary_prompt.format_messages()
                        summary_response = llm.invoke(summary_messages)
                        
                        # 応答をチャット履歴に追加
                        confirmation_message = summary_response.content
                        st.chat_message("assistant").write(confirmation_message)
                        st.session_state.chat_history.append(("assistant", confirmation_message))
                        
                        # パラメータ収集モードに移行
                        st.session_state.chat_mode = "parameter_collection"
                        
                        # 再ロードして状態を更新
                        st.rerun()
                    
                    elif st.session_state.chat_mode == "parameter_collection":
                        # 現在のパラメータを取得
                        if st.session_state.current_parameter_index < len(st.session_state.required_parameters):
                            current_param = st.session_state.required_parameters[st.session_state.current_parameter_index]
                            
                            # ユーザー入力から値を抽出
                            extracted = extract_value_from_message(user_input, current_param)
                            
                            if extracted["extracted"]:
                                # 値が抽出できた場合
                                st.session_state.collected_values[current_param["name"]] = {
                                    "value": extracted["value"],
                                    "unit": extracted["unit"] or current_param["unit"]
                                }
                                
                                # 次のパラメータに進む
                                st.session_state.current_parameter_index += 1
                                
                                # まだパラメータがある場合
                                if st.session_state.current_parameter_index < len(st.session_state.required_parameters):
                                    next_param = st.session_state.required_parameters[st.session_state.current_parameter_index]
                                    
                                    # 改善: より会話的なパラメータ質問
                                    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                                    
                                    convo_prompt = ChatPromptTemplate.from_messages([
                                        ("system", """あなたはROI計算に必要なパラメータを収集するアシスタントです。
会話的で自然な方法でパラメータ情報を収集してください。直前の会話の流れを考慮し、
前のパラメータに対する回答に肯定的なフィードバックを与えた上で、次のパラメータについて質問してください。

質問は簡潔でわかりやすく、必要に応じて例を示して回答しやすくしてください。
ユーザーが回答しにくい場合は、業界平均や一般的な数値も提案すると良いでしょう。"""),
                                        ("human", f"""ノード: {st.session_state.current_node_label}
直前のユーザー回答: {user_input}
直前に収集したパラメータ: {current_param["name"]} = {extracted["value"]} {extracted["unit"] or current_param["unit"]}
次に収集するパラメータ: {next_param["name"]}
パラメータの質問文: {next_param["question"]}
単位: {next_param["unit"]}
デフォルト値: {next_param["default"]}

上記の情報を元に、会話的で自然なパラメータ収集の質問文を作成してください。""")
                                    ])
                                    
                                    # 応答を生成
                                    convo_messages = convo_prompt.format_messages()
                                    convo_response = llm.invoke(convo_messages)
                                    
                                    # 応答をチャット履歴に追加
                                    param_message = convo_response.content
                                    st.chat_message("assistant").write(param_message)
                                    st.session_state.chat_history.append(("assistant", param_message))
                                else:
                                    # すべてのパラメータが収集された場合
                                    # 改善: 収集された情報の確認と次のステップへの移行
                                    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
                                    
                                    completion_prompt = ChatPromptTemplate.from_messages([
                                        ("system", """あなたはROI計算に必要なパラメータ収集を完了したアシスタントです。
収集したパラメータを全て要約し、ユーザーに確認してから次のステップ（ROI計算）に進みます。
収集された情報を整理して、よくまとめてください。"""),
                                        ("human", f"""ノード: {st.session_state.current_node_label}

収集したパラメータ:
{json.dumps(st.session_state.collected_values, ensure_ascii=False, indent=2)}

収集した情報を要約し、ユーザーに確認を求めるメッセージを作成してください。""")
                                    ])
                                    
                                    # 応答を生成
                                    completion_messages = completion_prompt.format_messages()
                                    completion_response = llm.invoke(completion_messages)
                                    
                                    # 応答をチャット履歴に追加
                                    completion_message = completion_response.content
                                    st.chat_message("assistant").write(completion_message)
                                    st.session_state.chat_history.append(("assistant", completion_message))
                                    
                                    # ROI計算モードに移行
                                    st.session_state.chat_mode = "roi_calculation"
                            else:
                                # 値の抽出に失敗した場合、より自然な再質問
                                llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
                                
                                retry_prompt = ChatPromptTemplate.from_messages([
                                    ("system", """あなたはROI計算に必要なパラメータを収集するアシスタントです。
ユーザーの回答から必要な情報を抽出できなかった場合に、より明確に質問を再度行います。
ユーザーが回答しやすいよう、具体例や選択肢を提示するようにしてください。"""),
                                    ("human", f"""ノード: {st.session_state.current_node_label}
抽出に失敗したパラメータ: {current_param["name"]}
パラメータの質問文: {current_param["question"]}
単位: {current_param["unit"]}
デフォルト値: {current_param["default"]}
ユーザーの回答: {user_input}

上記のパラメータについて再度質問してください。ユーザーが回答しやすいよう、例や選択肢を提示してください。""")
                                ])
                                
                                # 応答を生成
                                retry_messages = retry_prompt.format_messages()
                                retry_response = llm.invoke(retry_messages)
                                
                                # 応答をチャット履歴に追加
                                retry_message = retry_response.content
                                st.chat_message("assistant").write(retry_message)
                                st.session_state.chat_history.append(("assistant", retry_message))
                            
                            # 再ロードして状態を更新
                            st.rerun()
                    
                    elif st.session_state.chat_mode == "roi_calculation":
                        # パラメータから必要な値を取得
                        initial_cost = 0
                        monthly_cost = 0
                        monthly_benefit = 0
                        implementation_months = 3  # デフォルト値
                        
                        for param in st.session_state.required_parameters:
                            param_name = param["name"]
                            if param_name in st.session_state.collected_values:
                                value = st.session_state.collected_values[param_name]["value"]
                                
                                if "初期" in param_name or "導入" in param_name:
                                    initial_cost = value
                                elif "月額" in param_name or "運用" in param_name:
                                    monthly_cost = value
                                elif "効果" in param_name or "削減" in param_name or "増加" in param_name:
                                    # 年間効果を月額に変換
                                    if "年間" in param_name:
                                        monthly_benefit = value / 12
                                    else:
                                        monthly_benefit = value
                                elif "期間" in param_name or "月数" in param_name:
                                    implementation_months = value
                        
                        # 年間のコストと効果を計算
                        annual_cost = initial_cost + (monthly_cost * 12)
                        annual_benefit = monthly_benefit * 12
                        
                        # ROI計算（投資対効果）
                        if annual_cost > 0:
                            roi_percentage = ((annual_benefit - annual_cost) / annual_cost) * 100
                        else:
                            roi_percentage = 0
                        
                        # 投資回収期間を計算
                        if monthly_benefit > monthly_cost:
                            payback_months = initial_cost / (monthly_benefit - monthly_cost)
                            if payback_months > 36:  # 3年以上なら「長期」と表示
                                payback_period = "3年以上（長期）"
                            else:
                                years = int(payback_months / 12)
                                months = int(payback_months % 12)
                                payback_period = f"{years}年{months}ヶ月"
                        else:
                            payback_period = "効果が運用コストを下回るため回収不能"
                        
                        # 改善: より洞察のあるROI計算結果の表示
                        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                        
                        roi_prompt = ChatPromptTemplate.from_messages([
                            ("system", """あなたはROI計算の専門家です。計算結果を分かりやすく解説し、
ビジネス的な洞察も加えて説明してください。数値は正確に表示し、簡潔でありながらも、
重要なポイントや示唆を提供するようにしてください。"""),
                            ("human", f"""ノード: {st.session_state.current_node_label}

計算結果:
- ROI: {roi_percentage:.1f}%
- 初期投資: {initial_cost:,.0f}円
- 年間運用コスト: {monthly_cost * 12:,.0f}円
- 年間効果: {annual_benefit:,.0f}円
- 投資回収期間: {payback_period}

上記の計算結果について、分かりやすく解説し、ビジネス的な洞察も加えてください。
また、この結果をROIツリーに反映したことも伝えてください。""")
                        ])
                        
                        # 応答を生成
                        roi_messages = roi_prompt.format_messages()
                        roi_response = llm.invoke(roi_messages)
                        
                        # 応答をチャット履歴に追加
                        roi_result = roi_response.content
                        st.chat_message("assistant").write(roi_result)
                        st.session_state.chat_history.append(("assistant", roi_result))
                        
                        # Mermaidダイアグラムの更新
                        current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                        updated_mermaid = update_node_value_with_roi_data(
                            current_mermaid,
                            st.session_state.current_node_id,
                            "ROI計算済み",
                            cost=annual_cost,
                            benefit=annual_benefit,
                            implementation_period=f"{implementation_months}ヶ月"
                        )
                        
                        # 更新されたダイアグラムを履歴に追加
                        st.session_state.mermaid_history.append(updated_mermaid)
                        
                        # 処理キューから現在のノードを削除
                        st.session_state.processing_queue.pop(0)
                        
                        # 次のノードがあれば処理を続ける
                        if st.session_state.processing_queue:
                            next_node = st.session_state.processing_queue[0]
                            
                            # 改善: 次のノードへの自然な移行
                            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                            
                            transition_prompt = ChatPromptTemplate.from_messages([
                                ("system", """あなたはROI計算の専門家です。現在のノードのROI計算が完了し、
次のノードのROI計算に移行する際のメッセージを作成してください。
自然な流れで次のノードに移行するメッセージを作成してください。"""),
                                ("human", f"""現在完了したノード: {st.session_state.current_node_label}
次に計算するノード: {next_node[2]}

次のノードの計算に移行するメッセージを作成してください。質問は「どのようなDX施策を想定しているか」などの
オープンな質問から始めるようにしてください。""")
                            ])
                            
                            # 応答を生成
                            transition_messages = transition_prompt.format_messages()
                            transition_response = llm.invoke(transition_messages)
                            
                            # 応答をチャット履歴に追加
                            next_node_message = transition_response.content
                            st.chat_message("assistant").write(next_node_message)
                            st.session_state.chat_history.append(("assistant", next_node_message))
                            
                            # 次のノードの処理を開始
                            st.session_state.current_node_id = next_node[0]
                            st.session_state.current_node_label = next_node[2]
                            
                            # 必要なパラメータを判断
                            params_result = determine_required_parameters(next_node[2], st.session_state.challenge_text)
                            st.session_state.required_parameters = params_result.get("parameters", [])
                            st.session_state.current_parameter_index = 0
                            st.session_state.collected_values = {}
                            
                            st.session_state.chat_mode = "node_introduction"
                        else:
                            # すべてのノードの処理が完了
                            # 改善: 全体のまとめと次のステップの提案
                            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
                            
                            completion_prompt = ChatPromptTemplate.from_messages([
                                ("system", """あなたはROI計算の専門家です。すべてのノードのROI計算が完了した際に、
全体のまとめと次のステップについて提案するメッセージを作成してください。
具体的な成果と次のアクションを簡潔に説明してください。"""),
                                ("human", f"""すべてのノードのROI計算が完了しました。

全体のまとめと、「提案生成」タブで具体的な提案を作成できることを案内するメッセージを作成してください。""")
                            ])
                            
                            # 応答を生成
                            completion_messages = completion_prompt.format_messages()
                            completion_response = llm.invoke(completion_messages)
                            
                            # 応答をチャット履歴に追加
                            completion_message = completion_response.content
                            st.chat_message("assistant").write(completion_message)
                            st.session_state.chat_history.append(("assistant", completion_message))
                            
                            # チャットモードを非アクティブに
                            st.session_state.chat_mode = "inactive"
                    
                    # 再ロードして状態を更新
                    st.rerun()
            
            # すでにROI計算済みのノードの一覧表示
            roi_calculated_nodes = []
            
            for node_id, node_label in leaf_nodes:
                if "ROI:" in node_label or "コスト:" in node_label or "効果:" in node_label:
                    # ノード名から括弧の部分を削除
                    clean_label = re.sub(r'\s*\([^)]+\)', '', node_label)
                    
                    # ROI情報を抽出
                    roi_match = re.search(r'ROI:(\d+\.?\d*)%', node_label)
                    cost_match = re.search(r'コスト:(\d+\.?\d*)(億円|万円|円)', node_label)
                    benefit_match = re.search(r'効果:(\d+\.?\d*)(億円|万円|円)', node_label)
                    
                    roi_value = float(roi_match.group(1)) if roi_match else 0
                    
                    cost_value = 0
                    if cost_match:
                        cost_num = float(cost_match.group(1))
                        cost_unit = cost_match.group(2)
                        if cost_unit == "億円":
                            cost_value = cost_num * 100000000
                        elif cost_unit == "万円":
                            cost_value = cost_num * 10000
                        else:
                            cost_value = cost_num
                    
                    benefit_value = 0
                    if benefit_match:
                        benefit_num = float(benefit_match.group(1))
                        benefit_unit = benefit_match.group(2)
                        if benefit_unit == "億円":
                            benefit_value = benefit_num * 100000000
                        elif benefit_unit == "万円":
                            benefit_value = benefit_num * 10000
                        else:
                            benefit_value = benefit_num
                    
                    roi_calculated_nodes.append({
                        "node_id": node_id,
                        "node_label": clean_label,
                        "roi": roi_value,
                        "cost": cost_value,
                        "benefit": benefit_value
                    })
            
            if roi_calculated_nodes:
                st.markdown("---")
                st.markdown("### ROI計算済みノード一覧")
                
                # ROI計算済みノードを表示
                roi_df = pd.DataFrame([
                    {
                        "ノード": node["node_label"],
                        "ROI": node["roi"],
                        "コスト": node["cost"],
                        "効果": node["benefit"]
                    } for node in roi_calculated_nodes
                ])
                
                # ROIの降順でソート
                roi_df = roi_df.sort_values("ROI", ascending=False)
                
                # 表示
                st.dataframe(roi_df.style.format({
                    "ROI": "{:.1f}%",
                    "コスト": "{:,.0f}円",
                    "効果": "{:,.0f}円"
                }), use_container_width=True)
                
                # ROIが高いノードの優先提案
                st.markdown("### 優先すべきノード")
                
                for i, row in roi_df.head(3).iterrows():
                    st.markdown(f"{i+1}. **{row['ノード']}** (ROI: {row['ROI']:.1f}%)")
    
    # タブ2と3は前のコードとほぼ同じなので省略（必要に応じて追加）
    
    
    # タブ2: 提案生成の修正部分
    with tab2:
        st.header("提案生成")
        st.markdown("課題分析に基づいて、特定の末端ノードに対して具体的な提案を生成します。")
        
        # 課題ツリーが生成されているか確認
        if "mermaid_diagram" not in st.session_state and not st.session_state.mermaid_history:
            st.warning("最初に「課題分析」タブで課題ツリーを生成してください。")
        else:
            # 現在のROIツリーを取得（更新履歴がある場合は最新のものを使用）
            current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
            
            # ROI計算済みのノードがあるか確認
            leaf_nodes = extract_leaf_nodes(current_mermaid)
            roi_calculated_nodes = []
            
            for node_id, node_label in leaf_nodes:
                if "ROI:" in node_label or "コスト:" in node_label or "効果:" in node_label:
                    # ROI情報が含まれるノードを追加
                    roi_calculated_nodes.append((node_id, node_label))
            
            if not roi_calculated_nodes:
                st.warning("ROI計算が完了したノードがありません。「課題分析」タブでノードのROI計算を行ってください。")
            else:
                # セッション状態の初期化
                if "selected_node" not in st.session_state:
                    st.session_state.selected_node = None
                if "node_selection_complete" not in st.session_state:
                    st.session_state.node_selection_complete = False
                if "proposal_text_input" not in st.session_state:
                    st.session_state.proposal_text_input = ""
                
                # ステップ1: ノード選択セクション
                st.subheader("ステップ1: ROI計算済みノードの選択")
                
                # ROI計算済みノードのドロップダウン
                node_options = [label for _, label in roi_calculated_nodes]
                node_ids = [node_id for node_id, _ in roi_calculated_nodes]
                
                # 選択セクション
                selected_index = 0
                if st.session_state.selected_node in node_options:
                    selected_index = node_options.index(st.session_state.selected_node)
                    
                selected_node = st.selectbox(
                    "提案を生成する対象ノードを選択してください",
                    node_options,
                    index=selected_index,
                    help="ROI計算が完了したノードを選択すると、そのノードに対する提案が生成されます。"
                )
                
                if st.button("ノードを確定", disabled=st.session_state.node_selection_complete):
                    st.session_state.selected_node = selected_node
                    st.session_state.node_selection_complete = True
                    
                    # 対応するノードIDを検索
                    selected_node_id = None
                    for i, (node_id, label) in enumerate(roi_calculated_nodes):
                        if label == selected_node:
                            selected_node_id = node_id
                            break
                    
                    st.session_state.selected_node_id = selected_node_id
                    st.rerun()
                
                # ノード選択の表示
                if st.session_state.node_selection_complete and st.session_state.selected_node:
                    st.success(f"選択されたノード: {st.session_state.selected_node}")
                    
                    # 選択されたノード用のMermaidコード生成
                    selected_node_mermaid = "flowchart TD\n"
                    for node_id, label in roi_calculated_nodes:
                        if label == st.session_state.selected_node:
                            # 選択されたノードを強調表示
                            selected_node_mermaid += f'    {node_id}["{label}"]:::selected\n'
                            break
                    
                    selected_node_mermaid += "    classDef selected fill:#f9a826,stroke:#333,stroke-width:2px;"
                    
                    # Mermaidダイアグラム表示
                    render_mermaid(selected_node_mermaid, height=150)
                    
                    # ステップ2: 提案内容の直接入力
                    st.subheader("ステップ2: 提案内容の入力")
                    
                    # 提案種類のドロップダウン
                    proposal_types = [
                        "コスト効率重視",
                        "段階的導入",
                        "高い効果を優先",
                        "導入しやすさ優先",
                        "人材活用重視",
                        "リスク軽減",
                        "カスタム提案"
                    ]
                    
                    if "selected_proposal_type" not in st.session_state:
                        st.session_state.selected_proposal_type = proposal_types[0]
                    
                    selected_proposal_type = st.selectbox(
                        "提案の種類",
                        options=proposal_types,
                        index=proposal_types.index(st.session_state.selected_proposal_type)
                    )
                    
                    st.session_state.selected_proposal_type = selected_proposal_type
                    
                    # 提案のテンプレート表示
                    proposal_templates = {
                        "コスト効率重視": "初期投資を抑えながら効果を最大化する施策に焦点を当てます。",
                        "段階的導入": "リスクを分散し、段階的に機能を拡張していく導入アプローチを提案します。",
                        "高い効果を優先": "投資対効果が高い施策を優先的に実施し、早期に成果を出すことを重視します。",
                        "導入しやすさ優先": "現場の抵抗を最小限に抑え、スムーズな導入を実現する方法を提案します。",
                        "人材活用重視": "既存スタッフのスキルアップと人材育成を含めた総合的なアプローチを提案します。",
                        "リスク軽減": "失敗リスクを最小化するための慎重なアプローチを重視します。",
                        "カスタム提案": "以下に独自の提案内容を記入してください。"
                    }
                    
                    st.info(proposal_templates[selected_proposal_type])
                    
                    # 提案内容のテキスト入力
                    proposal_text = st.text_area(
                        "提案内容を入力してください",
                        value=st.session_state.proposal_text_input,
                        height=200,
                        placeholder="具体的な提案内容や実施方法、注意点などを記載してください。"
                    )
                    
                    st.session_state.proposal_text_input = proposal_text
                    
                    # ステップ3: ROI計算
                    st.subheader("ステップ3: 提案のROI計算")
                    
                    # ROI計算用のパラメータ入力
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if "proposal_initial_cost" not in st.session_state:
                            st.session_state.proposal_initial_cost = 1000000
                            
                        proposal_initial_cost = st.number_input(
                            "初期導入コスト（円）",
                            min_value=0,
                            value=st.session_state.proposal_initial_cost,
                            step=100000,
                            format="%d"
                        )
                        
                        st.session_state.proposal_initial_cost = proposal_initial_cost
                        
                        if "proposal_monthly_cost" not in st.session_state:
                            st.session_state.proposal_monthly_cost = 50000
                            
                        proposal_monthly_cost = st.number_input(
                            "月額運用コスト（円/月）",
                            min_value=0,
                            value=st.session_state.proposal_monthly_cost,
                            step=10000,
                            format="%d"
                        )
                        
                        st.session_state.proposal_monthly_cost = proposal_monthly_cost
                    
                    with col2:
                        if "proposal_monthly_benefit" not in st.session_state:
                            st.session_state.proposal_monthly_benefit = 200000
                            
                        proposal_monthly_benefit = st.number_input(
                            "月間効果額（円/月）",
                            min_value=0,
                            value=st.session_state.proposal_monthly_benefit,
                            step=50000,
                            format="%d"
                        )
                        
                        st.session_state.proposal_monthly_benefit = proposal_monthly_benefit
                        
                        if "proposal_implementation_months" not in st.session_state:
                            st.session_state.proposal_implementation_months = 3
                            
                        proposal_implementation_months = st.slider(
                            "導入期間（月）",
                            min_value=1,
                            max_value=12,
                            value=st.session_state.proposal_implementation_months
                        )
                        
                        st.session_state.proposal_implementation_months = proposal_implementation_months
                    
                    # ROI計算ボタン
                    if st.button("提案のROI計算", type="primary"):
                        with st.spinner("ROIを計算中..."):
                            # 初年度の総コスト計算
                            total_cost = proposal_initial_cost + (proposal_monthly_cost * 12)
                            
                            # 効果の計算（実装後の月数分）
                            effective_months = max(0, 12 - proposal_implementation_months)
                            first_year_benefit = proposal_monthly_benefit * effective_months
                            annual_benefit = proposal_monthly_benefit * 12
                            
                            # ROI計算
                            if total_cost > 0:
                                first_year_roi = ((first_year_benefit - total_cost) / total_cost) * 100
                                annual_roi = ((annual_benefit - (proposal_monthly_cost * 12)) / total_cost) * 100
                            else:
                                first_year_roi = 0
                                annual_roi = 0
                            
                            # 投資回収期間
                            if proposal_monthly_benefit > proposal_monthly_cost:
                                monthly_net_benefit = proposal_monthly_benefit - proposal_monthly_cost
                                payback_months = proposal_initial_cost / monthly_net_benefit if monthly_net_benefit > 0 else float('inf')
                                payback_years = payback_months / 12
                            else:
                                payback_months = float('inf')
                                payback_years = float('inf')
                            
                            # 結果の保存
                            st.session_state.proposal_results = {
                                "total_cost": total_cost,
                                "first_year_benefit": first_year_benefit,
                                "annual_benefit": annual_benefit,
                                "first_year_roi": first_year_roi,
                                "annual_roi": annual_roi,
                                "payback_months": payback_months,
                                "payback_years": payback_years
                            }
                    
                    # 計算結果の表示
                    if "proposal_results" in st.session_state:
                        st.markdown("### ROI計算結果")
                        
                        results = st.session_state.proposal_results
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("初年度ROI", f"{results['first_year_roi']:.1f}%")
                        
                        with col2:
                            st.metric("通年ROI", f"{results['annual_roi']:.1f}%")
                        
                        with col3:
                            if results['payback_years'] < float('inf'):
                                payback_years = int(results['payback_years'])
                                payback_months = int((results['payback_years'] - payback_years) * 12)
                                st.metric("投資回収期間", f"{payback_years}年{payback_months}ヶ月")
                            else:
                                st.metric("投資回収期間", "回収不能")
                        
                        st.markdown("### 収支詳細")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### コスト")
                            st.markdown(f"- 初期投資: {proposal_initial_cost:,.0f}円")
                            st.markdown(f"- 年間運用コスト: {proposal_monthly_cost * 12:,.0f}円")
                            st.markdown(f"- 初年度総コスト: {results['total_cost']:,.0f}円")
                        
                        with col2:
                            st.markdown("#### 効果")
                            st.markdown(f"- 月間効果: {proposal_monthly_benefit:,.0f}円/月")
                            st.markdown(f"- 初年度効果: {results['first_year_benefit']:,.0f}円（導入期間を考慮）")
                            st.markdown(f"- 通年効果: {results['annual_benefit']:,.0f}円（フル稼働時）")
                        
                        # 提案とROI結果のエクスポート機能（CSVダウンロード）
                        export_data = {
                            "項目": [
                                "選択ノード",
                                "提案種類",
                                "提案内容",
                                "初期導入コスト",
                                "月額運用コスト",
                                "月間効果額",
                                "導入期間",
                                "初年度ROI",
                                "通年ROI",
                                "投資回収期間"
                            ],
                            "値": [
                                st.session_state.selected_node,
                                selected_proposal_type,
                                proposal_text,
                                f"{proposal_initial_cost:,.0f}円",
                                f"{proposal_monthly_cost:,.0f}円/月",
                                f"{proposal_monthly_benefit:,.0f}円/月",
                                f"{proposal_implementation_months}ヶ月",
                                f"{results['first_year_roi']:.1f}%",
                                f"{results['annual_roi']:.1f}%",
                                f"{int(results['payback_years'])}年{int((results['payback_years'] - int(results['payback_years'])) * 12)}ヶ月" if results['payback_years'] < float('inf') else "回収不能"
                            ]
                        }
                        
                        export_df = pd.DataFrame(export_data)
                        csv = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label="提案・ROI結果をCSVでダウンロード",
                            data=csv,
                            file_name="proposal_roi_results.csv",
                            mime="text/csv"
                        )
                
                # 「やり直す」ボタン - ノード選択をリセット
                if st.session_state.node_selection_complete:
                    if st.button("選択し直す"):
                        st.session_state.node_selection_complete = False
                        st.session_state.selected_node = None
                        st.session_state.selected_node_id = None
                        
                        # 提案結果もクリア
                        if "proposal_text" in st.session_state:
                            del st.session_state.proposal_text
                        if "proposal_summary" in st.session_state:
                            del st.session_state.proposal_summary
                        if "proposal_results" in st.session_state:
                            del st.session_state.proposal_results
                        
                        st.rerun()
    
    # タブ3: 簡素化されたROI計算
    with tab3:
        st.header("ROI計算")
        st.markdown("提案のROI（投資対効果）の計算を行います。単年ROIを基本とし、必要に応じて多年度分析も可能です。")
        
        # 単年ROIの計算と表示
        st.subheader("単年ROI計算")
        
        # 入力フォーム
        col1, col2 = st.columns(2)
        
        with col1:
            investment = st.number_input("初期投資額（円）", min_value=0, value=1000000, step=100000, format="%d")
            monthly_cost = st.number_input("月額費用（円/月）", min_value=0, value=50000, step=10000, format="%d")
            
        with col2:
            monthly_benefit = st.number_input("月額効果（円/月）", min_value=0, value=200000, step=10000, format="%d")
            implementation_months = st.slider("初年度の実装月数", min_value=1, max_value=12, value=10)
        
        # 単年ROI計算
        annual_cost = investment + (monthly_cost * implementation_months)
        annual_benefit = monthly_benefit * implementation_months
        
        if annual_cost > 0:
            roi_percentage = (annual_benefit - annual_cost) / annual_cost * 100
        else:
            roi_percentage = 0
        
        # 結果表示
        st.markdown("### 単年ROI計算結果")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総投資額", f"{annual_cost:,.0f}円")
        with col2:
            st.metric("総効果額", f"{annual_benefit:,.0f}円")
        with col3:
            st.metric("ROI", f"{roi_percentage:.1f}%")
        
        # 詳細表示
        st.markdown("### 計算式と内訳")
        st.markdown(f"ROI = (総効果額 - 総投資額) / 総投資額 × 100%")
        st.markdown(f"ROI = ({annual_benefit:,.0f}円 - {annual_cost:,.0f}円) / {annual_cost:,.0f}円 × 100% = {roi_percentage:.1f}%")
        
        st.markdown("#### 投資内訳")
        st.markdown(f"- 初期投資: {investment:,.0f}円")
        st.markdown(f"- 運用コスト: {monthly_cost:,.0f}円/月 × {implementation_months}ヶ月 = {monthly_cost * implementation_months:,.0f}円")
        
        st.markdown("#### 効果内訳")
        st.markdown(f"- 月間効果: {monthly_benefit:,.0f}円/月")
        st.markdown(f"- 年間効果: {monthly_benefit:,.0f}円/月 × {implementation_months}ヶ月 = {annual_benefit:,.0f}円")
        
        # 投資回収期間
        if monthly_benefit > monthly_cost:
            monthly_net_benefit = monthly_benefit - monthly_cost
            months_to_recover = investment / monthly_net_benefit if monthly_net_benefit > 0 else float('inf')
            
            st.markdown("#### 投資回収期間")
            if months_to_recover < float('inf'):
                years = int(months_to_recover / 12)
                months = int(months_to_recover % 12)
                st.markdown(f"投資回収期間: 約{years}年{months}ヶ月")
            else:
                st.markdown("月間効果が月間コストを上回らないため、投資回収期間は計算できません")
        
        # 多年度分析（オプション）
        with st.expander("多年度ROI分析（オプション）"):
            st.markdown("### 多年度ROI分析")
            
            years = st.slider("分析年数", min_value=1, max_value=5, value=3)
            
            # 入力設定
            st.markdown("#### 年度別設定")
            
            # 初期投資配分
            st.markdown("**初期投資配分**")
            investment_distribution = st.slider(
                "初期投資の年度配分（初年度の割合）", 
                min_value=0.5, 
                max_value=1.0, 
                value=0.7,
                help="1.0に近いほど初年度に投資が集中し、後年の投資は少なくなります"
            )
            
            # 効果の発現遅延
            st.markdown("**効果の発現カーブ**")
            benefit_curve = st.slider(
                "効果の発現カーブ", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.3,
                help="0.0は初年度から最大効果、1.0に近いほど効果の発現が遅れます"
            )
            
            # 年度別データを計算
            yearly_data = []
            cumulative_data = []
            
            total_investment = investment + (monthly_cost * 12 * years)
            monthly_full_benefit = monthly_benefit  # 最大月間効果
            
            cum_investment = 0
            cum_benefit = 0
            
            for year in range(1, years + 1):
                # 投資額計算（初期投資配分 + 年間運用コスト）
                year_progress = year / years
                
                # 初期投資の配分（指数関数的減少）
                if investment_distribution > 0:
                    investment_factor = (1 - year_progress) ** (investment_distribution * 3)
                    year_initial_investment = (
                        investment * investment_factor / 
                        sum([(1 - y/years) ** (investment_distribution * 3) for y in range(1, years + 1)])
                    )
                else:
                    # 均等配分
                    year_initial_investment = investment / years
                
                # 年間運用コスト
                year_operation_cost = monthly_cost * 12
                
                # 年間総投資額
                year_total_investment = year_initial_investment + year_operation_cost
                
                # 効果計算（指数関数的増加）
                if benefit_curve > 0:
                    benefit_factor = year_progress ** (benefit_curve * 3)
                    ramp_up_factor = benefit_factor / sum([(y/years) ** (benefit_curve * 3) for y in range(1, years + 1)])
                else:
                    # 均等効果
                    ramp_up_factor = 1 / years
                
                year_benefit = monthly_full_benefit * 12 * min(1.0, ramp_up_factor * years)
                
                # 年間ROI
                if year_total_investment > 0:
                    year_roi = (year_benefit - year_total_investment) / year_total_investment * 100
                else:
                    year_roi = 0
                
                # 累積データを更新
                cum_investment += year_total_investment
                cum_benefit += year_benefit
                
                if cum_investment > 0:
                    cum_roi = (cum_benefit - cum_investment) / cum_investment * 100
                else:
                    cum_roi = 0
                
                # 年間データを追加
                yearly_data.append({
                    "年": year,
                    "初期投資": year_initial_investment,
                    "運用コスト": year_operation_cost,
                    "総投資": year_total_investment,
                    "効果": year_benefit,
                    "純利益": year_benefit - year_total_investment,
                    "ROI": year_roi
                })
                
                # 累積データを追加
                cumulative_data.append({
                    "年": year,
                    "累積投資": cum_investment,
                    "累積効果": cum_benefit,
                    "累積純利益": cum_benefit - cum_investment,
                    "累積ROI": cum_roi
                })
            
            # データ表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 年別データ")
                yearly_df = pd.DataFrame(yearly_data)
                st.dataframe(yearly_df.style.format({
                    "初期投資": "{:,.0f}円",
                    "運用コスト": "{:,.0f}円",
                    "総投資": "{:,.0f}円",
                    "効果": "{:,.0f}円",
                    "純利益": "{:,.0f}円",
                    "ROI": "{:.1f}%"
                }), use_container_width=True)
            
            with col2:
                st.markdown("#### 累積データ")
                cumulative_df = pd.DataFrame(cumulative_data)
                st.dataframe(cumulative_df.style.format({
                    "累積投資": "{:,.0f}円",
                    "累積効果": "{:,.0f}円",
                    "累積純利益": "{:,.0f}円",
                    "累積ROI": "{:.1f}%"
                }), use_container_width=True)
            
            # 損益分岐点
            breakeven_year = None
            for i, data in enumerate(cumulative_data):
                if data["累積純利益"] >= 0:
                    if i > 0 and cumulative_data[i-1]["累積純利益"] < 0:
                        # 線形補間で月単位の損益分岐点を計算
                        prev_profit = cumulative_data[i-1]["累積純利益"]
                        curr_profit = data["累積純利益"]
                        
                        # 損益がゼロになる比率を計算
                        ratio = -prev_profit / (curr_profit - prev_profit) if (curr_profit - prev_profit) != 0 else 0
                        breakeven_year = i + ratio
                    else:
                        breakeven_year = i + 1
                    break
            
            # 損益分岐点の表示
            st.markdown("#### 損益分岐点分析")
            if breakeven_year:
                years_part = int(breakeven_year)
                months_part = int((breakeven_year - years_part) * 12)
                st.success(f"損益分岐点: {years_part}年{months_part}ヶ月")
            else:
                st.warning(f"設定した{years}年間では損益分岐点に達しません")
            
            # 多年度分析結果サマリー
            st.markdown("#### 多年度分析サマリー")
            
            final_data = cumulative_data[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{years}年間総投資", f"{final_data['累積投資']:,.0f}円")
            with col2:
                st.metric(f"{years}年間総効果", f"{final_data['累積効果']:,.0f}円")
            with col3:
                st.metric(f"{years}年間ROI", f"{final_data['累積ROI']:.1f}%")


if __name__ == "__main__":
    main()