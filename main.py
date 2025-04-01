"""
main.py
Streamlit app for ROI Analysis with natural language understanding and real streaming
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.deepdive_agent import ChallengeAgent
from agents.proposal_agent import ProposalAgent
from domain.roitree import ROINode, mermaid_to_roi_tree, get_leaf_nodes


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


def has_numerical_value(node_label: str) -> bool:
    """
    ノードラベルに数値情報（金額、パーセンテージなど）が含まれているかをチェック
    """
    # 金額のパターン（例: 1億円, 500万円, 1,000円）
    amount_pattern = r'(\d+[,\.]?\d*\s*[億万千]?円)'
    # パーセンテージのパターン（例: 25%, 3.5%）
    percentage_pattern = r'(\d+[,\.]?\d*\s*%)'
    
    return bool(re.search(amount_pattern, node_label)) or bool(re.search(percentage_pattern, node_label))


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


def generate_node_question(nodes_without_values):
    """
    数値が欠けているノードについて質問を生成する
    """
    if not nodes_without_values:
        return "すべてのノードには既に数値が設定されています。ROIツリーについて他に質問はありますか？"
    
    # ランダムに1つのノードを選択
    node_id, node_label = random.choice(nodes_without_values)
    
    # 質問のバリエーション
    questions = [
        f"「{node_label}」について、具体的な数値目標はありますか？金額や割合など、どのくらいの効果を期待していますか？",
        f"「{node_label}」について、どのくらいの影響があると予想されますか？例えば売上増加額や削減率など教えていただけますか？",
        f"「{node_label}」の目標値や期待効果について教えてください。",
        f"「{node_label}」に関してはどの程度の効果を見込んでいますか？"
    ]
    
    return random.choice(questions)


class NLUAgent:
    """
    自然言語理解を行い、ユーザーのメッセージからノード情報と数値を抽出するエージェント
    """
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            streaming=False  # 解析には素早いレスポンスが必要
        )
        
        # 解析用のプロンプト
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析の専門家です。ユーザーの自然言語メッセージからノード名と数値情報を正確に抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_node": true/false,  // ノード名が見つかったかどうか
  "node_name": "抽出したノード名", // 見つかった場合のノード名
  "found_value": true/false,  // 数値が見つかったかどうか
  "value": "抽出した数値",     // 見つかった場合の数値（単位を含む）
  "confidence": 0-100        // 抽出結果の確信度（0-100）
}}
```

注意:
- 数値は「円」「万円」「億円」「%」などの単位を含めて抽出してください
- ノード名は完全一致でなくても、明らかに指しているノードがあれば抽出してください
- 抽出できない場合は対応するフィールドをfalseにしてください
- 確信度は抽出結果の信頼性を0-100で表してください
"""),
            ("human", """以下のユーザーメッセージから、ノード名と数値情報を抽出してください。

ユーザーメッセージ: "{message}"

利用可能なノード:
{node_list}

JSON形式で回答してください。
""")
        ])
    
    def extract_node_and_value(self, message: str, mermaid_diagram: str) -> Dict:
        """ユーザーメッセージからノード名と数値を抽出する"""
        # 利用可能なノードのリストを作成
        leaf_nodes = extract_leaf_nodes(mermaid_diagram)
        node_list = "\n".join([f"- {label}" for _, label in leaf_nodes])
        
        # LLMで解析
        result = self.llm.invoke(
            self.extract_prompt.format(
                message=message,
                node_list=node_list
            )
        )
        
        # JSONを抽出
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                extraction_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                extraction_result = json.loads(result.content)
            
            # ノードIDの検索（抽出されたノード名に近いノードを検索）
            node_id = None
            if extraction_result.get("found_node", False) and extraction_result.get("node_name"):
                node_name = extraction_result["node_name"]
                
                # ノード名が類似するノードを検索
                for nid, label in leaf_nodes:
                    # 簡易的な類似度チェック（部分文字列）
                    if node_name.lower() in label.lower() or label.lower() in node_name.lower():
                        node_id = nid
                        extraction_result["matched_label"] = label
                        break
            
            extraction_result["node_id"] = node_id
            return extraction_result
            
        except Exception as e:
            print(f"JSON解析エラー: {e}")
            return {
                "found_node": False,
                "found_value": False,
                "confidence": 0,
                "error": str(e)
            }
    
    def analyze_conversation(self, conversation_history: List, current_message: str, mermaid_diagram: str) -> Dict:
        """会話の文脈を考慮してメッセージを分析する"""
        # まず単純に現在のメッセージだけで分析
        initial_result = self.extract_node_and_value(current_message, mermaid_diagram)
        
        # 高確信度の結果が得られた場合はそのまま返す
        if initial_result.get("confidence", 0) > 80:
            return initial_result
        
        # 低確信度の場合は、会話履歴も含めて再分析
        recent_context = "\n".join([
            f"{'ユーザー' if role == 'user' else 'アシスタント'}: {content}"
            for role, content in conversation_history[-3:] if role in ['user', 'assistant']
        ])
        
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析の専門家です。会話の文脈を考慮して、最新のユーザーメッセージからノード名と数値情報を抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_node": true/false,
  "node_name": "抽出したノード名",
  "found_value": true/false,
  "value": "抽出した数値",
  "confidence": 0-100
}}
```"""),
            ("human", """以下の会話の文脈を考慮して、最新のユーザーメッセージからノード名と数値情報を抽出してください。

会話の文脈:
{context}

最新のメッセージ: "{message}"

利用可能なノード:
{node_list}

JSON形式で回答してください。
""")
        ])
        
        # 利用可能なノードのリストを作成
        leaf_nodes = extract_leaf_nodes(mermaid_diagram)
        node_list = "\n".join([f"- {label}" for _, label in leaf_nodes])
        
        result = self.llm.invoke(
            context_prompt.format(
                context=recent_context,
                message=current_message,
                node_list=node_list
            )
        )
        
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                context_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                context_result = json.loads(result.content)
            
            # ノードIDの検索
            node_id = None
            if context_result.get("found_node", False) and context_result.get("node_name"):
                node_name = context_result["node_name"]
                
                for nid, label in leaf_nodes:
                    if node_name.lower() in label.lower() or label.lower() in node_name.lower():
                        node_id = nid
                        context_result["matched_label"] = label
                        break
            
            context_result["node_id"] = node_id
            return context_result
            
        except Exception as e:
            print(f"文脈解析エラー: {e}")
            return initial_result  # エラーの場合は最初の結果を返す


class ROIChatAgent:
    """
    ROIツリーを対話的に構築するチャットエージェント
    """
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,
            streaming=True  # 実際のストリーミング
        )
        
        self.nlu_agent = NLUAgent(model_name)
        
        # チャット用のプロンプト
        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析のエキスパートです。ユーザーと対話しながら、ROIツリーの各ノードに適切な数値を設定し、ビジネス分析を支援します。

現在のROIツリーには、まだ数値が設定されていないノードがあります。自然な会話の中で、こうしたノードに適切な数値を設定できるよう誘導してください。

レスポンスでは以下を心がけてください：
- 会話は親しみやすく自然な流れを保つ
- 数値情報のリクエストは押し付けがましくならないよう配慮する
- ユーザーが提供した情報を受け止め、適切なフィードバックを提供する
- ROIツリーの構造や目的に関する質問にも答える
- 一度に複数のノードの情報を求めないよう注意する

以下のノードについての情報を対話的に収集してください：
{nodes_without_values}
"""),
            ("human", "{user_message}")
        ])
    
    def get_response(self, user_message: str, conversation_history: List, mermaid_diagram: str, callback=None):
        """ユーザーメッセージに対する応答を生成する"""
        # 数値が設定されていないノードを取得
        nodes_without_values = extract_nodes_without_values(mermaid_diagram)
        nodes_text = "\n".join([f"- {label}" for _, label in nodes_without_values])
        
        # LLMで応答を生成 （コールバック付きの場合はストリーミング）
        if callback:
            response = self.llm.with_config({"callbacks": [callback]}).invoke(
                self.chat_prompt.format(
                    nodes_without_values=nodes_text,
                    user_message=user_message
                )
            )
        else:
            response = self.llm.invoke(
                self.chat_prompt.format(
                    nodes_without_values=nodes_text,
                    user_message=user_message
                )
            )
        
        return response.content
    
    def analyze_message(self, message: str, conversation_history: List, mermaid_diagram: str) -> Dict:
        """メッセージを分析し、ノード更新情報を抽出する"""
        return self.nlu_agent.analyze_conversation(conversation_history, message, mermaid_diagram)


def render_combined_trees(challenge_mermaid, proposal_mermaid, height=800):
    """
    課題ツリーと提案ツリーを組み合わせて表示する
    課題ツリーは上から下、提案ツリーは下から上に表示
    課題の末端ノードと対応する提案ノードを接続する
    """
    # 課題ツリーからgraph TDなどの冒頭部分を削除
    challenge_lines = challenge_mermaid.strip().split('\n')
    if challenge_lines[0].startswith('graph ') or challenge_lines[0].startswith('flowchart '):
        challenge_lines = challenge_lines[1:]
    challenge_content = '\n'.join(challenge_lines)
    
    # 提案ツリーからflowchart BTなどの冒頭部分を削除
    proposal_lines = proposal_mermaid.strip().split('\n')
    if proposal_lines[0].startswith('graph ') or proposal_lines[0].startswith('flowchart '):
        proposal_lines = proposal_lines[1:]
    proposal_content = '\n'.join(proposal_lines)
    
    # 課題ツリーから末端ノードを抽出
    challenge_leaf_nodes = extract_leaf_nodes(challenge_mermaid)
    
    # 提案ツリーから提案ノードを抽出（ここは簡易的な実装）
    proposal_nodes = extract_leaf_nodes(proposal_mermaid)
    
    # 末端ノードと提案ノードの接続を生成
    connections = []
    
    # 末端ノードと提案ノードの数が同じ場合は1対1で接続
    if len(challenge_leaf_nodes) == len(proposal_nodes):
        for i in range(len(challenge_leaf_nodes)):
            challenge_node = challenge_leaf_nodes[i][0]
            proposal_node = proposal_nodes[i][0]
            # 課題ノードから提案ノードへの接続
            connections.append(f"{challenge_node} -.-> {proposal_node}")
    else:
        # 数が一致しない場合は、できるだけマッチさせる（簡易的な実装）
        max_connections = min(len(challenge_leaf_nodes), len(proposal_nodes))
        for i in range(max_connections):
            challenge_node = challenge_leaf_nodes[i][0]
            proposal_node = proposal_nodes[i][0]
            connections.append(f"{challenge_node} -.-> {proposal_node}")
    
    # 両方を組み合わせたMermaid図を作成
    combined_mermaid = f"""
flowchart TD
    %% スタイル定義
    classDef challenge fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef proposal fill:#e6f7ff,stroke:#0066cc,stroke-width:1px
    classDef connection stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    
    %% 課題ツリー（上から下）
    subgraph 課題分析["課題分析"]
    {challenge_content}
    end
    
    %% 提案ツリー（下から上）
    subgraph 提案策["提案策"]
    {proposal_content}
    end
    
    %% 課題末端ノードと提案ノードを接続
    {'\n    '.join(connections)}
    
    %% クラス適用
    class 課題分析 challenge
    class 提案策 proposal
    """
    
    render_mermaid(combined_mermaid, height=height)


def main():
    """Main Streamlit application"""
    # アプリのタイトルとヘッダー
    st.set_page_config(
        page_title="ROI分析ツール",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 ROIツリー分析ツール")
    st.markdown("""
    このツールは、ビジネス課題をROIの観点から構造化し、分析・提案するためのツールです。
    テキストボックスにビジネス課題を入力して「分析開始」ボタンをクリックしてください。
    """)
    
    # サイドバーの設定
    with st.sidebar:
        st.title("設定")
        model_name = st.selectbox(
            "使用するモデル",
            ["gpt-4o", "gpt-3.5-turbo"],
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
    tab1, tab2 = st.tabs(["課題分析", "提案生成"])
    
    # タブ1: 課題分析
    with tab1:
        st.header("課題分析")
        st.markdown("ビジネス課題をROIツリーとして構造化します。")
        
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
        
        challenge_text = st.text_area(
            "ビジネス課題を入力してください",
            value=st.session_state.challenge_text,
            height=150,
            placeholder="例: ゴム製品メーカーでDX推進を担当しています。製造ラインの自動化により..."
        )
        
        # 分析実行ボタン
        if st.button("分析開始", type="primary", disabled=not challenge_text):
            with st.spinner("ROIツリーを生成中..."):
                # ChallengeAgentを初期化して実行
                agent = ChallengeAgent(model_name=model_name)
                root_node, mermaid_diagram = agent.create_roi_tree(challenge_text)
                
                # 結果を保存
                st.session_state.mermaid_diagram = mermaid_diagram
                st.session_state.root_node = root_node
                st.session_state.challenge_agent = agent
                
                # ROIチャットエージェントを初期化
                st.session_state.roi_chat_agent = ROIChatAgent(model_name=model_name)
                
                # 末端ノードの分析
                analysis = agent.analyze_leaf_nodes(root_node, mermaid_diagram)
                st.session_state.analysis = analysis
                
                # チャット履歴をリセット
                st.session_state.chat_history = []
                st.session_state.mermaid_history = [mermaid_diagram]
                st.session_state.chat_active = False
        
        # 分析結果の表示
        if "mermaid_diagram" in st.session_state:
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
                
                # コードとしても表示
                with st.expander("Mermaidコード"):
                    if st.session_state.mermaid_history:
                        st.code(st.session_state.mermaid_history[-1], language="text")
                    else:
                        st.code(st.session_state.mermaid_diagram, language="text")
            
            with col2:
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
                        
                        if amount_match:
                            nodes_with_values.append((node_id, node_label, amount_match.group(1)))
                    
                    return nodes_with_values
                
                # 現在の末端ノードの値を表示
                current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                nodes_with_values = get_node_values(current_mermaid)
                
                if nodes_with_values:
                    items = []
                    for _, node_label, value in nodes_with_values:
                        items.append({"ノード": node_label.split(" (")[0], "値": value})
                    
                    st.dataframe(pd.DataFrame(items), use_container_width=True)
                else:
                    st.info("数値が設定されたノードがありません")
            
            st.markdown("---")
            st.subheader("チャットでROIツリーを編集")
            
            # チャット開始ボタン
            chat_started = False
            if not st.session_state.chat_active:
                if st.button("チャットを開始", type="primary"):
                    st.session_state.chat_active = True
                    chat_started = True
                    
                    # 数値がないノードを抽出
                    current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                    nodes_without_values = extract_nodes_without_values(current_mermaid)
                    
                    # 初期メッセージを生成
                    if nodes_without_values:
                        initial_message = f"ROIツリーの分析を始めます。{len(nodes_without_values)}個のノードにはまだ数値が設定されていません。自由に会話をしながら、各ノードの数値目標を設定していきましょう。"
                        next_question = generate_node_question(nodes_without_values)
                    else:
                        initial_message = "ROIツリーの分析を始めます。すべてのノードに数値が設定されています。何か詳しく知りたい点はありますか？"
                        next_question = ""
                    
                    # 初期メッセージをチャット履歴に追加
                    st.session_state.chat_history.append(("assistant", initial_message))
                    
                    # 次の質問があれば追加
                    if next_question:
                        st.session_state.chat_history.append(("assistant", next_question))
            
            # チャット履歴の表示
            chat_container = st.container()
            
            with chat_container:
                for i, (role, text) in enumerate(st.session_state.chat_history):
                    if role == "user":
                        st.chat_message("user").write(text)
                    else:
                        st.chat_message("assistant").write(text)
                    
                    # チャットメッセージの後にROIツリーが更新された場合、表示
                    if i < len(st.session_state.mermaid_history) - 1:
                        st.caption("ROIツリーが更新されました：")
                        render_mermaid(st.session_state.mermaid_history[i+1])
            
            # チャットが有効な場合、ユーザー入力を受け付ける
            if st.session_state.chat_active and st.session_state.roi_chat_agent:
                # ユーザー入力
                if prompt := st.chat_input("メッセージを入力..."):
                    # ユーザーのメッセージを表示
                    st.chat_message("user").write(prompt)
                    st.session_state.chat_history.append(("user", prompt))
                    
                    # 現在のROIツリー状態を取得
                    current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
                    current_root = mermaid_to_roi_tree(current_mermaid)
                    
                    # メッセージを分析して数値情報を抽出
                    analysis_result = st.session_state.roi_chat_agent.analyze_message(
                        prompt, 
                        st.session_state.chat_history,
                        current_mermaid
                    )
                    
                    # ツリーを更新すべきかどうかを判定
                    should_update = (
                        analysis_result.get("found_node", False) and 
                        analysis_result.get("found_value", False) and 
                        analysis_result.get("node_id") and 
                        analysis_result.get("value") and
                        analysis_result.get("confidence", 0) > 50  # 確信度が50%以上
                    )
                    
                    # 実際のレスポンスを生成（LLMストリーミング）
                    with st.chat_message("assistant"):
                        # リアルストリーミング用のプレースホルダー
                        stream_handler = StreamHandler(st)
                        
                        # ストリーミングレスポンスを取得
                        response = st.session_state.roi_chat_agent.get_response(
                            prompt,
                            st.session_state.chat_history,
                            current_mermaid,
                            callback=stream_handler
                        )
                    
                    # チャット履歴に追加
                    st.session_state.chat_history.append(("assistant", response))
                    
                    # ROIツリーを更新する必要がある場合
                    updated_mermaid = current_mermaid
                    if should_update:
                        node_id = analysis_result["node_id"]
                        value = analysis_result["value"]
                        
                        # ノードの値を更新
                        updated_mermaid = update_node_value_in_mermaid(current_mermaid, node_id, value)
                        updated_root = mermaid_to_roi_tree(updated_mermaid)
                        
                        # 更新を保存
                        st.session_state.mermaid_history.append(updated_mermaid)
                        
                        # 更新されたROIツリーを表示
                        st.caption("ROIツリーが更新されました：")
                        render_mermaid(updated_mermaid)
                        
                        # 再分析を実行
                        if "challenge_agent" in st.session_state:
                            updated_analysis = st.session_state.challenge_agent.analyze_leaf_nodes(
                                updated_root, updated_mermaid
                            )
                            st.session_state.analysis = updated_analysis
                    
                    # 次の質問（自動生成）は次のユーザー入力を待つ
                    
                    # ページをリロードして最新の状態を表示
                    st.rerun()
            
            if chat_started:
                st.rerun()
    
    # タブ2: 提案生成
    with tab2:
        st.header("提案生成")
        st.markdown("課題分析に基づいて、具体的な提案を生成します。提案は課題ツリーの末端ノードと1対1で対応し、下から上に逆さまに展開します。")
        
        # 課題ツリーが生成されているか確認
        if "mermaid_diagram" not in st.session_state and not st.session_state.mermaid_history:
            st.warning("最初に「課題分析」タブで課題ツリーを生成してください。")
        else:
            # 提案の方向性を入力するテキストエリア
            if "proposal_guidance_text" not in st.session_state:
                st.session_state.proposal_guidance_text = ""
            
            proposal_guidance = st.text_area(
                "提案の方向性（任意）",
                value=st.session_state.proposal_guidance_text,
                height=100,
                placeholder="例: コスト効率を重視し、段階的に導入できるソリューションを希望します。現場の反発を最小限に抑える方法も考慮してください。"
            )
            
            # 提案のサンプル文を表示
            with st.expander("提案の方向性の例"):
                st.markdown("""
                **例1: コスト効率重視**
                > コスト効率を重視し、初期投資を抑えながら効果を最大化する提案が欲しい。ROIは30%以上を目標とし、1年以内に効果が見えるソリューションを優先したい。

                **例2: 段階的導入**
                > リスクを分散するため、段階的に導入できる提案が望ましい。第一フェーズは3ヶ月以内に開始でき、効果が早く見えるものから始めたい。

                **例3: 人材活用**
                > 現場のAI不安に配慮し、既存スタッフのスキルアップを含めた人材活用型の提案を希望。教育コストも含めた総合的なROI計算を示してほしい。
                """)
            
            # 現在のROIツリーを取得（更新履歴がある場合は最新のものを使用）
            current_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else st.session_state.mermaid_diagram
            
            # 提案生成ボタン
            if st.button("提案を生成", type="primary"):
                with st.spinner("提案を生成中..."):
                    # 数値がないノードをチェック
                    nodes_without_values = extract_nodes_without_values(current_mermaid)
                    if nodes_without_values:
                        st.warning(f"まだ{len(nodes_without_values)}個のノードに数値が設定されていません。より正確な提案のために、すべてのノードに数値を設定することをお勧めします。")
                        missing_list = ", ".join([label for _, label in nodes_without_values])
                        st.info(f"数値が設定されていないノード: {missing_list}")
                    
                    # ProposalAgentを初期化して実行
                    proposal_agent = ProposalAgent(model_name=model_name)
                    
                    # ストリーミング出力用のコンテナ
                    proposal_container = st.empty()
                    
                    # ストリーミングハンドラ
                    stream_handler = StreamHandler(proposal_container)
                    
                    # ストリーミングでの提案生成
                    proposal_text, summary = proposal_agent.generate_proposal_streaming(
                        current_mermaid,
                        proposal_guidance,
                        callback=stream_handler
                    )
                    
                    # 結果を保存
                    st.session_state.proposal_text = proposal_text
                    st.session_state.proposal_summary = summary
                    st.session_state.proposal_guidance_text = proposal_guidance
            
            # 提案結果の表示
            if "proposal_text" in st.session_state:
                st.markdown("---")
                st.header("提案結果")
                
                # 提案テキストからMermaidダイアグラムを抽出
                # flowchart BTを探す（逆さまツリー）
                mermaid_match = re.search(r'```(?:mermaid)?\s*(flowchart\s+BT[\s\S]*?)```', st.session_state.proposal_text, re.DOTALL)
                if mermaid_match:
                    proposal_mermaid = mermaid_match.group(1)
                else:
                    # 他の形式も試してみる
                    alternative_matches = [
                        re.search(r'```(?:mermaid)?\s*(flowchart\s+TD[\s\S]*?)```', st.session_state.proposal_text, re.DOTALL),
                        re.search(r'```(?:mermaid)?\s*(graph\s+BU[\s\S]*?)```', st.session_state.proposal_text, re.DOTALL),
                        re.search(r'```(?:mermaid)?\s*(graph\s+BT[\s\S]*?)```', st.session_state.proposal_text, re.DOTALL),
                        re.search(r'```(?:mermaid)?\s*(graph\s+TD[\s\S]*?)```', st.session_state.proposal_text, re.DOTALL)
                    ]
                    
                    for match in alternative_matches:
                        if match:
                            # 見つかった場合、flowchart BTに変換
                            diagram_text = match.group(1)
                            if diagram_text.startswith("graph TD"):
                                diagram_text = "flowchart BT" + diagram_text[8:]
                            elif diagram_text.startswith("graph BU"):
                                diagram_text = "flowchart BT" + diagram_text[8:]
                            elif diagram_text.startswith("graph BT"):
                                diagram_text = "flowchart BT" + diagram_text[8:]
                                
                            proposal_mermaid = diagram_text
                            break
                    else:
                        # どのパターンにも一致しない場合
                        proposal_mermaid = None
                
                # 提案ツリーと提案サマリーを表示するためのタブ
                prop_tab1, prop_tab2, prop_tab3 = st.tabs(["統合ビュー", "提案ツリー", "提案サマリー"])
                
                with prop_tab1:
                    st.subheader("課題と提案の統合ビュー")
                    if proposal_mermaid:
                        # 末端ノードと提案ノードを結ぶ接続線が含まれる統合ビューを表示
                        render_combined_trees(
                            current_mermaid,
                            proposal_mermaid,
                            height=900
                        )
                    else:
                        st.warning("提案ツリーの表示に問題があります。個別のツリーをご確認ください。")
                
                with prop_tab2:
                    if proposal_mermaid:
                        st.subheader("提案ツリー図（下から上に生える逆さまツリー）")
                        render_mermaid(proposal_mermaid, height=600)
                    else:
                        st.warning("提案ツリーの表示に問題があります。Mermaidコードを確認してください。")
                        with st.expander("生成されたテキスト"):
                            st.markdown(st.session_state.proposal_text)
                    
                    # 完全な提案テキストも表示
                    with st.expander("提案テキスト全文"):
                        st.markdown(st.session_state.proposal_text)
                
                with prop_tab3:
                    summary = st.session_state.proposal_summary
                    
                    # ROI結果のハイライト表示
                    st.subheader("ROI計算結果")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("総投資額", f"{summary.total_investment:,.0f}円")
                    with col2:
                        st.metric("総期待効果", f"{summary.total_benefit:,.0f}円")
                    with col3:
                        st.metric("ROI", f"{summary.roi_percentage:.1f}%")
                    
                    # 期待利益の計算
                    if summary.total_investment > 0:
                        net_profit = summary.total_benefit - summary.total_investment
                        roi_ratio = net_profit/summary.total_investment
                        st.metric("期待利益", f"{net_profit:,.0f}円", delta=f"{roi_ratio:.1%}")
                    else:
                        st.metric("期待利益", "計算できません", delta=None)
                    
                    st.subheader("実装期間")
                    st.info(summary.implementation_timeframe)
                    
                    st.subheader("主要提案ポイント")
                    for i, point in enumerate(summary.key_recommendations, 1):
                        st.markdown(f"{i}. {point}")


if __name__ == "__main__":
    main()