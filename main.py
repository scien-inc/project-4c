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

from agents.deepdive_agent import ChallengeAgent
from agents.proposal_agent import ProposalAgent
from agents.solution_agent import SolutionAgent
from agents.roi_chat_agent import ROIChatAgent
from agents.conversion_agent import ConversionAgent
from domain.roitree import ROINode, mermaid_to_roi_tree, get_leaf_nodes
from domain.schemas import NumericalValue


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
    tab1, tab2, tab3 = st.tabs(["1. 課題分析", "2. 提案生成", "3. ROI計算"])
    
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
        
        if "chat_focus" not in st.session_state:
            st.session_state.chat_focus = "data_collection"
        
        if "roi_chat_agent" not in st.session_state:
            st.session_state.roi_chat_agent = None
        
        if "prioritized_nodes" not in st.session_state:
            st.session_state.prioritized_nodes = []
        
        if "conversion_info" not in st.session_state:
            st.session_state.conversion_info = {}
        
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
                st.session_state.challenge_text = challenge_text
                
                # ROIチャットエージェントを初期化
                st.session_state.roi_chat_agent = ROIChatAgent(model_name=model_name)
                
                # 末端ノードの分析
                analysis = agent.analyze_leaf_nodes(root_node, mermaid_diagram)
                st.session_state.analysis = analysis
                
                # ソリューションの提案も準備
                solution_suggestions = agent.suggest_solutions(root_node, mermaid_diagram)
                st.session_state.solution_suggestions = solution_suggestions
                
                # チャット履歴をリセット
                st.session_state.chat_history = []
                st.session_state.mermaid_history = [mermaid_diagram]
                st.session_state.chat_active = False
                st.session_state.chat_focus = "data_collection"
        
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
                
                if analysis.incomplete_nodes:
                    st.subheader("改善が必要なノード")
                    for node in analysis.incomplete_nodes:
                        st.markdown(f"- **{node['name']}**: {node['issue']}")
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
                        initial_message = "ROIツリーの分析を始めます。すべてのノードに数値が設定されています。次は優先的に解決したいノードを選びましょう。"
                        next_question = "どのノードを優先的に解決したいですか？"
                        st.session_state.chat_focus = "prioritization"
                    
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
                    
                    # 現在のチャットフォーカスに応じた処理
                    if st.session_state.chat_focus == "unit_conversion":
                        # 単位変換の場合は変換情報を更新
                        conversion_info = st.session_state.conversion_info
                        
                        # 単位変換エージェントで追加情報を解析
                        conversion_agent = ConversionAgent(model_name=model_name)
                        additional_info = prompt
                        
                        # 変換を実行
                        conversion_result = conversion_agent.perform_conversion(
                            conversion_info.get("value", 0),
                            conversion_info.get("unit", ""),
                            conversion_info.get("description", ""),
                            additional_info
                        )
                        
                        # 変換結果をセッションに保存
                        st.session_state.conversion_result = conversion_result
                        
                        # 変換結果に基づいてノードを更新
                        if conversion_result.get("converted_value") is not None:
                            node_id = conversion_info.get("node_id")
                            if node_id:
                                new_value = f"{conversion_result.get('converted_value')}円"
                                updated_mermaid = update_node_value_in_mermaid(current_mermaid, node_id, new_value)
                                st.session_state.mermaid_history.append(updated_mermaid)
                        
                        # チャットフォーカスを元に戻す
                        st.session_state.chat_focus = "data_collection"
                    elif st.session_state.chat_focus == "prioritization":
                        # 優先ノードの選択の場合
                        # メッセージを分析して優先ノードを特定
                        # 簡易的な実装として、メッセージ内にノード名が含まれているか確認
                        leaf_nodes = extract_leaf_nodes(current_mermaid)
                        prioritized_nodes = []
                        
                        for node_id, node_label in leaf_nodes:
                            # ノード名からカッコ部分を除去
                            clean_label = re.sub(r'\s*\([^)]+\)', '', node_label)
                            if clean_label.lower() in prompt.lower():
                                prioritized_nodes.append({
                                    "node_id": node_id,
                                    "node_name": clean_label
                                })
                        
                        # 優先ノードが見つかった場合は保存
                        if prioritized_nodes:
                            st.session_state.prioritized_nodes = prioritized_nodes
                            priority_text = ", ".join([node["node_name"] for node in prioritized_nodes])
                            st.session_state.priority_nodes_text = f"優先ノード: {priority_text}"
                    else:
                        # 通常のデータ収集モード
                        # メッセージを分析して数値情報を抽出
                        analysis_result = st.session_state.roi_chat_agent.analyze_message(
                            prompt, 
                            st.session_state.chat_history,
                            current_mermaid
                        )
                        
                        # 単位情報も分析
                        unit_analysis = st.session_state.roi_chat_agent.analyze_unit_conversion(prompt)
                        
                        # ツリーを更新すべきかどうかを判定
                        should_update = (
                            analysis_result.get("found_node", False) and 
                            analysis_result.get("found_value", False) and 
                            analysis_result.get("node_id") and 
                            analysis_result.get("value") and
                            analysis_result.get("confidence", 0) > 50  # 確信度が50%以上
                        )
                        
                        # 単位変換が必要かチェック
                        needs_conversion = (
                            unit_analysis.get("found_value", False) and
                            unit_analysis.get("needs_conversion", False) and
                            len(unit_analysis.get("additional_info_needed", [])) > 0
                        )
                        
                        if needs_conversion and should_update:
                            # 単位変換に必要な情報を保存
                            conversion_info = {
                                "node_id": analysis_result.get("node_id"),
                                "node_name": analysis_result.get("node_name"),
                                "value": unit_analysis.get("value"),
                                "unit": unit_analysis.get("value_unit"),
                                "description": analysis_result.get("matched_label", ""),
                                "required_info": unit_analysis.get("additional_info_needed", [])
                            }
                            st.session_state.conversion_info = conversion_info
                            
                            # チャットフォーカスを単位変換に変更
                            st.session_state.chat_focus = "unit_conversion"
                        elif should_update:
                            # 単位変換が不要な場合は直接更新
                            node_id = analysis_result["node_id"]
                            value = analysis_result["value"]
                            
                            # ノードの値を更新
                            updated_mermaid = update_node_value_in_mermaid(current_mermaid, node_id, value)
                            st.session_state.mermaid_history.append(updated_mermaid)
                    
                    # 実際のレスポンスを生成（LLMストリーミング）
                    with st.chat_message("assistant"):
                        # リアルストリーミング用のプレースホルダー
                        stream_handler = StreamHandler(st)
                        
                        # ストリーミングレスポンスを取得
                        response = st.session_state.roi_chat_agent.get_response(
                            prompt,
                            st.session_state.chat_history,
                            current_mermaid,
                            current_focus=st.session_state.chat_focus,
                            callback=stream_handler
                        )
                    
                    # チャット履歴に追加
                    st.session_state.chat_history.append(("assistant", response))
                    
                    # すべてのノードにデータが揃ったかチェック
                    if st.session_state.chat_focus == "data_collection":
                        # 最新のMermaidを取得
                        latest_mermaid = st.session_state.mermaid_history[-1] if st.session_state.mermaid_history else current_mermaid
                        nodes_without_values = extract_nodes_without_values(latest_mermaid)
                        
                        # データが揃ったら優先ノード選択へ
                        if not nodes_without_values and not st.session_state.prioritized_nodes:
                            st.session_state.chat_focus = "prioritization"
                            st.session_state.chat_history.append(("assistant", "すべてのノードに数値が設定されました。次は、優先的に解決したいノードを教えていただけますか？"))
                    
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
            # 優先ノードの表示
            if st.session_state.prioritized_nodes:
                st.success("選択された優先ノード: " + ", ".join([node["node_name"] for node in st.session_state.prioritized_nodes]))
            
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
            
            # 優先ノードをテキスト形式に変換
            priority_nodes_text = ""
            if st.session_state.prioritized_nodes:
                priority_nodes_text = "\n".join([f"- {node['node_name']}" for node in st.session_state.prioritized_nodes])
            
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
                        priority_nodes=priority_nodes_text,
                        callback=stream_handler
                    )
                    
                    # 結果を保存
                    st.session_state.proposal_text = proposal_text
                    st.session_state.proposal_summary = summary
                    st.session_state.proposal_guidance_text = proposal_guidance
                    
                    # ソリューション推薦エージェントも実行
                    solution_agent = SolutionAgent(model_name=model_name)
                    leaf_nodes_text = "\n".join([f"- {label}" for _, label in extract_leaf_nodes(current_mermaid)])
                    
                    solution_recommendations = solution_agent.recommend_solutions(
                        leaf_nodes_text,
                        priority_nodes_text,
                        proposal_guidance
                    )
                    
                    st.session_state.solution_recommendations = solution_recommendations
            
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
                prop_tab1, prop_tab2, prop_tab3, prop_tab4 = st.tabs(["統合ビュー", "提案ツリー", "提案サマリー", "具体的ソリューション"])
                
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
                    
                    # 優先度ランキングがあれば表示
                    if summary.priority_ranking:
                        st.subheader("提案の優先度ランキング")
                        ranking_df = pd.DataFrame(summary.priority_ranking)
                        st.dataframe(ranking_df, use_container_width=True)
                
                with prop_tab4:
                    st.subheader("具体的なソリューション提案")
                    
                    if "solution_recommendations" in st.session_state:
                        recommendations = st.session_state.solution_recommendations.get("solution_recommendations", [])
                        
                        if recommendations:
                            # 優先度でソート
                            sorted_recommendations = sorted(recommendations, key=lambda x: x.get("priority", 999))
                            
                            for i, rec in enumerate(sorted_recommendations):
                                with st.expander(f"#{rec.get('priority', i+1)} {rec.get('solution_name', 'ソリューション')}"):
                                    st.markdown(f"**ノード**: {rec.get('leaf_node_name', '不明')}")
                                    st.markdown(f"**説明**: {rec.get('solution_description', '説明なし')}")
                                    
                                    est_cost = rec.get('estimated_cost', {})
                                    est_benefit = rec.get('estimated_benefit', {})
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.markdown(f"**実装コスト**: {est_cost.get('value', '不明')} {est_cost.get('unit', '')}")
                                    with col2:
                                        st.markdown(f"**期待効果**: {est_benefit.get('value', '不明')} {est_benefit.get('unit', '')}")
                                    with col3:
                                        st.markdown(f"**実装期間**: {rec.get('implementation_timeframe', '不明')}")
                        else:
                            st.info("具体的なソリューション提案は生成されていません。")
                    else:
                        st.info("具体的なソリューション提案は生成されていません。")
    
    # タブ3: ROI計算
    with tab3:
        st.header("ROI計算")
        st.markdown("提案に基づいたROI（投資対効果）の詳細計算を行います。")
        
        if "proposal_summary" not in st.session_state:
            st.warning("「提案生成」タブで提案を生成してからROI計算を行ってください。")
        else:
            summary = st.session_state.proposal_summary
            
            # ROI計算の詳細表示
            st.subheader("ROI計算シミュレーション")
            
            # 期間選択
            years = st.slider("計算期間（年）", 1, 5, 3)
            
            # コストと効果の分布
            cost_front_loaded = st.slider("コスト前倒し度", 0.0, 1.0, 0.7, 
                                        help="1.0に近いほど初期コストが大きく、後年は小さくなります")
            benefit_delay = st.slider("効果発現の遅延度", 0.0, 1.0, 0.3,
                                    help="1.0に近いほど効果の発現が遅れます")
            
            # 年ごとのシミュレーション計算
            yearly_data = []
            total_cost = summary.total_investment
            total_benefit = summary.total_benefit
            
            for year in range(1, years + 1):
                # コスト計算（前倒し）
                year_progress = year / years
                if cost_front_loaded > 0:
                    # 指数関数的減少
                    cost_factor = (1 - year_progress) ** (cost_front_loaded * 3)
                    yearly_cost = total_cost * cost_factor / sum([(1 - y/years) ** (cost_front_loaded * 3) for y in range(1, years + 1)])
                else:
                    # 均等配分
                    yearly_cost = total_cost / years
                
                # 効果計算（徐々に増加）
                if benefit_delay > 0:
                    # 指数関数的増加
                    benefit_factor = year_progress ** (benefit_delay * 3)
                    yearly_benefit = total_benefit * benefit_factor / sum([(y/years) ** (benefit_delay * 3) for y in range(1, years + 1)])
                else:
                    # 均等配分
                    yearly_benefit = total_benefit / years
                
                # 年間ROI
                yearly_roi = (yearly_benefit - yearly_cost) / yearly_cost * 100 if yearly_cost > 0 else 0
                
                yearly_data.append({
                    "年": year,
                    "コスト": yearly_cost,
                    "効果": yearly_benefit,
                    "利益": yearly_benefit - yearly_cost,
                    "ROI": yearly_roi
                })
            
            # 累積データの計算
            cumulative_data = []
            cum_cost = 0
            cum_benefit = 0
            
            for year_data in yearly_data:
                cum_cost += year_data["コスト"]
                cum_benefit += year_data["効果"]
                cum_roi = (cum_benefit - cum_cost) / cum_cost * 100 if cum_cost > 0 else 0
                
                cumulative_data.append({
                    "年": year_data["年"],
                    "累積コスト": cum_cost,
                    "累積効果": cum_benefit,
                    "累積利益": cum_benefit - cum_cost,
                    "累積ROI": cum_roi
                })
            
            # データ表示
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("年間ROI推移")
                yearly_df = pd.DataFrame(yearly_data)
                st.dataframe(yearly_df.style.format({
                    "コスト": "{:,.0f}円",
                    "効果": "{:,.0f}円",
                    "利益": "{:,.0f}円",
                    "ROI": "{:.1f}%"
                }), use_container_width=True)
            
            with col2:
                st.subheader("累積ROI推移")
                cumulative_df = pd.DataFrame(cumulative_data)
                st.dataframe(cumulative_df.style.format({
                    "累積コスト": "{:,.0f}円",
                    "累積効果": "{:,.0f}円",
                    "累積利益": "{:,.0f}円",
                    "累積ROI": "{:.1f}%"
                }), use_container_width=True)
            
            # 損益分岐点の計算
            breakeven_year = None
            for i, data in enumerate(cumulative_data):
                if data["累積利益"] >= 0:
                    if i > 0:
                        # 線形補間で月単位の損益分岐点を計算
                        prev_profit = cumulative_data[i-1]["累積利益"]
                        curr_profit = data["累積利益"]
                        prev_year = cumulative_data[i-1]["年"]
                        
                        # 損益がゼロになる比率を計算
                        if curr_profit - prev_profit != 0:  # ゼロ除算防止
                            ratio = -prev_profit / (curr_profit - prev_profit)
                            breakeven_year = prev_year + ratio
                        else:
                            breakeven_year = data["年"]
                    else:
                        breakeven_year = data["年"]
                    break
            
            # 損益分岐点の表示
            st.subheader("ROI分析結果")
            if breakeven_year:
                years_part = int(breakeven_year)
                months_part = int((breakeven_year - years_part) * 12)
                st.success(f"損益分岐点: {years_part}年{months_part}ヶ月")
            else:
                st.warning(f"設定した{years}年間では損益分岐点に達しません")
            
            # 最終年のROI
            final_roi = cumulative_data[-1]["累積ROI"]
            st.metric(f"{years}年後の累積ROI", f"{final_roi:.1f}%")
            
            # ROIが最大になる年
            max_roi_year = max(yearly_data, key=lambda x: x["ROI"])
            st.info(f"年間ROIが最大になるのは {max_roi_year['年']}年目 ({max_roi_year['ROI']:.1f}%)")


if __name__ == "__main__":
    main()