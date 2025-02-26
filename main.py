"""
ROIエージェントのStreamlitアプリケーション
"""
import os
import streamlit as st
from typing import Dict, List, Any, Optional
import json
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.callbacks import get_openai_callback

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# プロジェクト名を設定（オプション）
os.environ["LANGCHAIN_PROJECT"] = "roi_tree_explorer"


from agents.deepdive_agent import build_deepdive_graph, init_deepdive_state
from agents.proposal_agent import build_proposal_graph, init_proposal_state
from domain.roitree import ROINode, create_default_roi_tree
from domain.reflection import ReflectionManager
from utils.visualization import format_tree_for_display, generate_mermaid_diagram, format_roi_calculation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 定数
DEEPDIVE_STATE_KEY = "deepdive_state"
PROPOSAL_STATE_KEY = "proposal_state"
PAGE_TITLE = "ROIツリーエクスプローラー"
PAGE_ICON = "💼"

# Set page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

# =========================
# Streaming Handler for LangChain
# =========================

class StreamHandler(BaseCallbackHandler):
    """StreamlitにLLMのレスポンスをストリーミングするためのコールバックハンドラ"""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.message_placeholder = container.empty()
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新しいLLMトークンが生成されたときに実行"""
        self.text += token
        self.message_placeholder.markdown(self.text)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """LLMが終了したときに実行"""
        pass


# =========================
# Helper Functions
# =========================

def run_one_step(app, current_state, stream_handler=None):
    """
    LangGraphアプリを1ステップ実行する（オプションでストリーミング）
    
    Args:
        app: コンパイル済みのLangGraphアプリ
        current_state: 現在の状態
        stream_handler: オプションのストリーミングハンドラ
        
    Returns:
        更新された状態
    """
    if stream_handler:
        # ストリーミング使用
        gen = app.stream(current_state, {"callbacks": [stream_handler]})
    else:
        # ストリーミングなし
        gen = app.stream(current_state)
    
    try:
        # 次のノードが完了するまで実行
        latest_state = None
        for state in gen:
            latest_state = state
        return latest_state
    except StopIteration:
        # グラフの終了
        return current_state


def display_messages(messages: List[Any], container):
    """
    Streamlitコンテナに会話メッセージを表示（チャットの形式で）
    
    Args:
        messages: メッセージリスト
        container: Streamlitコンテナ
    """
    for msg in messages:
        if msg.type == "human":
            container.chat_message("user").markdown(msg.content)
        elif msg.type == "ai":
            container.chat_message("assistant").markdown(msg.content)
        elif msg.type == "system":
            container.info(msg.content)


def display_roi_tree(root_node: ROINode, container):
    """
    Display ROI tree in a Streamlit container
    
    Args:
        root_node: Root of the ROI tree
        container: Streamlit container
    """
    # Generate Mermaid diagram
    mermaid_diagram = generate_mermaid_diagram(root_node)
    
    # Display it using st.graphviz_chart
    container.subheader("ROI Tree Visualization")
    container.markdown(f"```mermaid\n{mermaid_diagram}\n```")
    
    # Also show as text
    container.subheader("ROI Tree Structure")
    tree_text = format_tree_for_display(root_node)
    container.markdown(tree_text)


def display_roi_calculations(roi_calculations: Dict[str, Any], container):
    """
    Display ROI calculations in a Streamlit container
    
    Args:
        roi_calculations: ROI calculation results
        container: Streamlit container
    """
    if not roi_calculations:
        container.info("No ROI calculations have been performed yet.")
        return
        
    container.subheader("ROI Calculations")
    
    if "summary" in roi_calculations:
        summary = roi_calculations["summary"]
        container.markdown(f"### 合計ROI: ¥{summary.get('weighted_value', 0)*1000:,.0f}")
        
        # 詳細な内訳を表示
        container.markdown("### ROI内訳")
        roi_text = format_roi_calculation(summary)
        container.markdown(roi_text)
    else:
        # 個別のノード計算を表示
        for node_id, calc in roi_calculations.items():
            if node_id != "summary":
                container.markdown(f"**{calc['name']}**: ¥{calc['value']*1000:,.0f}")
                
                if "confidence" in calc:
                    container.markdown(f"信頼度: {calc['confidence']:.0%}")
                
                if "assumptions" in calc and calc["assumptions"]:
                    container.markdown("前提条件:")
                    for assumption in calc["assumptions"]:
                        container.markdown(f"- {assumption}")
                
                container.markdown("---")


# =========================
# Main Streamlit App
# =========================

def main():
    """メインStreamlitアプリケーション"""
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    
    # サイドバーを追加
    with st.sidebar:
        st.title("設定")
        st.markdown("### 課題深掘り設定")
        
        # 課題深掘り設定
        min_nodes = st.slider("ブランチごとの最小ノード数", 1, 5, 3)
        max_iterations_deepdive = st.slider("最大反復回数（課題深掘り）", 3, 20, 10)
        
        st.markdown("### 提案設定")
        max_iterations_proposal = st.slider("最大反復回数（提案）", 3, 20, 10)
        
        # リセットボタン
        st.markdown("### リセット")
        if st.button("課題深掘りをリセット"):
            if DEEPDIVE_STATE_KEY in st.session_state:
                del st.session_state[DEEPDIVE_STATE_KEY]
            st.rerun()
            
        if st.button("提案をリセット"):
            if PROPOSAL_STATE_KEY in st.session_state:
                del st.session_state[PROPOSAL_STATE_KEY]
            st.rerun()
            
        if st.button("すべてリセット"):
            if DEEPDIVE_STATE_KEY in st.session_state:
                del st.session_state[DEEPDIVE_STATE_KEY]
            if PROPOSAL_STATE_KEY in st.session_state:
                del st.session_state[PROPOSAL_STATE_KEY]
            st.rerun()
    
    # タブを作成
    tab_deepdive, tab_proposal, tab_reflections = st.tabs(["課題深掘りエージェント", "提案エージェント", "反省データベース"])
    
    # =========================
    # 反省データベースタブ
    # =========================
    with tab_reflections:
        st.header("反省データベース")
        st.markdown("""
        このタブでは、エージェントが過去のタスク実行から学んだ反省データを確認できます。
        これらの反省は、将来のタスク実行の質を向上させるために使用されます。
        """)
        reflection_manager = ReflectionManager()
        
        # すべての反省を取得
        all_reflections = reflection_manager.get_all_reflections()
        
        if not all_reflections:
            st.info("まだ反省データはありません。エージェントを使用してデータを生成してください。")
        else:
            # 反省を日付でソート
            all_reflections.sort(key=lambda r: r.timestamp, reverse=True)
            
            # 反省データをテーブル表示用に変換
            reflection_data = []
            for reflection in all_reflections:
                reflection_data.append({
                    "ID": reflection.id,
                    "タイムスタンプ": reflection.timestamp,
                    "タスク": reflection.task,
                    "判断": reflection.judgment.result,
                    "再試行": "必要" if reflection.judgment.needs_retry else "不要",
                    "タグ": ", ".join(reflection.tags)
                })
            
            st.subheader("反省リスト")
            reflection_df = pd.DataFrame(reflection_data)
            st.dataframe(reflection_df, use_container_width=True)
            
            # 詳細表示
            st.subheader("反省詳細")
            selected_reflection_id = st.selectbox(
                "詳細を表示する反省を選択",
                options=[r.id for r in all_reflections],
                format_func=lambda id: next((r.task for r in all_reflections if r.id == id), id)
            )
            
            if selected_reflection_id:
                selected_reflection = reflection_manager.get_reflection(selected_reflection_id)
                if selected_reflection:
                    # 反省の詳細を表示
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**タスク:**")
                        st.info(selected_reflection.task)
                        
                        st.markdown(f"**コンテキスト:**")
                        st.json(selected_reflection.context)
                        
                        st.markdown(f"**判断:**")
                        st.success(f"{selected_reflection.judgment.result}: {selected_reflection.judgment.reason}")
                        
                        if selected_reflection.judgment.suggested_approach:
                            st.markdown(f"**推奨アプローチ:**")
                            st.warning(selected_reflection.judgment.suggested_approach)
                    
                    with col2:
                        st.markdown(f"**結果:**")
                        st.info(selected_reflection.result[:500] + "..." if len(selected_reflection.result) > 500 else selected_reflection.result)
                        
                        st.markdown(f"**反省:**")
                        st.info(selected_reflection.reflection[:500] + "..." if len(selected_reflection.reflection) > 500 else selected_reflection.reflection)
            
            # 反省の活用状況
            st.subheader("反省の活用状況")
            st.markdown("""
            これらの反省データは、次のように活用されています：
            
            1. 新しいタスクを実行する際に、関連する過去の反省が検索されます
            2. 過去の反省に基づいて、タスクの進め方が改善されます
            3. 同様の課題が発生した場合に、より効率的な解決策が提案されます
            """)
            
            # タグベースのフィルタリング
            unique_tags = set()
            for reflection in all_reflections:
                unique_tags.update(reflection.tags)
            
            if unique_tags:
                st.subheader("タグでフィルタリング")
                selected_tag = st.selectbox(
                    "タグを選択",
                    options=["すべて表示"] + sorted(list(unique_tags))
                )
                
                if selected_tag != "すべて表示":
                    filtered_reflections = [r for r in all_reflections if selected_tag in r.tags]
                    
                    if filtered_reflections:
                        # フィルタリングされた反省データをテーブル表示用に変換
                        filtered_data = []
                        for reflection in filtered_reflections:
                            filtered_data.append({
                                "ID": reflection.id,
                                "タイムスタンプ": reflection.timestamp,
                                "タスク": reflection.task,
                                "判断": reflection.judgment.result,
                                "再試行": "必要" if reflection.judgment.needs_retry else "不要"
                            })
                        
                        st.dataframe(pd.DataFrame(filtered_data), use_container_width=True)
                    else:
                        st.info(f"タグ '{selected_tag}' を持つ反省データはありません。")
    
    
    # 既存のタブコードを続ける...
    with tab_deepdive:
        st.header("課題深掘りエージェント - ROIツリー探索")
        st.markdown("""
        このエージェントは、的確な質問によってROIツリーの構築と探索をサポートします。
        ビジネス効果の様々な側面を探り、重要度係数を持つ階層的なツリーとして整理します。
        エージェントとのチャットを通じて、課題を深掘りしていきましょう。
        """)
        
        # 必要に応じて課題深掘り状態を初期化
        if DEEPDIVE_STATE_KEY not in st.session_state:
            deepdive_state = init_deepdive_state()
            deepdive_state["min_nodes_per_branch"] = min_nodes
            deepdive_state["max_iterations"] = max_iterations_deepdive
            st.session_state[DEEPDIVE_STATE_KEY] = deepdive_state
        else:
            # 設定を更新
            st.session_state[DEEPDIVE_STATE_KEY]["min_nodes_per_branch"] = min_nodes
            st.session_state[DEEPDIVE_STATE_KEY]["max_iterations"] = max_iterations_deepdive
            
        state = st.session_state[DEEPDIVE_STATE_KEY]
        
        # チャットと可視化のためのカラムを作成
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("チャット")
            chat_container = st.container()
            
            # メッセージを表示（チャット形式）
            display_messages(state["messages"], chat_container)
            
            # ストリーミング出力用のコンテナを作成
            stream_container = st.empty()
            
            # 入力処理用のフラグを初期化
            if "process_deepdive_input" not in st.session_state:
                st.session_state.process_deepdive_input = False
                st.session_state.deepdive_input_value = ""
                
            # 送信ボタンをクリックしたときの処理
            def submit_deepdive_input():
                st.session_state.process_deepdive_input = True
                st.session_state.deepdive_input_value = st.session_state.deepdive_input
                
            # ユーザー入力フィールド
            st.text_input("メッセージを入力してください:", key="deepdive_input", on_change=submit_deepdive_input)
            
            # 入力が処理待ちの場合
            if st.session_state.process_deepdive_input:
                user_input = st.session_state.deepdive_input_value
                
                # フラグをリセット
                st.session_state.process_deepdive_input = False
                st.session_state.deepdive_input_value = ""
                
                # ユーザーメッセージを追加
                user_message = HumanMessage(content=user_input)
                state["messages"].append(user_message)
                st.session_state[DEEPDIVE_STATE_KEY] = state
                
                # 会話をすぐに表示（レスポンス生成前）
                chat_container.chat_message("user").markdown(user_input)
                
                # グラフを作成
                deepdive_graph = build_deepdive_graph()
                
                with get_openai_callback() as cb:
                    # レスポンスをストリーミング
                    stream_handler = StreamHandler(stream_container)
                    
                    try:
                        # エージェントの応答を追加
                        new_state = run_one_step(deepdive_graph, state, stream_handler)
                        
                        # 最新の状態をセッションに保存
                        st.session_state[DEEPDIVE_STATE_KEY] = new_state
                        
                        # トークン使用量を表示
                        st.caption(f"使用トークン: {cb.total_tokens} (¥{cb.total_cost*130:.2f})")
                    
                        # 探索が完了したかチェック
                        if new_state["exploration_complete"]:
                            if new_state["self_reflection"]:
                                reason = new_state["self_reflection"].reason
                                st.success(f"探索完了！理由: {reason}")
                            else:
                                st.success("探索完了！")
                        
                        # UIを更新するためにページを再読み込み
                        st.rerun()
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
        
        with col2:
            st.subheader("ROIツリー")
            tree_container = st.container()
            
            # ツリーを表示
            if state["root_node"]:
                display_roi_tree(state["root_node"], tree_container)
            else:
                tree_container.info("まだROIツリーが作成されていません。")
            
            # 自己反省を表示
            if state["self_reflection"]:
                st.subheader("自己分析")
                st.markdown(f"**完了率**: {state['self_reflection'].deepdive_completion_percentage:.0f}%")
                st.markdown(f"**分析**: {state['self_reflection'].reason}")
                
                if state["self_reflection"].suggested_focus:
                    st.markdown(f"**推奨フォーカス**: {state['self_reflection'].suggested_focus}")
    
    # =========================
    # 提案エージェントタブ
    # =========================
    with tab_proposal:
        st.header("提案エージェント - ROI計算")
        st.markdown("""
        このエージェントは、課題深掘りフェーズで作成したROIツリーに基づいてROIを計算します。
        各コンポーネントを分析し、価値を見積もり、全体的なROIを計算します。
        エージェントとのチャットを通じて、具体的な提案とROI試算を作成していきましょう。
        """)
        
        # 課題深掘りからツリーがあるかチェック
        if DEEPDIVE_STATE_KEY not in st.session_state or not st.session_state[DEEPDIVE_STATE_KEY]["root_node"]:
            st.warning("まず、課題深掘りフェーズを完了してROIツリーを作成してください。")
        else:
            # 課題深掘りからROIツリーを取得
            root_node = st.session_state[DEEPDIVE_STATE_KEY]["root_node"]
            
            # 必要に応じて提案状態を初期化
            if PROPOSAL_STATE_KEY not in st.session_state:
                proposal_state = init_proposal_state(root_node)
                proposal_state["max_iterations"] = max_iterations_proposal
                st.session_state[PROPOSAL_STATE_KEY] = proposal_state
            else:
                # 設定を更新
                st.session_state[PROPOSAL_STATE_KEY]["max_iterations"] = max_iterations_proposal
            
            state = st.session_state[PROPOSAL_STATE_KEY]
            
            # チャットと可視化のためのカラムを作成
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # チャット履歴コンテナ
                chat_container = st.container()
                
                # メッセージを表示
                display_messages(state["messages"], chat_container)
                
                # ストリーミング出力用のコンテナを作成
                stream_container = st.empty()
                
                # 入力処理用のフラグを初期化
                if "process_proposal_input" not in st.session_state:
                    st.session_state.process_proposal_input = False
                    st.session_state.proposal_input_value = ""
                
            # 送信ボタンをクリックしたときの処理
            def submit_proposal_input():
                st.session_state.process_proposal_input = True
                st.session_state.proposal_input_value = st.session_state.proposal_input
                
            # ユーザー入力フィールド
            st.text_input("メッセージを入力してください:", key="proposal_input", on_change=submit_proposal_input)
            
            # 入力が処理待ちの場合
            if st.session_state.process_proposal_input:
                user_input = st.session_state.proposal_input_value
                
                # フラグをリセット
                st.session_state.process_proposal_input = False
                st.session_state.proposal_input_value = ""
                
                # ユーザーメッセージを追加
                state["messages"].append(HumanMessage(content=user_input))
                st.session_state[PROPOSAL_STATE_KEY] = state
                    
                    # グラフを作成
                proposal_graph = build_proposal_graph()
                    
                with get_openai_callback() as cb:
                    # レスポンスをストリーミング
                    stream_handler = StreamHandler(stream_container)
                    
                    try:
                        # エージェントの応答を追加
                        new_state = run_one_step(proposal_graph, state, stream_handler)
                        
                        # 最新の状態をセッションに保存
                        st.session_state[PROPOSAL_STATE_KEY] = new_state
                        
                        # トークン使用量を表示
                        st.caption(f"使用トークン: {cb.total_tokens} (¥{cb.total_cost*130:.2f})")
                    
                        # 提案が完了したかチェック
                        if new_state["proposal_complete"]:
                            if new_state["self_reflection"]:
                                reason = new_state["self_reflection"].reason
                                st.success(f"提案完了！理由: {reason}")
                            else:
                                st.success("提案完了！")
                        
                        # UIを更新するためにページを再読み込み
                        st.rerun()
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
            
            with col2:
                st.subheader("ROI計算")
                roi_container = st.container()
                
                # ROI計算を表示
                display_roi_calculations(state["roi_calculations"], roi_container)
                
                # 自己反省を表示
                if state["self_reflection"]:
                    st.subheader("自己分析")
                    st.markdown(f"**ROI信頼度**: {state['self_reflection'].roi_confidence:.0%}")
                    st.markdown(f"**分析**: {state['self_reflection'].reason}")
                    
                    if state["self_reflection"].missing_information:
                        st.markdown("**不足している情報**:")
                        for item in state["self_reflection"].missing_information:
                            st.markdown(f"- {item}")


# Run the app
if __name__ == "__main__":
    main()