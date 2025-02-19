# roi_agents/agents/deepdive_agent.py

import os
from typing import Literal

# langgraph関連
from langgraph.graph import StateGraph, END

# langchain関連
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from domain.schemas import DeepdiveState
from domain.roitree import ROINode, print_roi_tree
from dotenv import load_dotenv

load_dotenv()  # .envファイルを読み込みます

api_key = os.getenv("OPENAI_API_KEY")

# Chatモデルの初期化
# gpt-4 が使えない場合は model="gpt-3.5-turbo" などへ変更
model = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0.5,
    streaming=False,  # streamlitでの表示はmain.py側で実装
)

# --------------------------------------
# ノードを深掘りする処理
# --------------------------------------
def deepdive_conversation(state: DeepdiveState) -> DeepdiveState:
    """
    ユーザと対話し、ROIツリーにノードを追加 or 更新する。
    """
    system_prompt = SystemMessage(content="""
あなたは課題深掘りアシスタントです。
ユーザと対話しながら、課題をROI観点(CostReduction, RevenueIncreaseなど)で細分化し、
最終的にツリー状に整理します。
十分に深掘りできたと判断するまで、ユーザに問いかけを行ってください。
深掘りが不十分な場合は、さらに追加の質問を行ってください。
最終的に深掘りが完了したら、その時点でタスクを終了してください。
""")

    user_prompt = f"""\
現在のROIトップノード: {state["root_node"].name if state["root_node"] else "未設定"}

あなたは「課題エージェント」です。ROIツリーをより詳細なノードへ分解するための会話を進めてください。
どのような観点でさらに深掘りすべきか、まずはユーザに質問してください。
"""

    ai_message = model([system_prompt, *state["messages"], HumanMessage(content=user_prompt)])

    # 会話ログに追加
    state["messages"].append(HumanMessage(content=user_prompt))
    state["messages"].append(ai_message)

    # ダミー: ノードの実際の追加・削除はしていない。
    return state


# --------------------------------------
# セルフリフレクション: 深掘りを継続するか判断
# --------------------------------------
def self_reflect_deepdive(state: DeepdiveState) -> DeepdiveState:
    """
    例では iteration_count >= max_iterations で終了判定。
    """
    state["iteration_count"] += 1
    if state["iteration_count"] >= state["max_iterations"]:
        state["deepdive_needed"] = False
    else:
        state["deepdive_needed"] = True
    return state

# --------------------------------------
# 継続判断
# --------------------------------------
def should_continue_deepdive(state: DeepdiveState) -> Literal["continue", "__end__"]:
    return "continue" if state["deepdive_needed"] else "__end__"

# --------------------------------------
# StateGraph構築
# --------------------------------------
def build_deepdive_graph():
    """
    Deepdive用のStateGraphを構築して返す。
    """
    graph = StateGraph(DeepdiveState)

    # ステートマシンのノード登録
    graph.add_node("deepdive_conversation", deepdive_conversation)
    graph.add_node("self_reflect_deepdive", self_reflect_deepdive)

    # エッジを定義
    graph.add_edge("deepdive_conversation", "self_reflect_deepdive")
    graph.add_conditional_edges(
        "self_reflect_deepdive",
        should_continue_deepdive,
        {"continue": "deepdive_conversation", "__end__": END}
    )

    # エントリーポイントを設定
    graph.set_entry_point("deepdive_conversation")

    return graph.compile()

def get_default_deepdive_state() -> DeepdiveState:
    """
    DeepdiveState を初期化して返すユーティリティ。
    iteration_count や max_iterations などのキーを必ず含める。
    """
    return DeepdiveState(
        messages=[],
        root_node=ROINode("Gain", details="ROIツリーのトップノード"),
        current_node=None,
        iteration_count=0,
        max_iterations=3,
        deepdive_needed=True
    )
