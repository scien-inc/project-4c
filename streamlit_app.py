import streamlit as st
import os
from typing import Dict, Any
from langgraph.graph import StateGraph

from agents.deepdive_agent import build_deepdive_graph, get_default_deepdive_state
from agents.proposal_agent import build_proposal_graph
from domain.schemas import DeepdiveState, ProposalState
from domain.roitree import ROINode

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ["LANGCHAIN_HANDLER"] = "langchain"

class StreamlitCallbackHandler(BaseCallbackHandler):
    """langchainのストリーム出力をStreamlitに表示するためのハンドラ例。"""
    def __init__(self, container):
        self.container = container
        self.text_so_far = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text_so_far += token
        self.container.markdown(self.text_so_far)

    def on_llm_end(self, response, **kwargs) -> None:
        self.text_so_far = ""

# ============================================
# DeepdiveAgent 実行関数
# ============================================
def run_deepdive_agent():
    deepdive_app = build_deepdive_graph()
    initial_state = get_default_deepdive_state()

    # deepdive_app.stream(initial_state) はイテレーションごとに
    # "AddableUpdatesDict" オブジェクトを返す（.state などの属性は無い）
    for step_event in deepdive_app.stream(initial_state):
        # step_event は "AddableUpdatesDict"
        # なので直接 step_event['iteration_count'] や step_event['messages'] を参照可能
        # ただし必要に応じて下記のように dict(...) へ変換してもOK
        current_state = dict(step_event)  # これで通常のdict化

        yield current_state  # これで current_state["iteration_count"] など呼べる


def run_proposal_agent(root_node: ROINode):
    proposal_app = build_proposal_graph()
    initial_state: ProposalState = {
        "messages": [],
        "root_node": root_node,
        "roi_calculations": {},
        "iteration_count": 0,
        "max_iterations": 2,
        "proposal_complete": False
    }
    for step_event in proposal_app.stream(initial_state):
        current_state = dict(step_event)
        yield current_state


# ============================================
# Streamlit アプリ本体
# ============================================
def main():
    st.set_page_config(page_title="ROI Agents Demo", layout="wide")
    st.title("ROI Agents (Deepdive & Proposal)")

    st.header("1. 課題エージェント(Deepdive)")
    deepdive_container = st.empty()

    if st.button("Run Deepdive Agent"):
        with deepdive_container:
            st.write("**Deepdive Agent Starting...**")

        for step_idx, state in enumerate(run_deepdive_agent()):
            # ここで state はすでに dict(step_event) されたもの
            with deepdive_container:
                # iteration_count があれば参照可能
                st.markdown(f"**[Step {step_idx}]** iteration_count={state['iteration_count']}")
                # 会話ログを表示
                for msg in state["messages"]:
                    if isinstance(msg, HumanMessage):
                        st.markdown(f"**User:** {msg.content}")
                    elif isinstance(msg, AIMessage):
                        st.markdown(f"**Assistant:** {msg.content}")

        st.success("Deepdive Agent Finished.")
        st.session_state["deepdive_state"] = state

    st.header("2. 提案エージェント(Proposal)")
    proposal_container = st.empty()

    if st.button("Run Proposal Agent"):
        if "deepdive_state" not in st.session_state:
            st.warning("先にDeepdive Agentを実行してください。")
        else:
            root_node = st.session_state["deepdive_state"]["root_node"]
            cost_node = ROINode("CostReduction", details="Demo: メインのコスト削減")
            rev_node = ROINode("RevenueIncrease", details="Demo: メインの売上増")
            root_node.add_child(cost_node)
            root_node.add_child(rev_node)

            with proposal_container:
                st.write("**Proposal Agent Starting...**")

            for step_idx, p_state in enumerate(run_proposal_agent(root_node)):
                with proposal_container:
                    st.markdown(f"**[Step {step_idx}]** iteration_count={p_state['iteration_count']}")
                    for msg in p_state["messages"]:
                        if isinstance(msg, HumanMessage):
                            st.markdown(f"**User:** {msg.content}")
                        elif isinstance(msg, AIMessage):
                            st.markdown(f"**Assistant:** {msg.content}")

            st.success("Proposal Agent Finished.")
            st.write("**最終ROI結果**:", p_state["roi_calculations"].get("summary"))


if __name__ == "__main__":
    main()
