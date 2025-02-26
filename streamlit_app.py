# roi_agents/main.py

import streamlit as st
import os 

from langgraph.graph import StateGraph
from langchain.callbacks import get_openai_callback

from agents.deepdive_agent import build_deepdive_graph
from agents.proposal_agent import build_proposal_graph
from domain.schemas import DeepdiveState, ProposalState
from domain.roitree import ROINode

"""
Streamlit アプリとして「Deepdive Agent」と「Proposal Agent」をタブで切り替えて体験できる例。
"""


DEEPDIVE_STATE_KEY = "deepdive_state"
PROPOSAL_STATE_KEY = "proposal_state"

def init_deepdive_state():
    """DeepdiveState の初期化"""
    return DeepdiveState(
        messages=[],
        root_node=ROINode("Gain", details="トップノード"),
        current_node=None,
        iteration_count=0,
        max_iterations=3,
        deepdive_needed=True
    )

def init_proposal_state(root_node: ROINode):
    """ProposalState の初期化"""
    return ProposalState(
        messages=[],
        root_node=root_node,
        roi_calculations={},
        iteration_count=0,
        max_iterations=3,
        proposal_complete=False
    )

def run_one_step(app, current_state):
    """
    langgraphの stream() はジェネレータ。
    1回だけイテレータから取り出して返すことで「1ステップだけ進行」させる。
    """
    gen = app.stream(current_state)
    try:
        new_state = next(gen)
        return new_state
    except StopIteration:
        return current_state

def main():
    st.title("ROI Agents Demo")

    tab_deepdive, tab_proposal = st.tabs(["Deepdive Agent", "Proposal Agent"])

    with tab_deepdive:
        st.header("Deepdive Agent")
        if DEEPDIVE_STATE_KEY not in st.session_state:
            st.session_state[DEEPDIVE_STATE_KEY] = init_deepdive_state()

        state = st.session_state[DEEPDIVE_STATE_KEY]
        deepdive_graph = build_deepdive_graph()

        st.subheader("Chat Messages")
        for msg in state["messages"]:
            if msg.type == "human":
                st.markdown(f"**User:** {msg.content}")
            else:
                st.markdown(f"**Agent:** {msg.content}")

        # 「次へ」ボタンで1ステップ進める
        if st.button("次へ（1ステップ進行）"):
            with get_openai_callback() as cb:
                try:
                    new_state = run_one_step(deepdive_graph, state)
                    st.session_state[DEEPDIVE_STATE_KEY] = new_state
                    st.info("1ステップ進行しました。")
                    st.write(f"Tokens used: {cb.total_tokens}")
                except StopIteration:
                    st.warning("すでに対話が終了しているようです。")

        st.write("---")
        st.subheader("現在のROIツリー構造")

        if state["root_node"] is not None:
            def _render_node(node: ROINode, indent=0):
                prefix = "&nbsp;" * (indent * 4)
                st.markdown(f"{prefix}- **{node.name}** (details: {node.details})")
                for c in node.children:
                    _render_node(c, indent+1)

            _render_node(state["root_node"], 0)

    with tab_proposal:
        st.header("Proposal Agent")

        if DEEPDIVE_STATE_KEY not in st.session_state or st.session_state[DEEPDIVE_STATE_KEY]["root_node"] is None:
            st.warning("先にDeepdive AgentでROIツリーを構築してください。")
        else:
            if PROPOSAL_STATE_KEY not in st.session_state:
                root_node = st.session_state[DEEPDIVE_STATE_KEY]["root_node"]
                st.session_state[PROPOSAL_STATE_KEY] = init_proposal_state(root_node)

            proposal_state = st.session_state[PROPOSAL_STATE_KEY]
            proposal_graph = build_proposal_graph()

            st.subheader("Proposal Chat Messages")
            for msg in proposal_state["messages"]:
                if msg.type == "human":
                    st.markdown(f"**User:** {msg.content}")
                else:
                    st.markdown(f"**Agent:** {msg.content}")

            if st.button("次へ（1ステップ進行）", key="proposal_step"):
                with get_openai_callback() as cb:
                    try:
                        new_state = run_one_step(proposal_graph, proposal_state)
                        st.session_state[PROPOSAL_STATE_KEY] = new_state
                        st.info("Proposal: 1ステップ進行しました。")
                        st.write(f"Tokens used: {cb.total_tokens}")
                    except StopIteration:
                        st.warning("提案対話がすでに終了しています。")

            st.write("---")
            st.subheader("ROI試算結果")
            roi_calc = proposal_state["roi_calculations"]
            if roi_calc:
                st.json(roi_calc)


if __name__ == "__main__":
    main()
