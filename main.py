# roi_agents/main.py

import os
from typing import Dict, Any
from langgraph.graph import StateGraph
from langchain.callbacks import get_openai_callback

from agents.deepdive_agent import build_deepdive_graph
from agents.proposal_agent import build_proposal_graph
from domain.schemas import DeepdiveState, ProposalState
from domain.roitree import ROINode

def run_deepdive_agent():
    # DeepdiveエージェントのStateGraph構築
    app = build_deepdive_graph()

    # 初期状態
    initial_state: DeepdiveState = {
        "messages": [],
        "root_node": ROINode("Gain", details="ROIツリーのトップノード"),
        "current_node": None,  # まだ何も設定していない
        "iteration_count": 0,
        "max_iterations": 3,
        "deepdive_needed": True
    }

    with get_openai_callback() as cb:
        # ストリーミング実行
        for s in app.stream(initial_state, config={"recursion_limit": 20}):
            # 途中経過をコンソール等に出力
            print(s)
            print("---------")

        print("Deepdive Agent Done.")
        print(f"Tokens used: {cb.total_tokens}")

    return initial_state

def run_proposal_agent(root_node: ROINode):
    # ProposalエージェントのStateGraph構築
    app = build_proposal_graph()

    # 初期状態
    initial_state: ProposalState = {
        "messages": [],
        "root_node": root_node,  # deepdive結果のROIツリーを参照
        "roi_calculations": {},
        "iteration_count": 0,
        "max_iterations": 3,
        "proposal_complete": False
    }

    with get_openai_callback() as cb:
        for s in app.stream(initial_state, config={"recursion_limit": 20}):
            print(s)
            print("---------")

        print("Proposal Agent Done.")
        print(f"Tokens used: {cb.total_tokens}")

    return initial_state


def main():
    # 1. 課題エージェントを起動し、ROIツリーを深掘りする
    deepdive_state = run_deepdive_agent()

    # 2. deepdiveが終わったら、ROIツリーができている想定
    #   （実際にはdeepdive内でノード追加しているはずだが、ここでは簡略化）
    #   例えばroot_node以下にCostReduction, RevenueIncreaseがあると想定
    cost_node = ROINode("CostReduction", details="メインのコスト削減項目")
    rev_node = ROINode("RevenueIncrease", details="メインの売上増項目")
    deepdive_state["root_node"].add_child(cost_node)
    deepdive_state["root_node"].add_child(rev_node)

    # 3. 提案エージェントを起動し、ROI試算を行う
    proposal_state = run_proposal_agent(deepdive_state["root_node"])

    # 4. 結果表示
    print("最終的な提案ROI: ", proposal_state["roi_calculations"].get("summary"))


if __name__ == "__main__":
    main()
