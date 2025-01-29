import streamlit as st
import json
import os
import mermaid as md
from mermaid.graph import Graph
import streamlit.components.v1 as components
import re

###############################################################################
# Utilities for displaying Mermaid code via `mermaid` library
###############################################################################

def normalize_mermaid_code(mermaid_code: str) -> str:
    """
    Mermaidコード中の行頭インデントを自動調整して、パーサがエラーを起こしにくい形に整える。
    """
    lines = mermaid_code.splitlines()
    # 空行を除いた最小インデントを求める
    min_indent = None
    for line in lines:
        if line.strip() == "":
            continue
        leading_spaces = len(line) - len(line.lstrip())
        if min_indent is None or leading_spaces < min_indent:
            min_indent = leading_spaces

    # 各行から min_indent 分だけ左に寄せる
    if min_indent and min_indent > 0:
        new_lines = []
        for line in lines:
            # 行が空の場合はそのまま
            if line.strip() == "":
                new_lines.append("")
            else:
                new_lines.append(line[min_indent:])
        return "\n".join(new_lines)
    else:
        return mermaid_code

def render_mermaid_diagram(code: str, diagram_title: str = "MermaidDiagram"):
    """
    与えられた Mermaid 'code' を HTML に変換し、Streamlit 上で表示する。
    1) インデントを正規化してパーサのエラーを回避
    2) 先にMermaidコードをベタ書きで表示
    3) MermaidコードをHTMLとして表示 (mermaid ライブラリを使用)
    4) もし解析失敗したら、Mermaidコードをマークダウンブロックとして直接表示
    """
    # 1) インデントなどを正規化
    normalized_code = normalize_mermaid_code(code)

    # 2) 先に「元のMermaidコード」をベタ書き表示
    st.markdown("#### ▼ Mermaidコード（ベタ書き）")
    st.code(normalized_code, language='mermaid')

    try:
        # 3) MermaidライブラリでHTMLを生成し、表示
        graph = Graph(diagram_title, normalized_code)
        rendered = md.Mermaid(graph)
        mermaid_html = rendered._repr_html_()

        components.html(
            mermaid_html,
            height=800,  # 必要に応じて調整
            scrolling=True
        )

    except Exception as e:
        # 4) 失敗した場合は Mermaid コードをマークダウンブロックとして出す
        st.warning(f"Mermaid解析に失敗しました (理由: {e}). Mermaidコードを直接表示します。")
        st.markdown(f"```mermaid\n{normalized_code}\n```")


def parse_mermaid_node_labels(mermaid_code: str) -> dict:
    """
    Mermaidコードから node -> label の対応を正規表現で抽出して返す。
    例: CR_A1[作業時間短縮(約150時間/月)] → { "CR_A1": "作業時間短縮(約150時間/月)" }
         SCR_A1 --> SCR_A1_1[ピッキング工程の自動化]
         なども拾う。
    """
    pattern = r"([A-Za-z0-9_]+)\[([^\]]+)\]"
    node_label_map = {}
    for match in re.finditer(pattern, mermaid_code):
        node = match.group(1).strip()
        label = match.group(2).strip()
        node_label_map[node] = label
    return node_label_map

def parse_mermaid_edges(mermaid_code: str) -> dict:
    """
    Mermaidコードから `nodeA --> nodeB` または `nodeA->nodeB` の形を抽出して
    adjacency (隣接リスト) を返す。
    """
    edge_pattern = r"([A-Za-z0-9_]+)\s*-+>\s*([A-Za-z0-9_]+)"
    adjacency = {}
    for match in re.finditer(edge_pattern, mermaid_code):
        parent = match.group(1).strip()
        child = match.group(2).strip()
        adjacency.setdefault(parent, []).append(child)
        if child not in adjacency:
            adjacency[child] = []
    return adjacency

###############################################################################
# 子ノードごとに1本のスライダーを用意し、合計を親factorに正規化する方式
###############################################################################
def render_hierarchical_sliders(
    node: str,
    adjacency: dict,
    node_label_map: dict,
    base_key: str,
    parent_factor: float = 1.0,
    level: int = 0
):
    """
    指定された node を起点に、子ノードへの配分を行いながら階層構造を表示。
    """
    label = node_label_map.get(node, node)
    children = adjacency.get(node, [])

    prefix = "    " * level  # 階層表示のためのインデント

    # 親ノード（現在のnode）を表示
    st.write(f"{prefix}**{label}**: {parent_factor:.2f}")

    if not children:
        # 子がなければリーフなので再帰終了
        return

    # 子ノード分だけスライダーを表示
    slider_holder_key = f"{base_key}_{node}_slider_values"
    if slider_holder_key not in st.session_state:
        n = len(children)
        init = [round(1.0 / n, 2) for _ in range(n)] if n > 0 else [0.0]
        st.session_state[slider_holder_key] = init

    current_values = st.session_state[slider_holder_key]
    updated_values = []

    for i, child_node in enumerate(children):
        child_label = node_label_map.get(child_node, child_node)
        child_slider_key = f"{slider_holder_key}_{i}"

        val = st.slider(
            f"{prefix} └ 子ノード '{child_label}' の割合（0～1）",
            min_value=0.0,
            max_value=1.0,
            value=float(current_values[i]),
            step=0.01,
            key=child_slider_key,
        )
        updated_values.append(val)

    st.session_state[slider_holder_key] = updated_values

    total_slider = sum(updated_values)
    for i, child_node in enumerate(children):
        if total_slider > 0:
            child_factor = parent_factor * (updated_values[i] / total_slider)
        else:
            if len(children) == 1:
                child_factor = parent_factor
            else:
                child_factor = 0.0

        render_hierarchical_sliders(
            node=child_node,
            adjacency=adjacency,
            node_label_map=node_label_map,
            base_key=base_key,
            parent_factor=child_factor,
            level=level+1
        )

###############################################################################
# ユニークキー付きウィジェット
###############################################################################
def get_radio_value(label: str, options: list, state_key: str) -> str:
    """
    ラジオボタン。セッションステートに値を保存・取得する。
    """
    default_value = st.session_state["annotations"].get(state_key, options[-1])  # 例: "未評価"
    if default_value in options:
        idx = options.index(default_value)
    else:
        idx = 0
    selected = st.radio(label, options, index=idx, key=state_key)
    st.session_state["annotations"][state_key] = selected
    return selected

def get_text_area_value(label: str, state_key: str) -> str:
    """
    テキストエリア。セッションステートに値を保存・取得する。
    """
    default_text = st.session_state["annotations"].get(state_key, "")
    text = st.text_area(label, value=default_text, key=state_key)
    st.session_state["annotations"][state_key] = text
    return text

###############################################################################
# ROI算定
###############################################################################
def annotate_roi(file_idx: int, proj_idx: int, project: dict):
    st.subheader("■ ROI算定評価")
    if "ROI算定" in project["table"]:
        st.markdown("**ROI算定（原文）:**")
        for item in project["table"]["ROI算定"]:
            st.write(f"- {item}")

    base_key = f"file{file_idx}_proj{proj_idx}_roi"
    roi_good_or_bad_key = base_key + "_good_or_bad"
    roi_comment_key = base_key + "_comment"

    get_radio_value("ROI算定は良い？悪い？", ["良い", "悪い", "未評価"], roi_good_or_bad_key)
    get_text_area_value("どこが悪いか（自由記述）", roi_comment_key)

###############################################################################
# Q&A (Assignment / Suggest)
###############################################################################
def annotate_q_and_a(file_idx: int, proj_idx: int, q_and_a_dict: dict, qa_type: str = "assignment"):
    st.subheader(f"■ Q&A評価 ({qa_type})")

    for depth_key, qa_list in q_and_a_dict.items():
        st.markdown(f"### {depth_key}")

        for qa_item_idx, qa_item in enumerate(qa_list):
            parent = qa_item.get("parentNode", "")
            child = qa_item.get("childNode", "")
            if parent or child:
                if parent and child:
                    st.markdown(f"**{parent} → {child}**")
                elif child:
                    st.markdown(f"**{child}**")

            questions = qa_item.get("questions", [])
            for q_idx, q_item in enumerate(questions):
                question = q_item.get("question", "")
                answer = q_item.get("answer", "")

                # chat UI (Streamlit 1.31+ 以降)
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)

                base_key = f"file{file_idx}_proj{proj_idx}_{qa_type}_QAndA_{depth_key}_{qa_item_idx}_{q_idx}"
                qa_good_or_bad_key = base_key + "_good_or_bad"
                qa_comment_key = base_key + "_comment"

                get_radio_value(
                    f"このQ&Aは良い？悪い？ ( {depth_key}, {qa_item_idx}番目, 質問{q_idx} )",
                    ["良い", "悪い", "未評価"],
                    qa_good_or_bad_key
                )
                get_text_area_value(
                    f"どこが悪い？ ( {depth_key}, {qa_item_idx}番目, 質問{q_idx} )",
                    qa_comment_key
                )

###############################################################################
# ROIツリー (Assignment / Suggest)
###############################################################################
def annotate_roi_trees(
    file_idx: int,
    proj_idx: int,
    roi_trees_dict: dict,
    tree_type: str = "assignment"
):
    st.subheader(f"■ ROIツリー評価 ({tree_type})")

    for depth_key, tree_data in roi_trees_dict.items():
        st.markdown(f"### {depth_key}")

        # 取得した mermaid_code
        mermaid_code = tree_data.get("graph", "")
        # 必要に応じて全角置換など
        mermaid_code = mermaid_code.replace("(", "（").replace(")", "）")

        if not mermaid_code:
            st.warning(f"{depth_key} には 'graph' がありません。")
            continue

        # Mermaid グラフ表示
        render_mermaid_diagram(mermaid_code, diagram_title=f"{tree_type}_{depth_key}")

        # ノードラベルと adjacency の抽出
        node_label_map = parse_mermaid_node_labels(mermaid_code)
        adjacency = parse_mermaid_edges(mermaid_code)

        # adjacency の「親がいないノード」をルートとみなす
        all_children = set()
        for parent, children in adjacency.items():
            for c in children:
                all_children.add(c)
        roots = [node for node in adjacency.keys() if node not in all_children]
        if not roots:
            st.info("ルートノードが見つかりませんでした。")
            continue

        st.write("#### 階層スライダーで子ノードに配分")
        for root in roots:
            base_key_for_root = f"file{file_idx}_proj{proj_idx}_{tree_type}_roiTrees_{depth_key}_{root}"
            render_hierarchical_sliders(
                node=root,
                adjacency=adjacency,
                node_label_map=node_label_map,
                base_key=base_key_for_root,
                parent_factor=1.0,
                level=0
            )

        # 良い/悪い + コメント (ツリー全体に対する評価)
        base_key2 = f"file{file_idx}_proj{proj_idx}_{tree_type}_roiTrees_{depth_key}"
        good_or_bad_key = f"{base_key2}_good_or_bad"
        comment_key = f"{base_key2}_comment"
        get_radio_value(f"{depth_key} は良い？悪い？", ["良い", "悪い", "未評価"], good_or_bad_key)
        get_text_area_value(f"{depth_key} のどこが悪いか（自由記述）", comment_key)

###############################################################################
# Main
###############################################################################
def main():
    st.title("複数JSONファイルのDX Projects アノテーションツール (各子ノードにスライダー配置版)")

    if "annotations" not in st.session_state:
        st.session_state["annotations"] = {}

    uploaded_files = st.file_uploader(
        "アノテーション対象の JSON ファイルをアップロードしてください",
        type="json",
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("JSONファイルをアップロードしてください。")
        st.stop()

    for file_idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        st.markdown("---")
        st.markdown(f"## ファイル: `{file_name}`")

        try:
            data = json.loads(uploaded_file.read().decode("utf-8"))
        except Exception as e:
            st.error(f"JSONの読み込み中にエラーが発生しました: {e}")
            continue

        if "DXProjects" not in data:
            st.error(f"ファイル {file_name} に 'DXProjects' キーが見つかりません。スキップします。")
            continue

        for proj_idx, project in enumerate(data["DXProjects"]):
            company_name = project["table"].get("企業名", f"Unknown_{proj_idx}")
            purpose = project["table"].get("課題・目的", "不明な課題")

            with st.expander(f"[{company_name}] / 課題: {purpose}", expanded=False):
                # 1) ROI 算定
                annotate_roi(file_idx, proj_idx, project)

                # 2) ROIツリー (assignment / suggest)
                for mode in ["assignment", "suggest"]:
                    roiTree_keys = [k for k in project.keys() if k.startswith(f"roiTrees_{mode}")]
                    if roiTree_keys:
                        for rkey in roiTree_keys:
                            st.markdown(f"### ROIツリー: {rkey}")
                            if isinstance(project[rkey], dict):
                                annotate_roi_trees(file_idx, proj_idx, project[rkey], mode)
                            else:
                                st.warning(f"'{rkey}' の形式が想定と異なります。（dictを期待）")
                    else:
                        st.warning(f"この企業に '{mode}' 系の ROIツリーキーがありません。")

                # 3) Q&A (assignment / suggest)
                for mode in ["assignment", "suggest"]:
                    qa_keys = [k for k in project.keys() if k.startswith(f"QAndA_{mode}")]
                    if qa_keys:
                        for qkey in qa_keys:
                            st.markdown(f"### Q&A: {qkey}")
                            if isinstance(project[qkey], dict):
                                annotate_q_and_a(file_idx, proj_idx, project[qkey], mode)
                            else:
                                st.warning(f"'{qkey}' の形式が想定と異なります。（dictを期待）")
                    else:
                        st.warning(f"この企業に '{mode}' 系の Q&Aキーがありません。")

                # 保存ボタン（必要に応じて）
                save_button_key = f"save_btn_file{file_idx}_proj{proj_idx}"
                if st.button(f"『{company_name}』の評価を保存", key=save_button_key):
                    st.success(f"『{company_name}』の評価結果を保存しました（サンプル）")
                    # ここでJSONとして保存などの処理を行う

if __name__ == "__main__":
    main()
