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
    """
    normalized_code = normalize_mermaid_code(code)
    st.markdown("#### ▼ Mermaidコード（ベタ書き）")
    st.code(normalized_code, language='mermaid')

    try:
        graph = Graph(diagram_title, normalized_code)
        rendered = md.Mermaid(graph)
        mermaid_html = rendered._repr_html_()

        components.html(
            mermaid_html,
            height=800,
            scrolling=True
        )

    except Exception as e:
        st.warning(f"Mermaid解析に失敗しました (理由: {e}). Mermaidコードを直接表示します。")
        st.markdown(f"```mermaid\n{normalized_code}\n```")

def parse_mermaid_node_labels(mermaid_code: str) -> dict:
    """
    Mermaidコードから node -> label の対応を抽出。
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
    Mermaidコードから `nodeA --> nodeB` の形を抽出して隣接リストを返す。
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
# 階層スライダー表示（※変更箇所）
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
    指定ノードから下位へ、各子ノードへの重要度（importance_factor）の配分を行います。
    ※今回の変更点：
      ・「フォークリフト移動削減」および「メンテナンスコスト削減」は、すでに深さ3で評価済みとみなし、
        親スライダー（例：「コスト削減」のスライダー）で配分済みとします。
      ・そのため、これらのノードは【スライダーUIの対象から除外】し、代わりにそれぞれの子ノード
        （例：「ムダ移動経路再配置」「待機時間削減」や「タイヤ交換整備費見直し」）に対して、通常のスライダーUIを表示します。
    """
    indent = "    " * level
    label = node_label_map.get(node, node)
    st.markdown(f"{indent}**ノード: {label}**  |  重要度: **{parent_factor:.2f}**")
    st.session_state["annotations"][f"{base_key}_{node}_factor"] = parent_factor
    st.session_state["annotations"][f"{base_key}_{node}_label"] = label

    children = adjacency.get(node, [])
    if not children:
        return

    # ---【ここから新規処理】---
    # スライダーUIの対象から除外するラベル（深さ3で既に評価済みのもの）
    skip_slider_labels = {"フォークリフト移動削減", "メンテナンスコスト削減"}
    non_skip_children = []
    skip_children = []
    for child in children:
        child_label = node_label_map.get(child, child)
        if child_label in skip_slider_labels:
            skip_children.append(child)
        else:
            non_skip_children.append(child)
    # ---【ここまで新規処理】---

    # ※ 非対象（＝スライダー表示対象）の子ノードがある場合は、通常通りスライダーUIで配分
    if non_skip_children:
        if len(non_skip_children) == 1:
            new_values = [1.0]
            st.markdown(f"{indent}子ノード（{node_label_map.get(non_skip_children[0], non_skip_children[0])}）は1つのため、重みは自動で1.0となります。")
        elif len(non_skip_children) == 2:
            slider_key = f"{base_key}_{node}_custom_slider_nonskip"
            slider_val = st.slider(
                f"{indent}子ノード配分（{label} の一部）",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                key=slider_key,
            )
            new_values = [slider_val, 1 - slider_val]
            bar_html = f"""
            <div style="width:100%; height:20px; background: linear-gradient(to right, red 0%, red {slider_val*100}%, blue {slider_val*100}%, blue 100%);"></div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
        elif len(non_skip_children) == 3:
            slider_key = f"{base_key}_{node}_custom_slider_nonskip"
            slider_range = st.slider(
                f"{indent}子ノード配分（{label} の一部）",
                min_value=0.0,
                max_value=1.0,
                value=(round(1/3, 2), round(2/3, 2)),
                step=0.01,
                key=slider_key,
            )
            new_values = [slider_range[0], slider_range[1] - slider_range[0], 1 - slider_range[1]]
            first_pct = slider_range[0] * 100
            second_pct = (slider_range[1] - slider_range[0]) * 100
            bar_html = f"""
            <div style="width:100%; height:20px; background: linear-gradient(
                to right,
                red 0%, red {first_pct}%,
                yellow {first_pct}% , yellow {first_pct+second_pct}%,
                blue {first_pct+second_pct}%, blue 100%
                );"></div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
        else:
            new_values = []
            for i, child in enumerate(non_skip_children):
                child_label = node_label_map.get(child, child)
                child_slider_key = f"{base_key}_{node}_slider_values_nonskip_{i}"
                val = st.slider(
                    f"{indent}子ノード: **{child_label}** の割り当て (0～1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0 / len(non_skip_children),
                    step=0.01,
                    key=child_slider_key,
                )
                new_values.append(val)
        # 計算結果を表示＆セッションステートへ保存
        total_slider = sum(new_values)
        for i, child in enumerate(non_skip_children):
            child_label = node_label_map.get(child, child)
            if total_slider > 0:
                ratio = new_values[i] / total_slider
            else:
                ratio = 1.0 / len(non_skip_children)
            child_factor = parent_factor * ratio
            ratio_key = f"{base_key}_{node}_child_{child}_ratio"
            st.session_state["annotations"][ratio_key] = ratio
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"{indent}子ノード **{child_label}**: 割合 **{ratio:.2f}**  |  重要度 **{child_factor:.2f}**")
            with col2:
                st.progress(int(child_factor * 100))
            render_hierarchical_sliders(
                node=child,
                adjacency=adjacency,
                node_label_map=node_label_map,
                base_key=base_key,
                parent_factor=child_factor,
                level=level+1
            )
    # ※ スキップ対象（＝既に深さ3で評価済み）の子ノードについては、ここではスライダーUIを準備せず、
    #     直接再帰呼び出しする（後続で、その子ノードが持つ子供に対しては通常のスライダーUIを表示します）。
    for child in skip_children:
        render_hierarchical_sliders(
            node=child,
            adjacency=adjacency,
            node_label_map=node_label_map,
            base_key=base_key,
            parent_factor=parent_factor,
            level=level+1
        )

###############################################################################
# ユニークキー付きウィジェット
###############################################################################
def get_radio_value(label: str, options: list, state_key: str) -> str:
    default_value = st.session_state["annotations"].get(state_key, options[-1])
    if default_value in options:
        idx = options.index(default_value)
    else:
        idx = 0
    selected = st.radio(label, options, index=idx, key=state_key)
    st.session_state["annotations"][state_key] = selected
    return selected

def get_text_area_value(label: str, state_key: str) -> str:
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

        mermaid_code = tree_data.get("graph", "")
        mermaid_code = mermaid_code.replace("(", "（").replace(")", "）")

        if not mermaid_code:
            st.warning(f"{depth_key} には 'graph' がありません。")
            continue

        render_mermaid_diagram(mermaid_code, diagram_title=f"{tree_type}_{depth_key}")
        node_label_map = parse_mermaid_node_labels(mermaid_code)
        adjacency = parse_mermaid_edges(mermaid_code)

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

        base_key2 = f"file{file_idx}_proj{proj_idx}_{tree_type}_roiTrees_{depth_key}"
        good_or_bad_key = f"{base_key2}_good_or_bad"
        comment_key = f"{base_key2}_comment"
        get_radio_value(f"{depth_key} は良い？悪い？", ["良い", "悪い", "未評価"], good_or_bad_key)
        get_text_area_value(f"{depth_key} のどこが悪いか（自由記述）", comment_key)

def extract_annotations_for_project(file_idx: int, proj_idx: int) -> dict:
    prefix = f"file{file_idx}_proj{proj_idx}"
    result = {}
    for k, v in st.session_state["annotations"].items():
        if k.startswith(prefix):
            result[k] = v
    return result

###############################################################################
# Main
###############################################################################
def main():
    st.title("複数JSONファイルのDX Projects アノテーションツール")
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
                annotate_roi(file_idx, proj_idx, project)

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

                save_button_key = f"save_btn_file{file_idx}_proj{proj_idx}"
                if st.button(f"『{company_name}』の評価を保存", key=save_button_key):
                    st.success(f"『{company_name}』の評価結果を保存しました（サンプル）")
                    st.session_state[f"show_download_{file_idx}_{proj_idx}"] = True

                if st.session_state.get(f"show_download_{file_idx}_{proj_idx}", False):
                    proj_annotations = extract_annotations_for_project(file_idx, proj_idx)
                    proj_json_str = json.dumps(
                        {
                            "file_idx": file_idx,
                            "proj_idx": proj_idx,
                            "company_name": company_name,
                            "annotations": proj_annotations
                        },
                        ensure_ascii=False,
                        indent=2
                    )
                    st.download_button(
                        label=f"『{company_name}』の評価をJSONでダウンロード",
                        data=proj_json_str,
                        file_name=f"{company_name}_annotations.json",
                        mime="application/json",
                        key=f"download_btn_file{file_idx}_{proj_idx}"
                    )

    st.markdown("## 全ファイル・プロジェクトに対するアノテーション結果のダウンロード")
    download_json = json.dumps(st.session_state["annotations"], ensure_ascii=False, indent=2)
    st.download_button(
        label="すべてのアノテーション結果をダウンロード (JSON)",
        data=download_json,
        file_name="annotations_all.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
