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
    2) MermaidのHTMLを表示したあと、元のMermaidコードも下にベタ書きで表示
    3) もし解析失敗したら、Mermaidコードをマークダウンブロックとして直接表示
    """
    # 1) インデントなどを正規化
    normalized_code = normalize_mermaid_code(code)

    try:
        # 2) MermaidライブラリでHTML生成
        graph = Graph(diagram_title, normalized_code)
        rendered = md.Mermaid(graph)
        mermaid_html = rendered._repr_html_()

        # HTML(描画済みMermaid)を表示
        components.html(
            mermaid_html,
            height=800,  # 必要に応じて調整 (深い階層も見やすいよう大きめに)
            scrolling=True
        )

        # ▼ ここで「元のMermaidコード」を下にベタ書き表示する
        st.markdown("#### ▼ Mermaidコード（ベタ書き）")
        st.code(normalized_code, language='mermaid')

    except Exception as e:
        st.warning(f"Mermaid解析に失敗しました (理由: {e}). Mermaidコードを直接表示します。")
        # 失敗した場合は Mermaid コードをマークダウンコードブロックとして出す
        st.markdown(f"```mermaid\n{normalized_code}\n```")

def parse_mermaid_node_labels(mermaid_code: str) -> dict:
    """
    Mermaidコードから node -> label の対応を正規表現で抽出して返す。
    例: CR_A1[作業時間短縮(約150時間/月)] → { "CR_A1": "作業時間短縮(約150時間/月)" }
         SCR_A1 --> SCR_A1_1[ピッキング工程の自動化]
         なども拾う。
    """
    # ノード定義は "Something[...]"" の形を想定し、矢印などがあっても拾えるように調整
    pattern = r"([A-Za-z0-9_]+)\[([^\]]+)\]"
    node_label_map = {}
    for match in re.finditer(pattern, mermaid_code):
        node = match.group(1).strip()
        label = match.group(2).strip()
        node_label_map[node] = label
    return node_label_map

###############################################################################
# ユニークキー付きウィジェット（DuplicateWidgetID対策）
###############################################################################
def get_radio_value(label: str, options: list, state_key: str) -> str:
    """
    ラジオボタンを表示。セッションステートに値を保存・取得する。
    """
    default_value = st.session_state["annotations"].get(state_key, options[-1])  # 例: "未評価"をデフォルト
    if default_value in options:
        idx = options.index(default_value)
    else:
        idx = 0
    selected = st.radio(label, options, index=idx, key=state_key)
    st.session_state["annotations"][state_key] = selected
    return selected

def get_text_area_value(label: str, state_key: str) -> str:
    """
    テキストエリアを表示。セッションステートに値を保存・取得する。
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
# importance_factor を「スライダー（区切り）で調整」するためのヘルパー
###############################################################################
def render_importance_factor_sliders(base_key: str, importance_factors: list, node_label_map: dict):
    """
    同じ depth (階層) 内の importance_factors を編集するUI。
    - importance_factors: [{'node': str, 'importance_factor': float}, ...]
    - 基本的に先頭要素を親ノードとみなし、以降の子ノードだけで「合計1」を割り振る。
    - (子ノード数 - 1) 本のスライダーを並べ、区切り位置として解釈し、比率配分する。
    - node_label_map: mermaidコードから抽出した node → 日本語ラベル のマップ。
    """
    if not importance_factors:
        st.warning("importance_factors が空です。")
        return importance_factors

    # 先頭要素を親と仮定 (例えば 'CostReduction': 1 など)
    parent_node = importance_factors[0]
    child_nodes = importance_factors[1:]  # 残りを子ノードとする

    # 親ノードのラベル表示
    parent_name = parent_node["node"]
    parent_label = node_label_map.get(parent_name, parent_name)
    parent_factor = parent_node.get("importance_factor", 1.0)
    st.write(f"**Parent Node**: {parent_label} = {parent_factor} (固定想定)")

    if len(child_nodes) == 0:
        # 子ノードがない場合は何もしない
        st.info("子ノードがありません。")
        return importance_factors

    if len(child_nodes) == 1:
        # 子ノードが1つだけなら強制的に1
        single_node_name = child_nodes[0]["node"]
        single_node_label = node_label_map.get(single_node_name, single_node_name)
        st.write(f"- {single_node_label} は1.0（自動設定）")
        child_nodes[0]['importance_factor'] = 1.0
        return importance_factors

    # 子ノードが2つ以上ある場合は、(子ノード数 - 1) 本のスライダーで区切りを決める
    n_child = len(child_nodes)
    n_sliders = n_child - 1

    # セッションステートでスライダーの値を管理
    slider_state_key = base_key + "_importance_sliders"
    if slider_state_key not in st.session_state:
        # 初期値: 子ノードの importance_factor の累積和に相当する区切りを作る
        # 例: [0.5, 0.5] → スライダーは [0.5] 1本
        # 例: [0.3, 0.2, 0.5] → スライダーは [0.3, 0.5]
        sums = []
        accum = 0.0
        for c in child_nodes[:-1]:
            accum += c["importance_factor"]
            sums.append(round(accum, 2))
        st.session_state[slider_state_key] = sums

    current_sliders = st.session_state[slider_state_key]

    updated_sliders = []
    for i in range(n_sliders):
        slider_label = f"区切りスライダー {i+1}/{n_sliders}"
        current_val = current_sliders[i] if i < len(current_sliders) else 0.0
        val = st.slider(
            slider_label,
            min_value=0.0,
            max_value=1.0,
            value=float(current_val),
            step=0.01,
            key=f"{slider_state_key}_slider_{i}"
        )
        updated_sliders.append(val)

    updated_sliders.sort()
    st.session_state[slider_state_key] = updated_sliders

    # 分配率を計算
    child_importances = []
    prev = 0.0
    for i, node_data in enumerate(child_nodes):
        if i < n_sliders:
            portion = updated_sliders[i] - prev
            prev = updated_sliders[i]
        else:
            portion = 1.0 - prev
        portion = max(portion, 0.0)
        child_importances.append(round(portion, 2))

    for i, node_data in enumerate(child_nodes):
        node_data["importance_factor"] = child_importances[i]

    st.write("#### 子ノードの割り振り結果")
    for c in child_nodes:
        node_name = c["node"]
        label = node_label_map.get(node_name, node_name)
        st.write(f"- {label}: **{c['importance_factor']}**")

    return [parent_node] + child_nodes

###############################################################################
# ROIツリー (Assignment / Suggest) - Mermaid Preview & importance_factor editing
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

        # 取得した mermaid_code に対して () を全角に置換する（修正箇所）
        mermaid_code = tree_data.get("graph", "")
        # ここで半角の "(" と ")" を全角 "（" と "）" に変換
        mermaid_code = mermaid_code.replace("(", "（").replace(")", "）")

        if not mermaid_code:
            st.warning(f"{depth_key} には 'graph' がありません。")
            continue

        # Mermaid グラフ表示
        render_mermaid_diagram(mermaid_code, diagram_title=f"{tree_type}_{depth_key}")

        # Mermaidコードから node -> label 対応表を作成
        node_label_map = parse_mermaid_node_labels(mermaid_code)

        # importance_factors の編集UI
        if "importance_factors" in tree_data:
            base_key = f"file{file_idx}_proj{proj_idx}_{tree_type}_roiTrees_{depth_key}"
            st.write("#### importance_factors の編集")
            updated_factors = render_importance_factor_sliders(
                base_key,
                tree_data["importance_factors"],
                node_label_map
            )
            # UIで更新された値を反映
            tree_data["importance_factors"] = updated_factors
        else:
            st.warning(f"{depth_key} に 'importance_factors' キーがありません。")

        # 良い/悪い + コメント
        good_or_bad_key = f"{base_key}_good_or_bad"
        comment_key = f"{base_key}_comment"
        get_radio_value(f"{depth_key} は良い？悪い？", ["良い", "悪い", "未評価"], good_or_bad_key)
        get_text_area_value(f"{depth_key} のどこが悪いか（自由記述）", comment_key)

###############################################################################
# Q&A (Assignment / Suggest) - Chat-like UI
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

                # chat UI (Streamlit 1.31+)
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
# Main
###############################################################################
def main():
    st.title("複数JSONファイルのDX Projects アノテーションツール (Mermaid + importance_factor対応版)")

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

                # 保存ボタン
                save_button_key = f"save_btn_file{file_idx}_proj{proj_idx}"
                if st.button(f"『{company_name}』の評価を保存", key=save_button_key):
                    project.setdefault("annotation", {})

                    # ROI評価
                    roi_good_or_bad_key = f"file{file_idx}_proj{proj_idx}_roi_good_or_bad"
                    roi_comment_key = f"file{file_idx}_proj{proj_idx}_roi_comment"
                    project["annotation"]["ROI評価"] = {
                        "良いor悪い": st.session_state["annotations"].get(roi_good_or_bad_key, "未評価"),
                        "コメント": st.session_state["annotations"].get(roi_comment_key, "")
                    }

                    # ROIツリー評価: assignment / suggest
                    project["annotation"]["roiTrees評価"] = {}
                    for mode in ["assignment", "suggest"]:
                        rkeys = [k for k in project.keys() if k.startswith(f"roiTrees_{mode}")]
                        mode_dict = {}
                        for rkey in rkeys:
                            subtree = project[rkey]
                            if isinstance(subtree, dict):
                                sub_dict = {}
                                for depth_key, depth_data in subtree.items():
                                    base_key = f"file{file_idx}_proj{proj_idx}_{mode}_roiTrees_{depth_key}"
                                    gkey = f"{base_key}_good_or_bad"
                                    ckey = f"{base_key}_comment"
                                    imp_factors = depth_data.get("importance_factors", [])
                                    sub_dict[depth_key] = {
                                        "良いor悪い": st.session_state["annotations"].get(gkey, "未評価"),
                                        "コメント": st.session_state["annotations"].get(ckey, ""),
                                        "importance_factors": imp_factors
                                    }
                                mode_dict[rkey] = sub_dict
                            else:
                                st.warning(f"保存処理: '{rkey}' が想定の形式(dict)ではありません。")
                        if mode_dict:
                            project["annotation"]["roiTrees評価"][mode] = mode_dict

                    # QAndA評価: assignment / suggest
                    project["annotation"]["QAndA評価"] = {}
                    for mode in ["assignment", "suggest"]:
                        qkeys = [k for k in project.keys() if k.startswith(f"QAndA_{mode}")]
                        qa_mode_dict = {}
                        for qkey in qkeys:
                            qa_dict = project[qkey]
                            if isinstance(qa_dict, dict):
                                depth_evals = {}
                                for depth_key, qa_list in qa_dict.items():
                                    qa_items_eval = []
                                    for qa_item_idx, qa_item in enumerate(qa_list):
                                        questions_eval = []
                                        for q_idx, _q_item in enumerate(qa_item.get("questions", [])):
                                            base_key = f"file{file_idx}_proj{proj_idx}_{mode}_QAndA_{depth_key}_{qa_item_idx}_{q_idx}"
                                            good_bad = st.session_state["annotations"].get(base_key + "_good_or_bad", "未評価")
                                            comment = st.session_state["annotations"].get(base_key + "_comment", "")
                                            questions_eval.append({
                                                "良いor悪い": good_bad,
                                                "コメント": comment
                                            })
                                        qa_items_eval.append(questions_eval)
                                    depth_evals[depth_key] = qa_items_eval
                                qa_mode_dict[qkey] = depth_evals
                            else:
                                st.warning(f"保存処理: '{qkey}' が想定の形式(dict)ではありません。")
                        if qa_mode_dict:
                            project["annotation"]["QAndA評価"][mode] = qa_mode_dict

                    # 最終的にこの1社だけ書き出し
                    single_project_data = {
                        "DXProjects": [project]
                    }

                    base_root, base_ext = os.path.splitext(file_name)
                    safe_company_name = company_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                    out_filename = f"{base_root}_{safe_company_name}_annotated.json"

                    try:
                        with open(out_filename, "w", encoding="utf-8") as out_f:
                            json.dump(single_project_data, out_f, ensure_ascii=False, indent=2)
                        st.success(f"『{company_name}』の評価結果をローカルファイル '{out_filename}' に保存しました。")
                    except Exception as e:
                        st.warning(f"ローカルへの保存でエラーが発生しました: {e}")

                    download_data = json.dumps(single_project_data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="評価結果(この1社分)をダウンロード",
                        data=download_data.encode("utf-8"),
                        file_name=out_filename,
                        mime="application/json"
                    )

if __name__ == "__main__":
    main()
