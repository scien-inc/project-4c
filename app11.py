import streamlit as st
import json
import os
import mermaid as md
from mermaid.graph import Graph
import streamlit.components.v1 as components

###############################################################################
# Utilities for displaying Mermaid code via `mermaid` Python library
###############################################################################
def render_mermaid_diagram(code: str, diagram_title: str = "MermaidDiagram"):
    """
    Given a Mermaid 'code' string, render it in Streamlit using the `mermaid` library.
    """
    # Create a Graph object from the code
    # The first parameter (e.g. 'Sequence-diagram') is just an internal title;
    # it doesn't necessarily have to match the type of diagram unless the syntax needs it.
    graph = Graph(diagram_title, code)
    # Wrap in the Mermaid object
    rendered = md.Mermaid(graph)
    
    # Convert to HTML
    mermaid_html = rendered._repr_html_()
    
    # Use streamlit's HTML component to display
    # Adjust height, scrolling, etc. as desired
    components.html(
        mermaid_html,
        height=400,  # You can adjust this
        scrolling=True
    )

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
# ROIツリー (Assignment / Suggest) - Mermaid Preview using `mermaid` library
###############################################################################
def annotate_roi_trees(
    file_idx: int,
    proj_idx: int,
    roi_trees_dict: dict,
    tree_type: str = "assignment"
):
    st.subheader(f"■ ROIツリー評価 ({tree_type})")

    for depth_key, mermaid_code in roi_trees_dict.items():
        st.markdown(f"### {depth_key}")
        # --- Here we call our custom render function ---
        render_mermaid_diagram(mermaid_code, diagram_title=f"{tree_type}_{depth_key}")

        base_key = f"file{file_idx}_proj{proj_idx}_{tree_type}_roiTrees_{depth_key}"
        good_or_bad_key = base_key + "_good_or_bad"
        comment_key = base_key + "_comment"

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
            if parent and child:
                st.markdown(f"**{parent} → {child}**")

            questions = qa_item["questions"]
            for q_idx, q_item in enumerate(questions):
                question = q_item["question"]
                answer = q_item["answer"]

                # Display in chat style (if Streamlit >= 1.31).
                # Otherwise, you can simply do:
                # st.write("**User:**", question)
                # st.write("**Assistant:**", answer)
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
st.title("複数JSONファイルのDX Projects アノテーションツール (Mermaidプレビュー版)")

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

            # 2) ROIツリー: assignment
            if "roiTrees_assignment" in project:
                annotate_roi_trees(file_idx, proj_idx, project["roiTrees_assignment"], "assignment")
            else:
                st.warning("この企業のデータに 'roiTrees_assignment' がありません。")

            # 2') ROIツリー: suggest
            if "roiTrees_suggest" in project:
                annotate_roi_trees(file_idx, proj_idx, project["roiTrees_suggest"], "suggest")
            else:
                st.warning("この企業のデータに 'roiTrees_suggest' がありません。")

            # 3) Q&A: assignment
            if "QAndA_assignment" in project:
                annotate_q_and_a(file_idx, proj_idx, project["QAndA_assignment"], "assignment")
            else:
                st.warning("この企業のデータに 'QAndA_assignment' がありません。")

            # 3') Q&A: suggest
            if "QAndA_suggest" in project:
                annotate_q_and_a(file_idx, proj_idx, project["QAndA_suggest"], "suggest")
            else:
                st.warning("この企業のデータに 'QAndA_suggest' がありません。")

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

                # ROIツリー assignment / suggest
                project["annotation"]["roiTrees評価"] = {}
                for mode in ["assignment", "suggest"]:
                    if f"roiTrees_{mode}" in project:
                        mode_dict = {}
                        for depth_key in project[f"roiTrees_{mode}"].keys():
                            base_key = f"file{file_idx}_proj{proj_idx}_{mode}_roiTrees_{depth_key}"
                            good_or_bad_key = base_key + "_good_or_bad"
                            comment_key = base_key + "_comment"
                            mode_dict[depth_key] = {
                                "良いor悪い": st.session_state["annotations"].get(good_or_bad_key, "未評価"),
                                "コメント": st.session_state["annotations"].get(comment_key, "")
                            }
                        project["annotation"]["roiTrees評価"][mode] = mode_dict

                # QAndA評価 assignment / suggest
                project["annotation"]["QAndA評価"] = {}
                for mode in ["assignment", "suggest"]:
                    if f"QAndA_{mode}" in project:
                        q_and_a_annotation = {}
                        for depth_key, qa_list in project[f"QAndA_{mode}"].items():
                            depth_evals = []
                            for qa_item_idx, qa_item in enumerate(qa_list):
                                question_evals = []
                                for q_idx in range(len(qa_item["questions"])):
                                    base_key = f"file{file_idx}_proj{proj_idx}_{mode}_QAndA_{depth_key}_{qa_item_idx}_{q_idx}"
                                    qa_good_or_bad_key = base_key + "_good_or_bad"
                                    qa_comment_key = base_key + "_comment"
                                    question_evals.append({
                                        "良いor悪い": st.session_state["annotations"].get(qa_good_or_bad_key, "未評価"),
                                        "コメント": st.session_state["annotations"].get(qa_comment_key, "")
                                    })
                                depth_evals.append(question_evals)
                            q_and_a_annotation[depth_key] = depth_evals
                        project["annotation"]["QAndA評価"][mode] = q_and_a_annotation

                # 最終的に「この1社だけ」書き出し
                single_project_data = {
                    "DXProjects": [project]
                }

                base_root, base_ext = os.path.splitext(file_name)
                safe_company_name = company_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                out_filename = f"{base_root}_{safe_company_name}_annotated.json"

                # ローカル保存（任意）
                try:
                    with open(out_filename, "w", encoding="utf-8") as out_f:
                        json.dump(single_project_data, out_f, ensure_ascii=False, indent=2)
                    st.success(f"『{company_name}』の評価結果をローカルファイル '{out_filename}' に保存しました。")
                except Exception as e:
                    st.warning(f"ローカルへの保存でエラーが発生しました: {e}")

                # ダウンロードボタン
                download_data = json.dumps(single_project_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="評価結果(この1社分)をダウンロード",
                    data=download_data.encode("utf-8"),
                    file_name=out_filename,
                    mime="application/json"
                )
