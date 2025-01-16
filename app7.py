import json
import streamlit as st
import os

JSON_FILE_PATH = "data.json"

def load_json_from_file(file_path: str):
    if not os.path.exists(file_path):
        st.error(f"ファイルが見つかりません: {file_path}")
        st.stop()
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if "annotations" not in st.session_state:
    st.session_state["annotations"] = {}

###############################################################################
# ラジオボタンとテキストエリアそれぞれに unique な key をつけるための
# ヘルパー関数を用意
###############################################################################
def get_radio_value(label: str, options: list, state_key: str) -> str:
    default_value = st.session_state["annotations"].get(state_key, options[-1])  # 未評価などをデフォルト
    # index は options の中で default_value が何番目かを探す
    idx = options.index(default_value) if default_value in options else 0
    selected = st.radio(label, options, index=idx, key=state_key)
    st.session_state["annotations"][state_key] = selected
    return selected

def get_text_area_value(label: str, state_key: str) -> str:
    default_text = st.session_state["annotations"].get(state_key, "")
    text = st.text_area(label, value=default_text, key=state_key)
    st.session_state["annotations"][state_key] = text
    return text

###############################################################################
# 修正した annotate_roi: ROI算定を画面に表示しつつ、評価できるようにする
###############################################################################
def annotate_roi(index: int, project: dict):
    """
    ROI算定部分の評価を記録するUI。
    """
    st.subheader("■ ROI算定評価")

    # --- ここで data.json の "ROI算定" 配列を表示 ---
    if "ROI算定" in project["table"]:
        st.markdown("**ROI算定（原文）:**")
        for item in project["table"]["ROI算定"]:
            st.write(f"- {item}")

    # 評価用ウィジェット
    roi_key_good_or_bad = f"roi_good_or_bad_{index}"
    roi_key_comment = f"roi_comment_{index}"

    get_radio_value("ROI算定は良い？悪い？", ["良い", "悪い", "未評価"], roi_key_good_or_bad)
    get_text_area_value("どこが悪いか（自由記述）", roi_key_comment)


def annotate_roi_trees(index: int, roi_trees: dict):
    st.subheader("■ ROIツリー評価")
    for depth_key, mermaid_code in roi_trees.items():
        st.markdown(f"### {depth_key}")
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

        unique_radio_key = f"roi_tree_good_or_bad_{index}_{depth_key}"
        get_radio_value(
            f"{depth_key} は良い？悪い？", 
            ["良い", "悪い", "未評価"], 
            unique_radio_key
        )
        unique_text_key = f"roi_tree_comment_{index}_{depth_key}"
        get_text_area_value(f"{depth_key} のどこが悪いか（自由記述）", unique_text_key)


def annotate_q_and_a(index: int, q_and_a: dict):
    st.subheader("■ Q&A評価")
    for depth_key, qa_list in q_and_a.items():
        st.markdown(f"### {depth_key}")
        for qa_item_idx, qa_item in enumerate(qa_list):
            parent = qa_item["parentNode"]
            child = qa_item["childNode"]
            st.markdown(f"**{parent} → {child}**")
            questions = qa_item["questions"]
            for q_idx, q_item in enumerate(questions):
                question = q_item["question"]
                answer = q_item["answer"]

                st.write(f"**Q:** {question}")
                st.write(f"**A:** {answer}")

                qa_good_or_bad_key = f"qa_good_or_bad_{index}_{depth_key}_{qa_item_idx}_{q_idx}"
                get_radio_value(
                    f"このQ&Aは良い？悪い？ ( {depth_key}, {qa_item_idx}番目, 質問{q_idx} )",
                    ["良い", "悪い", "未評価"],
                    qa_good_or_bad_key
                )
                qa_comment_key = f"qa_comment_{index}_{depth_key}_{qa_item_idx}_{q_idx}"
                get_text_area_value(
                    f"どこが悪い？ ( {depth_key}, {qa_item_idx}番目, 質問{q_idx} )",
                    qa_comment_key
                )


###############################################################################
# メイン処理
###############################################################################
st.title("DX Projects アノテーションツール")

# 1) ローカルファイルから JSON を読み込む
data = load_json_from_file(JSON_FILE_PATH)

# 2) JSONの DXProjects を順に表示
for i, project in enumerate(data["DXProjects"]):
    with st.expander(f"企業名: {project['table']['企業名']} / 課題: {project['table']['課題・目的']}"):
        # 修正した annotate_roi へ project を渡す
        annotate_roi(i, project)
        annotate_roi_trees(i, project["roiTrees"])
        annotate_q_and_a(i, project["QAndA"])

# 3) 保存ボタン
if st.button("評価結果を保存"):
    # アノテーション結果を JSON に反映してファイルに書き戻す
    for i, project in enumerate(data["DXProjects"]):
        # ROI算定
        roi_good_or_bad_key = f"roi_good_or_bad_{i}"
        roi_comment_key = f"roi_comment_{i}"

        project.setdefault("annotation", {})
        project["annotation"]["ROI評価"] = {
            "良いor悪い": st.session_state["annotations"].get(roi_good_or_bad_key, "未評価"),
            "コメント": st.session_state["annotations"].get(roi_comment_key, "")
        }

        # ROIツリー
        tree_annotation = {}
        for depth_key in project["roiTrees"].keys():
            good_or_bad_key = f"roi_tree_good_or_bad_{i}_{depth_key}"
            comment_key = f"roi_tree_comment_{i}_{depth_key}"
            tree_annotation[depth_key] = {
                "良いor悪い": st.session_state["annotations"].get(good_or_bad_key, "未評価"),
                "コメント": st.session_state["annotations"].get(comment_key, "")
            }
        project["annotation"]["roiTrees評価"] = tree_annotation

        # Q&A
        q_and_a_annotation = {}
        for depth_key, qa_list in project["QAndA"].items():
            depth_evals = []
            for qa_item_idx, qa_item in enumerate(qa_list):
                questions = qa_item["questions"]
                question_evals = []
                for q_idx, _ in enumerate(questions):
                    qa_good_or_bad_key = f"qa_good_or_bad_{i}_{depth_key}_{qa_item_idx}_{q_idx}"
                    qa_comment_key = f"qa_comment_{i}_{depth_key}_{qa_item_idx}_{q_idx}"
                    question_evals.append({
                        "良いor悪い": st.session_state["annotations"].get(qa_good_or_bad_key, "未評価"),
                        "コメント": st.session_state["annotations"].get(qa_comment_key, "")
                    })
                depth_evals.append(question_evals)
            q_and_a_annotation[depth_key] = depth_evals

        project["annotation"]["QAndA評価"] = q_and_a_annotation

    # JSONを保存用文字列に
    annotated_json_str = json.dumps(data, ensure_ascii=False, indent=2)

    # ローカルファイルに上書き保存
    with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(annotated_json_str)

    st.success(f"評価結果を {JSON_FILE_PATH} に保存しました。")

    st.download_button(
        label="評価済みJSONをダウンロード",
        data=annotated_json_str.encode("utf-8"),
        file_name="annotated_DXProjects.json",
        mime="application/json"
    )
