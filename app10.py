import streamlit as st
import json
import os

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
# ROI算定のみ（テキスト評価）
###############################################################################
def annotate_roi(file_idx: int, proj_idx: int, project: dict):
    """
    ROI算定部分のアノテーションUI。
    """
    st.subheader("■ ROI算定評価")

    # data.jsonから読み込んだ ROI算定 の内容を表示
    if "ROI算定" in project["table"]:
        st.markdown("**ROI算定（原文）:**")
        for item in project["table"]["ROI算定"]:
            st.write(f"- {item}")

    # 評価用ウィジェット（良い/悪い、コメント）
    base_key = f"file{file_idx}_proj{proj_idx}_roi"
    roi_good_or_bad_key = base_key + "_good_or_bad"
    roi_comment_key = base_key + "_comment"

    get_radio_value("ROI算定は良い？悪い？", ["良い", "悪い", "未評価"], roi_good_or_bad_key)
    get_text_area_value("どこが悪いか（自由記述）", roi_comment_key)


###############################################################################
# ROIツリー（Assignment / Suggest 共通）をアノテーションする
###############################################################################
def annotate_roi_trees(
    file_idx: int,
    proj_idx: int,
    roi_trees_dict: dict,
    tree_type: str = "assignment"
):
    """
    ROIツリー（Mermaid）のアノテーションUI。
    tree_type: "assignment" or "suggest" などを区別
    """
    st.subheader(f"■ ROIツリー評価 ({tree_type})")

    for depth_key, mermaid_code in roi_trees_dict.items():
        st.markdown(f"### {depth_key}")
        # Mermaidコードをテキストで表示（静的プレビュー）
        # Note: This requires Streamlit to support mermaid markdown rendering:
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

        base_key = f"file{file_idx}_proj{proj_idx}_{tree_type}_roiTrees_{depth_key}"
        good_or_bad_key = base_key + "_good_or_bad"
        comment_key = base_key + "_comment"

        get_radio_value(f"{depth_key} は良い？悪い？", ["良い", "悪い", "未評価"], good_or_bad_key)
        get_text_area_value(f"{depth_key} のどこが悪いか（自由記述）", comment_key)


###############################################################################
# Q&A（Assignment / Suggest 共通）をアノテーションする (チャット風UI)
###############################################################################
def annotate_q_and_a(
    file_idx: int,
    proj_idx: int,
    q_and_a_dict: dict,
    qa_type: str = "assignment"
):
    """
    Q&A (Depth3, Depth4, Depth5) へのアノテーションUI。チャットUI風に表示。

    qa_type: "assignment" or "suggest" を区別
    """
    st.subheader(f"■ Q&A評価 ({qa_type})")

    # 各 Depth レベルを順に処理
    for depth_key, qa_list in q_and_a_dict.items():
        st.markdown(f"### {depth_key}")

        for qa_item_idx, qa_item in enumerate(qa_list):
            # parent / child
            parent = qa_item.get("parentNode", "")
            child = qa_item.get("childNode", "")

            if parent and child:
                st.markdown(f"**{parent} → {child}**")

            # Q&A の一覧
            questions = qa_item["questions"]
            for q_idx, q_item in enumerate(questions):
                question = q_item["question"]
                answer = q_item["answer"]

                # チャット風に表示 (Requires Streamlit >= 1.31 for st.chat_message)
                # If your version doesn't support this, just use st.write("User: ..."); st.write("Assistant: ...")
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)

                # 良い/悪い の評価
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
# メイン処理
###############################################################################

st.title("複数JSONファイルのDX Projects アノテーションツール")

# セッションステート初期化
if "annotations" not in st.session_state:
    st.session_state["annotations"] = {}

# 1) 複数JSONファイルをアップロード
uploaded_files = st.file_uploader(
    "アノテーション対象の JSON ファイルをアップロードしてください",
    type="json",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("JSONファイルをアップロードしてください。")
    st.stop()

# 2) アップロードされたファイルを順に処理
for file_idx, uploaded_file in enumerate(uploaded_files):
    file_name = uploaded_file.name
    st.markdown("---")
    st.markdown(f"## ファイル: `{file_name}`")

    # JSONの読み込み
    try:
        data = json.loads(uploaded_file.read().decode("utf-8"))
    except Exception as e:
        st.error(f"JSONの読み込み中にエラーが発生しました: {e}")
        continue

    # DXProjects が無ければスキップ
    if "DXProjects" not in data:
        st.error(f"ファイル {file_name} に 'DXProjects' キーが見つかりません。スキップします。")
        continue

    # ファイルに含まれるプロジェクト(企業)を順に表示
    for proj_idx, project in enumerate(data["DXProjects"]):
        company_name = project["table"].get("企業名", f"Unknown_{proj_idx}")
        purpose = project["table"].get("課題・目的", "不明な課題")

        with st.expander(f"[{company_name}] / 課題: {purpose}", expanded=False):
            # 企業ごとのアノテーションUI

            # ROI算定（単純テキスト評価）
            annotate_roi(file_idx, proj_idx, project)

            # ROIツリー (assignment)
            if "roiTrees_assignment" in project:
                annotate_roi_trees(file_idx, proj_idx, project["roiTrees_assignment"], tree_type="assignment")
            else:
                st.warning("この企業のデータに 'roiTrees_assignment' がありません。")

            # ROIツリー (suggest)
            if "roiTrees_suggest" in project:
                annotate_roi_trees(file_idx, proj_idx, project["roiTrees_suggest"], tree_type="suggest")
            else:
                st.warning("この企業のデータに 'roiTrees_suggest' がありません。")

            # QAndA (assignment)
            if "QAndA_assignment" in project:
                annotate_q_and_a(file_idx, proj_idx, project["QAndA_assignment"], qa_type="assignment")
            else:
                st.warning("この企業のデータに 'QAndA_assignment' がありません。")

            # QAndA (suggest)
            if "QAndA_suggest" in project:
                annotate_q_and_a(file_idx, proj_idx, project["QAndA_suggest"], qa_type="suggest")
            else:
                st.warning("この企業のデータに 'QAndA_suggest' がありません。")

            # 「この会社(プロジェクト)の評価を保存」ボタン
            save_button_key = f"save_btn_file{file_idx}_proj{proj_idx}"
            if st.button(f"『{company_name}』の評価を保存", key=save_button_key):
                # アノテーション結果を反映した1社分の JSON を書き出す
                project.setdefault("annotation", {})

                ##############################
                # 1) ROI算定
                ##############################
                roi_good_or_bad_key = f"file{file_idx}_proj{proj_idx}_roi_good_or_bad"
                roi_comment_key = f"file{file_idx}_proj{proj_idx}_roi_comment"
                project["annotation"]["ROI評価"] = {
                    "良いor悪い": st.session_state["annotations"].get(roi_good_or_bad_key, "未評価"),
                    "コメント": st.session_state["annotations"].get(roi_comment_key, "")
                }

                ##############################
                # 2) ROIツリー (assignment / suggest)
                ##############################
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

                ##############################
                # 3) Q&A (assignment / suggest)
                ##############################
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

                # 出力用に「この1社だけ」の構造を作る
                single_project_data = {
                    "DXProjects": [project]
                }

                # 元ファイル名から拡張子除去
                base_root, base_ext = os.path.splitext(file_name)
                # 企業名に使えない文字があれば置換する
                safe_company_name = company_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                out_filename = f"{base_root}_{safe_company_name}_annotated.json"

                # ローカルに保存 (ローカル実行時のみ有効)
                try:
                    with open(out_filename, "w", encoding="utf-8") as out_f:
                        json.dump(single_project_data, out_f, ensure_ascii=False, indent=2)
                    st.success(f"『{company_name}』の評価結果をローカルファイル '{out_filename}' に保存しました。")
                except Exception as e:
                    st.warning(f"ローカルへの保存でエラーが発生しました: {e}")

                # ダウンロードボタン（ブラウザからもDLできるようにする）
                download_data = json.dumps(single_project_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="評価結果(この1社分)をダウンロード",
                    data=download_data.encode("utf-8"),
                    file_name=out_filename,
                    mime="application/json"
                )
