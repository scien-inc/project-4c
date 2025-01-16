import json
import glob
import os
import streamlit as st

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
# アノテーション部分のUI
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


def annotate_roi_trees(file_idx: int, proj_idx: int, roi_trees: dict):
    """
    ROIツリー（Mermaid）のアノテーションUI。
    """
    st.subheader("■ ROIツリー評価")

    for depth_key, mermaid_code in roi_trees.items():
        st.markdown(f"### {depth_key}")
        # Mermaidコードをテキストで表示（静的）
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

        base_key = f"file{file_idx}_proj{proj_idx}_roiTrees_{depth_key}"
        good_or_bad_key = base_key + "_good_or_bad"
        comment_key = base_key + "_comment"

        get_radio_value(f"{depth_key} は良い？悪い？", ["良い", "悪い", "未評価"], good_or_bad_key)
        get_text_area_value(f"{depth_key} のどこが悪いか（自由記述）", comment_key)


def annotate_q_and_a(file_idx: int, proj_idx: int, q_and_a: dict):
    """
    Q&A (Depth3, Depth4, Depth5) へのアノテーションUI。
    """
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

                # 良い/悪い の評価
                base_key = f"file{file_idx}_proj{proj_idx}_QAndA_{depth_key}_{qa_item_idx}_{q_idx}"
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

st.title("フォルダ内のDX Projects 一括アノテーションツール")

# 1) フォルダパスをテキスト入力で取得
folder_path = st.text_input("JSONファイルが格納されているフォルダパスを入力してください", value="json_data")

# フォルダが存在するかチェック
if not os.path.isdir(folder_path):
    st.warning("有効なフォルダパスを入力してください。")
    st.stop()

# 2) フォルダ内の .json ファイルをすべて取得
json_paths = glob.glob(os.path.join(folder_path, "*.json"))
if not json_paths:
    st.warning(f"指定フォルダ({folder_path})に json ファイルがありません。")
    st.stop()

st.write(f"以下のフォルダから {len(json_paths)} 件のJSONファイルを読み込みます:")
for path in json_paths:
    st.write(f"- {path}")

# セッションステート初期化
if "annotations" not in st.session_state:
    st.session_state["annotations"] = {}

# 3) 全ファイルを読み込んでUI生成
for file_idx, json_file_path in enumerate(json_paths):
    # ファイル読み込み
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # DXProjects が無ければスキップ
    if "DXProjects" not in data:
        st.error(f"ファイル {json_file_path} に 'DXProjects' キーが見つかりません。スキップします。")
        continue

    st.markdown("---")
    st.markdown(f"## ファイル: `{json_file_path}`")

    # ファイルに含まれるプロジェクト(企業)を順に表示
    for proj_idx, project in enumerate(data["DXProjects"]):
        company_name = project["table"].get("企業名", f"Unknown_{proj_idx}")
        purpose = project["table"].get("課題・目的", "不明な課題")

        with st.expander(f"[{company_name}] / 課題: {purpose}", expanded=False):
            # 企業ごとのアノテーションUI
            annotate_roi(file_idx, proj_idx, project)
            annotate_roi_trees(file_idx, proj_idx, project["roiTrees"])
            annotate_q_and_a(file_idx, proj_idx, project["QAndA"])

            # 「この会社(プロジェクト)の評価を保存」ボタン
            save_button_key = f"save_btn_file{file_idx}_proj{proj_idx}"
            if st.button(f"『{company_name}』の評価を保存", key=save_button_key):
                # アノテーション結果を反映した1社分の JSON を書き出す
                # (元データを読み直し → 該当プロジェクトだけに annotation を付与 → 別ファイルに出力)

                # まず、プロジェクトに annotation を付ける
                project.setdefault("annotation", {})

                # ROI
                roi_good_or_bad_key = f"file{file_idx}_proj{proj_idx}_roi_good_or_bad"
                roi_comment_key = f"file{file_idx}_proj{proj_idx}_roi_comment"
                project["annotation"]["ROI評価"] = {
                    "良いor悪い": st.session_state["annotations"].get(roi_good_or_bad_key, "未評価"),
                    "コメント": st.session_state["annotations"].get(roi_comment_key, "")
                }

                # ROIツリー
                roi_trees_annotation = {}
                for depth_key in project["roiTrees"].keys():
                    base_key = f"file{file_idx}_proj{proj_idx}_roiTrees_{depth_key}"
                    good_or_bad_key = base_key + "_good_or_bad"
                    comment_key = base_key + "_comment"
                    roi_trees_annotation[depth_key] = {
                        "良いor悪い": st.session_state["annotations"].get(good_or_bad_key, "未評価"),
                        "コメント": st.session_state["annotations"].get(comment_key, "")
                    }
                project["annotation"]["roiTrees評価"] = roi_trees_annotation

                # Q&A
                q_and_a_annotation = {}
                for depth_key, qa_list in project["QAndA"].items():
                    depth_evals = []
                    for qa_item_idx, qa_item in enumerate(qa_list):
                        question_evals = []
                        for q_idx in range(len(qa_item["questions"])):
                            base_key = f"file{file_idx}_proj{proj_idx}_QAndA_{depth_key}_{qa_item_idx}_{q_idx}"
                            qa_good_or_bad_key = base_key + "_good_or_bad"
                            qa_comment_key = base_key + "_comment"
                            question_evals.append({
                                "良いor悪い": st.session_state["annotations"].get(qa_good_or_bad_key, "未評価"),
                                "コメント": st.session_state["annotations"].get(qa_comment_key, "")
                            })
                        depth_evals.append(question_evals)
                    q_and_a_annotation[depth_key] = depth_evals
                project["annotation"]["QAndA評価"] = q_and_a_annotation

                # 出力用に「この1社だけ」の構造を作る
                single_project_data = {
                    "DXProjects": [project]
                }

                # ファイル名を決める (例: data.json -> data_A社_annotated.json)
                base_name = os.path.basename(json_file_path)  # 例: data.json
                base_root, base_ext = os.path.splitext(base_name)
                # 企業名に使えない文字があれば置換する
                safe_company_name = company_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                out_filename = f"{base_root}_{safe_company_name}_annotated.json"

                # 同じフォルダ内に保存する例
                out_path = os.path.join(folder_path, out_filename)

                # 書き出し
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(single_project_data, out_f, ensure_ascii=False, indent=2)

                st.success(f"『{company_name}』の評価結果を保存しました: {out_path}")

                # ダウンロードボタン（ブラウザからもDLできるようにする）
                download_data = json.dumps(single_project_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="評価結果(この1社分)をダウンロード",
                    data=download_data.encode("utf-8"),
                    file_name=out_filename,
                    mime="application/json"
                )
