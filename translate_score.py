import streamlit as st
import pandas as pd
import ollama


# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])


# 번역 품질 평가 함수
def evaluate_translation_quality(korean, fl, fl_index):
    prompt = (
        f"아래는 한국어를 {fl_index} 언어로 번역한 것입니다. 이 번역 품질을 100점 만점 기준으로 어느 수준인지 판단을 부탁 드립니다. \n"
        f"한국어: '{korean}'\n"
        f"{fl_index}: '{fl}'"
    )
    response = ollama.chat(
        model="llama3.2_ko:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# CSV 파일이 업로드되었을 때
if uploaded_file is not None:
    # CSV 파일을 pandas DataFrame으로 변환
    df = pd.read_csv(uploaded_file)

    column_indices = df.columns[8:14].tolist()
    st.write("업로드된 데이터:")
    st.write(df)

    # Streamlit에서 열 선택할 수 있도록 selectbox 제공
    for selected_column in column_indices:

        # 데이터가 있는지 확인
        if "국문(KO)" in df.columns and selected_column in df.columns:

            # 데이터를 리스트로 변환
            a_list = df["국문(KO)"].tolist()
            b_list = df[selected_column].tolist()

            # 평가 결과를 저장할 리스트
            evaluation_results = []
            temp_result = []
            f_score = selected_column + "score"

            # 각 항목에 대해 번역 품질 평가 실시
            for a, b in zip(a_list, b_list):
                score = evaluate_translation_quality(a, b, selected_column)
                temp_result.append(score)
                evaluation_results.append((a, b, score))

            selected_index = df.columns.get_loc(selected_column) + 1
            df.insert(selected_index, f_score, temp_result)

            # # 결과 출력
            # st.write("번역 품질 평가 결과:")
            # for korean, english, score in evaluation_results:
            #     st.write(f"한국어: {korean}")
            #     st.write(f"영어: {english}")
            #     st.write(f"번역 품질 점수: {score}/100")
            #     st.write("---")
        else:
            st.error(
                "말씀하신 언어 데이터가 파일에 없습니다. 올바른 형식의 파일을 업로드해주세요."
            )

    st.write(df)
    df.to_csv("output.csv", index=False, encoding="utf-8-sig")
