# 프롬프트 분석
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)  # 임포트 전에 sys.path 추가


import streamlit as st
import pandas as pd

# 환경변수
from dotenv import load_dotenv

load_dotenv()

# ABSA
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI

# SWOT Keywords
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

# # NewsLetter
# import os
# import openai
# import json
# import requests

# from langchain import LLMChain, OpenAI, PromptTemplate

# from langchain.chat_models import ChatOpenAI
# #검색을 통해 로드한 문서를 다룰 수 있도록 하는 라이브러리
# from langchain.document_loaders import UnstructuredURLLoader
# #가져온 텍스트를 가공
# from langchain.text_splitter import CharacterTextSplitter

# #임베딩, 벡터 스토어
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# #키워드를 검색할 수 있도록 한다.
# from langchain.utilities import GoogleSerperAPIWrapper

# 크롤링 및 시각화 및 기타
import matplotlib.pyplot as plt
from io import BytesIO
from crawling.crawl import Coupang, SaveData
from collections import Counter
import matplotlib.font_manager as fm
import altair as alt

link_history_list = []
latest_reviews_text = ""

def reivew_crawling(url):
    global latest_reviews_text

    if "https://www.coupang.com" in url: # 쿠팡링크만 가능하게 하기
        link_history_list.append(url)

        # Coupang 클래스 인스턴스 생성 및 크롤링 시작
        try:
            coupang = Coupang()
            coupang.start(url)  # URL을 start 메서드에 전달
        except Exception as e:
            st.error(f"크롤링 오류: {e}")
            return

        # 크롤링된 엑셀 파일 경로 설정 (가장 최신 파일 가져오기)
        try:
            review_folder = os.path.join(os.path.dirname(__file__), 'reviewxisx')
            latest_file = max([os.path.join(review_folder, f) for f in os.listdir(review_folder)], key=os.path.getctime)
            prod_name = os.path.splitext(os.path.basename(latest_file))[0]
            df = pd.read_excel(latest_file)
        except Exception as e:
            st.error(f"엑셀 파일 오류: {e}")
            return

        # 데이터가 비어 있는지 확인
        if df.empty:
            st.error("크롤링된 데이터가 없습니다. 유효한 URL을 입력해 주세요.")
            return

        # 리뷰 텍스트의 일부를 저장 (너무 길 경우, 일부만 사용)
        latest_reviews_text = df['리뷰 내용'].to_string()[:1500]  # 최대 1500자까지만 저장

    else:
        st.error("유효한 URL을 입력해 주세요.")
        return
    
    return prod_name
    
    



# # Streamlit 앱 구성 -> 일단 되는지 확인하려고 만든거 / 만약 리뷰 확인만 하고 싶으면 주석풀고 사용!
# st.title("리뷰 / SWOT 분석 Dashboard")

# url = st.text_input("아래의 URL을 입력하세요", key="url_input")
# if st.button("분석 시작"):
#     if url:
#         prod_name = reivew_crawling(url)
#         st.write(prod_name)
#         st.write(latest_reviews_text)
#     else:
#         st.error("URL을 입력해 주세요.")
