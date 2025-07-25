import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="리뷰 / SWOT 분석 Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import json
from review import reivew_crawling
import platform
from matplotlib import rc
import pandas as pd
import pdfkit

import os
import sys

pdf_content = ""


# 시스템 기본 폰트 설정
if platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')  # 윈도우: 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
else:
    rc('font', family='NanumGothic')  # 리눅스: 나눔고딕 (또는 시스템 기본 폰트)

# 환경변수
from dotenv import load_dotenv

load_dotenv()

# ABSA_LLM
# LLM 객체 생성

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# SWOT LLM
# chat_model
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# OutputParser
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

# 뉴스 레터
import requests
#검색을 통해 로드한 문서를 다룰 수 있도록 하는 라이브러리
from langchain_community.document_loaders import UnstructuredURLLoader
#가져온 텍스트를 가공
from langchain.text_splitter import CharacterTextSplitter
#임베딩, 벡터 스토어
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#키워드를 검색할 수 있도록 한다.
from langchain_community.utilities import GoogleSerperAPIWrapper



# 스타일 설정
st.markdown(
    """
    <style>
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; padding-left: 1rem; padding-right: 1rem; }
    .css-1d391kg, .sidebar-content { background-color: #F4E5DB; }
    .css-1r6slb0 { background-color: #fff; border-radius: 10px; border: 1px solid #ddd; }
    .stButton>button { color: white; background-color: #D4A373; border: None; border-radius: 5px; width: 100%; padding: 0.5rem 1rem; }
    .stMarkdown h1, h2, h3 { font-family: 'Arial', sans-serif; color: #5A4635; }
    .swot-box { background-color: #E8D4C1; color: black; border-radius: 10px; padding: 10px; margin: 10px 0; text-align: left; height: 200px; overflow-y: auto; }
    .divider-horizontal { border-top: 1px solid #d3d3d3; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

ABSA_llm = ChatOpenAI(
    temperature=0.7,  # 창의성
    model_name="gpt-4o",  # 모델명
)

# 긍정 템플릿 설정
positive_template = """
    Define the sentiment elements as follows:

    − The 'aspect term' refers to a specific feature, attribute, or aspect of an item, product, or service that a user might comment on. If implicit, the aspect term can be 'null'.
    − The 'opinion term' reflects the sentiment or attitude toward a specific aspect. It can also be 'null' for implicit opinions.
    − The 'aspect category' is the general category to which the aspect belongs. Examples may include 'quality', 'pricing', 'availability', 'appearance', 'functionality', 'service', 'general', and other suitable categories as applicable across diverse domains.
    − The 'sentiment polarity' expresses whether the sentiment is 'positive'.

    Based on the provided definitions, identify and organize all positive sentiment elements from the following text. Please provide **exactly two examples** for positive polarity.
    
    You should provide your answer in Korean, but leave 'null' as it is.
    Please answer in Korean with the following format:

    1.
       - **측면 용어**: (Aspect Term)
       - **의견 용어**: (Opinion Term)
       - **측면 카테고리**: (Aspect Category)
       - **감정 극성**: (sentiment polarity)

    2.
       - **측면 용어**: (Aspect Term)
       - **의견 용어**: (Opinion Term)
       - **측면 카테고리**: (Aspect Category)
       - **감정 극성**: (sentiment polarity)

    Text to analyze:
    {text}
"""

positive_prompt = PromptTemplate(
    template=positive_template,
    input_variables=["text"]
)

# 부정 템플릿 설정
negative_template = positive_template.replace("positive", "negative").replace("positive sentiment elements", "negative sentiment elements")
negative_prompt = PromptTemplate(
    template=negative_template,
    input_variables=["text"]
)

# 중립 템플릿 설정
neutral_template = positive_template.replace("positive", "neutral").replace("positive sentiment elements", "neutral sentiment elements")
neutral_prompt = PromptTemplate(
    template=neutral_template,
    input_variables=["text"]
)

# 각각 체인 생성
positive_chain = (
    positive_prompt
    | ABSA_llm
)

negative_chain = (
    negative_prompt
    | ABSA_llm
)

neutral_chain = (
    neutral_prompt
    | ABSA_llm
)

#######################################################################################################################################################
#######################################################################################################################################################


SWOT_llm = ChatOpenAI(
    temperature=0.7,  # 창의성
    model_name="gpt-4o",  # 모델명
)

class SWOTKeywords(BaseModel):
    strength_keyword: str = Field(description="제품의 경쟁 우위를 나타내는 긍정적인 키워드나 문구")
    weakness_keyword: str = Field(description="제품의 한계점이나 개선이 필요한 부분을 나타내는 키워드나 문구")
    opportunity_keyword: str = Field(description="리뷰를 통해 발견된 잠재적 성장 기회나 새로운 시장 기회를 나타내는 키워드나 문구")
    threat_keyword: str = Field(description="제품의 성공을 저해할 수 있는 우려사항이나 외부 위험 요소를 나타내는 키워드나 문구")

class SWOTResponse(BaseModel):
    keywords: List[SWOTKeywords] = Field(description="리뷰 텍스트에서 SWOT 카테고리별로 추출된 키워드 리스트")

# Parser 설정
swot_parser = PydanticOutputParser(pydantic_object=SWOTResponse)

swot_template = """
    다음 리뷰 텍스트를 분석하여 SWOT 분석을 수행하고, 정확히 아래 JSON 형식으로 출력해주세요.
    분석이 어려운 카테고리가 있더라도 반드시 모든 필드를 포함해야 합니다.
    키워드나 인사이트가 없는 경우 "없음" 또는 "관련 내용 없음"으로 표시해주세요.

    분석할 리뷰 텍스트:
    {review_text}

    반드시 다음 형식의 JSON으로 출력해주세요:
    {{
        "keywords": [
            {{
                "strength_keyword": "강점 키워드1",
                "weakness_keyword": "약점 키워드1",
                "opportunity_keyword": "기회 키워드1",
                "threat_keyword": "위협 키워드1"
            }},
            {{
                "strength_keyword": "강점 키워드2",
                "weakness_keyword": "약점 키워드2",
                "opportunity_keyword": "기회 키워드2",
                "threat_keyword": "위협 키워드2"
            }}
        ]
    }}

    주의사항:
    1. 반드시 위의 JSON 형식을 정확히 따라주세요
    2. 모든 필드는 필수이며
    3. 다른 설명이나 부가 텍스트 없이 JSON만 출력
    4. 실제 키워드는 한글로 작성

    {format_instructions}
    """

swot_prompt = PromptTemplate(
    template=swot_template,
    input_variables=["review_text"],
    partial_variables={"format_instructions": swot_parser.get_format_instructions()}
)

swot_chain = (
    swot_prompt
    | SWOT_llm
    | swot_parser
)

def collect_swot_keywords(swot_response):
    # 각 카테고리별 키워드를 저장할 딕셔너리
    collected = {
        'strengths': [],
        'weaknesses': [],
        'opportunities': [],
        'threats': []
    }
    
    # 각 키워드 세트에서 카테고리별로 수집
    for keyword_set in swot_response.keywords:
        # 강점 수집
        if keyword_set.strength_keyword and keyword_set.strength_keyword != "없음":
            # 쉼표로 구분된 경우 분리
            strengths = [s.strip() for s in keyword_set.strength_keyword.split(',')]
            collected['strengths'].extend(strengths)
        
        # 약점 수집
        if keyword_set.weakness_keyword and keyword_set.weakness_keyword != "없음":
            weaknesses = [w.strip() for w in keyword_set.weakness_keyword.split(',')]
            collected['weaknesses'].extend(weaknesses)
            
        # 기회 수집
        if keyword_set.opportunity_keyword and keyword_set.opportunity_keyword != "없음":
            opportunities = [o.strip() for o in keyword_set.opportunity_keyword.split(',')]
            collected['opportunities'].extend(opportunities)
            
        # 위협 수집
        if keyword_set.threat_keyword and keyword_set.threat_keyword != "없음":
            threats = [t.strip() for t in keyword_set.threat_keyword.split(',')]
            collected['threats'].extend(threats)
    
    # 중복 제거
    collected = {k: list(set(v)) for k, v in collected.items()}
    
    return collected

#######################################################################################################################################################
#######################################################################################################################################################

embeddings = OpenAIEmbeddings()

def search_serp(query):
  #k=몇 개의 리스트를 반환할 것인지
  search = GoogleSerperAPIWrapper(k=10, type="search")

  response_json = search.results(query)
  print(f"Response ====> , {response_json}")

  return response_json

def pick_best_articles_urls(response_json, query):
  response_str = json.dumps(response_json)
    
  llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
  template = """
    너는 소비자 트렌드와 시장 동향을 분석하는 전문 평론가야.
    지금 분석 중인 상품 브랜드의 기회와 위협 요인에 관한 최신 뉴스나 아티클을 찾고 있어.
    
    QUERY RESPONSE :{response_str}

    위의 리스트는 쿼리 결과에 대한 검색 리스트야 {query}.

    네가 생각하기에 쿼리의 주제를 잘 담고 있는 가장 훌륭한 3개의 아티클을 찾아봐.

    return ONLY an array of urls.
    also make sure the articles are recent and not too old.
    if the file, or URL is invalid, show www.google.com
    """
  
  prompt_template = PromptTemplate(input_variables=["response_str", "query"], template = template)

  article_chooser_chain = LLMChain(
      llm=llm,
      prompt=prompt_template,
      verbose=True
  )

  try:
      urls = article_chooser_chain.run(response_str=response_str, query=query)
      url_list = json.loads(urls)  # JSON 변환 시도
  except json.JSONDecodeError as e:
      print(f"JSONDecodeError: {e}")
      url_list = ["https://www.google.com"]  # 기본 값 설정

  print(url_list)
  return url_list

def extract_content_from_urls(urls):

  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()

  text_splitter = CharacterTextSplitter(
      separator='\n',
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
  )

  docs = text_splitter.split_documents(data)
  db = FAISS.from_documents(docs, embeddings)

  return db

def summarizer(db, query, k=4):
  docs = db.similarity_search(query, k=k)

  docs_page_content = " ".join([d.page_content for d in docs])

  llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
  template = """
    {docs}
    
    가이드라인에 맞춰서 적어줘.
    1. 콘텐츠가 긍정적인 측면과 함께 고려할 위험 요소도 염두에 두고 있는 균형잡힌 내용이어야 하며, 정보를 얻을 수 있고 유용해야 해.
    2. 콘텐츠가 너무 길지 않은지 확인해야 해.
    3. 콘텐츠는 {query}토픽을 잘 나타내고 있는 내용이어야 해.
    4. 콘텐츠는 읽기 쉽게 쓰여야 하고, 간결해야 해.
    5. 콘텐츠는 독자에게 영감을 줄 수 있어야 해.


    읽기 편하게 한국어로 작성해 줘.

    SUMMARY :
    """

  prompt_template = PromptTemplate(input_variables=["docs", "query"], template=template)

  summarize_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
  response = summarize_chain.run(docs=docs_page_content, query=query)

  return response.replace("\n", "")

#######################################################################################################################################################
#######################################################################################################################################################

# 상품 브랜드 전망
outlook_llm = ChatOpenAI(
    model= "gpt-4o",
    temperature=0.7
)

outlook_template = """
    너는 소비자 트렌드와 시장 동향을 분석하는 전문 평론가야.

    {article_summary}

    위의 정보는 해당 상품 브랜드의 소비자 리뷰를 사용한 SWOT 분석의 Opportunity와 Threat 키워드를 기반으로 수집한 뉴스 기사 요약이야.

    긍정적인 측면과 함께 고려할 위험 요소도 염두에 두고, 분석 중인 상품 브랜드의 향후 전망을 균형 잡힌 시각에서 알려줘.
"""

outlook_prompt = PromptTemplate(
    template= outlook_template,
    input_variables=["article_summary"]
)

outlook_chain = (
    outlook_prompt
    | outlook_llm
)

#######################################################################################################################################################
#######################################################################################################################################################



st.title("리뷰 / SWOT 분석 Dashboard")

# URL 입력 및 버튼
url = st.text_input("URL을 입력하세요.", placeholder="분석할 URL을 입력하세요.", key="anal_url_input")
start_button = st.button("분석시작", key="start_button")



# 사이드바 설정
st.sidebar.header("검색기록")
url_link = []
url_link.append(url)
st.sidebar.markdown(f"[링크 1]({url_link[-1]})", unsafe_allow_html=True)


placeholder = st.empty()

if start_button and url:  # 버튼 클릭 시 실행
    try:
        prod_name = reivew_crawling(url)
    except Exception as e:
        st.error(f"크롤링 오류가 발생했습니다: {e}")
        prod_name = None
    
    review_folder = os.path.join(os.path.dirname(__file__), 'reviewxisx')
    latest_file = max([os.path.join(review_folder, f) for f in os.listdir(review_folder)], key=os.path.getctime)
    df = pd.read_excel(latest_file, engine='openpyxl')
    latest_reviews_text = df['리뷰 내용'].to_string()  # 최대 1500자까지만 저장

    ABSA_positive_result = positive_chain.invoke({"text": latest_reviews_text})
    ABSA_negative_result = negative_chain.invoke({"text": latest_reviews_text})
    ABSA_neutral_result = neutral_chain.invoke({"text": latest_reviews_text})

    swot_response = swot_chain.invoke({"review_text": latest_reviews_text})
    swot_analysis_result = collect_swot_keywords(swot_response)

    # 분석 결과가 비어있는지 확인
    if swot_analysis_result:
        try:
            swot_results = swot_analysis_result
        except json.JSONDecodeError as e:
            st.error(f"오류: {e}")
            swot_results = None
    else:
        st.error("URL을 확인하거나 다시 시도해 주세요.")
        swot_results = None

    if swot_results:
        # 입력 및 버튼 제거
        placeholder = st.empty()
        placeholder.empty()
        
        # 분석 결과 레이아웃 출력
        top_left, top_right = st.columns(2)
        
        with top_left:
            st.subheader("리뷰 분석")
            
            # 긍정, 부정, 중립 리뷰 결과를 표시할 3개의 열 생성
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### 긍정적 리뷰")
                st.write(ABSA_positive_result.content)  # 긍정 리뷰 결과 출력

            with col2:
                st.write("### 부정적 리뷰")
                st.write(ABSA_negative_result.content)  # 부정 리뷰 결과 출력

            with col3:
                st.write("### 중립적 리뷰")
                st.write(ABSA_neutral_result.content)  # 중립 리뷰 결과 출력

        with top_right:
            st.subheader("SWOT 분석")
            swot_cols = st.columns(2)

            with swot_cols[0]:
                st.markdown(
                    f"""
                    <div class="swot-box">
                        <strong>Strengths</strong><br>
                        {'<br>'.join(f'- {item}' for item in swot_results.get('strengths', []))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with swot_cols[1]:
                st.markdown(
                    f"""
                    <div class="swot-box">
                        <strong>Weaknesses</strong><br>
                        {'<br>'.join(f'- {item}' for item in swot_results.get('weaknesses', []))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            swot_cols2 = st.columns(2)
            with swot_cols2[0]:
                st.markdown(
                    f"""
                    <div class="swot-box">
                        <strong>Opportunities</strong><br>
                        {'<br>'.join(f'- {item}' for item in swot_results.get('opportunities', []))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with swot_cols2[1]:
                st.markdown(
                    f"""
                    <div class="swot-box">
                        <strong>Threats</strong><br>
                        {'<br>'.join(f'- {item}' for item in swot_results.get('threats', []))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # 가로 구분선 추가
        st.markdown('<div class="divider-horizontal"></div>', unsafe_allow_html=True)

        # 아래 레이아웃이 항상 동일한 행에 위치하도록 컨테이너 유지
        bottom_left, bottom_right = st.columns((1, 1), gap="medium")

        test_oppor = " ".join(swot_results['opportunities'])
        test_threat = " ".join(swot_results['threats'])

        oppor_query = f'상품명은 {prod_name}, 해당 상품의 {test_oppor}'
        threat_query = f'상품명은 {prod_name}, 해당 상품의 {test_threat}'

        search_result_oppor = search_serp(oppor_query)
        oppor_urls = pick_best_articles_urls(response_json=search_result_oppor, query=oppor_query)
        oppor_data = extract_content_from_urls(oppor_urls)
        oppor_summaries = summarizer(db=oppor_data, query=oppor_query)

        search_result_threat = search_serp(threat_query)
        threat_urls = pick_best_articles_urls(response_json=search_result_threat, query=threat_query)
        threat_data = extract_content_from_urls(threat_urls)
        threat_summaries = summarizer(db=threat_data, query=threat_query)

        
        with bottom_left:
            st.subheader("뉴스 기사")

            st.write("### 기회 관련 뉴스 기사")
            st.write(oppor_summaries)
    
            st.markdown('<hr style="border:1px solid #ddd;">', unsafe_allow_html=True)
    
            st.write("### 위협 관련 뉴스 기사")
            st.write(threat_summaries)

        total_news_summary = oppor_summaries + threat_summaries

        outlook_result = outlook_chain.invoke({"article_summary" : total_news_summary})

        with bottom_right:
            st.markdown('<div class="divider-vertical"></div>', unsafe_allow_html=True) 
            st.subheader("향후 전망")
            st.write(outlook_result.content)
            
            