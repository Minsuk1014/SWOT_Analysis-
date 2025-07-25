# 시각화

import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import base64
from io import BytesIO
from matplotlib import rc
import platform

# 시스템 기본 폰트 설정
if platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')  # 윈도우: 맑은 고딕
elif platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
else:
    rc('font', family='NanumGothic')  # 리눅스: 나눔고딕 (또는 시스템 기본 폰트)

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# Step 1: 데이터 준비 (프롬프트 엔지니어링 결과 낼 때 속성별 긍부정 비율까지 나타내달라고 하면 좋을듯?)
data = {
    'Attribute': ['Product Performance', 'Ease of Use', 'Design & Size', 'Value for Money', 'Cleaning & Maintenance'],
    'Positive': [80, 90, 75, 85, 70],
    'Negative': [20, 10, 25, 15, 30]
}

df = pd.DataFrame(data)

# 프롬프트 결과 중 긍정과 부정을 따로 입력
positive_keywords = {
    'Product Performance': "easy yogurt steam convenient fast efficient reliable",
    'Ease of Use': "simple user-friendly intuitive straightforward quick setup",
    'Design & Size': "compact efficient sleek modern space-saving",
    'Value for Money': "affordable functional worth great deal cost-effective",
    'Cleaning & Maintenance': "easy-clean detachable quick-clean user-friendly"
}

negative_keywords = {
    'Product Performance': "slow time-consuming delayed inconsistent",
    'Ease of Use': "confusing unclear manual difficult hard",
    'Design & Size': "plain dull uninspired basic boring",
    'Value for Money': "fragile flimsy questionable quality cheap short-lived",
    'Cleaning & Maintenance': "cumbersome tedious difficult parts annoying"
}

# Step 2: 시각화 함수 정의
def create_bar_chart():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df['Attribute'], df['Positive'], label='긍정', color='skyblue')
    ax.bar(df['Attribute'], df['Negative'], bottom=df['Positive'], label='부정', color='salmon')
    #ax.set_title('Sentiment Analysis by Attribute', fontsize=16)
    ax.set_xlabel('속성', fontsize=12)
    ax.set_ylabel('리뷰 수', fontsize=12)
    ax.legend(title='감정')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    return fig

def create_wordcloud(keywords, colormap):
    wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(keywords)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# Step 3: 시각화 생성
bar_chart_fig = create_bar_chart()
bar_chart_base64 = fig_to_base64(bar_chart_fig)

# 모든 속성의 워드클라우드 생성
wordclouds = {}
for attr in positive_keywords:
    pos_wc = create_wordcloud(positive_keywords[attr], 'Blues')
    neg_wc = create_wordcloud(negative_keywords[attr], 'Reds')
    wordclouds[attr] = {
        "positive": fig_to_base64(pos_wc),
        "negative": fig_to_base64(neg_wc)
    }

# HTML 내부에 드롭다운과 워드클라우드 구현
attribute_options = "\n".join([f'<option value="{attr}">{attr}</option>' for attr in wordclouds.keys()])

popup_html = f"""
<div style="position:relative;">
    <button id="popup-button" style="font-size:20px; background:none; border:none; color:white; text-decoration:underline; cursor:pointer;">
        시각화보기
    </button>
    <div id="popup" style="display:none; position:absolute; top:50px; left:0; width:90vw; max-height:90vh; overflow-y:auto; border:1px solid black; background-color:white; padding:10px; z-index:100;">
        <h3>제품 속성별 분석 결과</h3>
        <img src="data:image/png;base64,{bar_chart_base64}" style="width:100%;">
        
        <!-- Gray Divider Line -->
        <div style="border-top: 1px solid lightgray; margin: 20px 0;"></div>

        <div>
            <label for="attribute-select" style="font-weight:bold;">속성 선택:</label>
            <select id="attribute-select" onchange="updateWordClouds()">
                {attribute_options}
            </select>
        </div>

        <div style="display:flex; justify-content:space-between; margin-top:0px;">
            <div style="width:48%;">
                <h4 style="margin-bottom:0px;">긍정</h4>
                <img id="positive-wc" src="data:image/png;base64,{wordclouds['Product Performance']['positive']}" style="width:100%;">
            </div>
            <div style="width:48%;">
                <h4 style="margin-bottom:0px;">부정</h4>
                <img id="negative-wc" src="data:image/png;base64,{wordclouds['Product Performance']['negative']}" style="width:100%;">
            </div>
        </div>
    </div>
</div>

<script>
    const button = document.getElementById('popup-button');
    const popup = document.getElementById('popup');
    const positive_wc = document.getElementById('positive-wc');
    const negative_wc = document.getElementById('negative-wc');
    const attributeSelect = document.getElementById('attribute-select');

    button.addEventListener('mouseenter', () => {{
        popup.style.display = 'block';
    }});

    button.addEventListener('mouseleave', () => {{
        if (!popup.hasAttribute('data-persistent')) {{
            popup.style.display = 'none';
        }}
    }});

    button.addEventListener('click', () => {{
        popup.setAttribute('data-persistent', true);
    }});

    function updateWordClouds() {{
        const selectedAttribute = attributeSelect.value;
        positive_wc.src = "data:image/png;base64," + wordclouds[selectedAttribute].positive;
        negative_wc.src = "data:image/png;base64," + wordclouds[selectedAttribute].negative;
    }}

    const wordclouds = {str(wordclouds).replace("'", '"')};  // JS-compatible JSON
</script>
"""

# Streamlit에서 HTML 렌더링
st.components.v1.html(popup_html, height=1000)
