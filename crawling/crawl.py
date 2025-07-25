# coupang_review.py

from bs4 import BeautifulSoup as bs
from openpyxl import Workbook
from requests.exceptions import RequestException
import time
import os
import re
import requests as rq
import json
import math
import sys

# 헤더 파일 로드
def get_headers(key: str) -> dict[str, str]:
    """헤더 정보 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, 'headers.json')
    with open(json_file_path, 'r', encoding='UTF-8') as file:
        headers = json.loads(file.read())
    
    try:
        return headers[key]
    except KeyError:
        raise EnvironmentError(f"{key}가 설정되지 않았습니다.")

# 쿠팡 크롤러 클래스
class Coupang:
    @staticmethod
    def get_product_code(url: str) -> str:
        """URL에서 상품 코드를 추출"""
        return url.split("products/")[-1].split("?")[0]

    @staticmethod
    def get_soup_object(resp: rq.Response) -> bs:
        """HTML 내용을 BeautifulSoup 객체로 변환"""
        return bs(resp.text, "html.parser")

    def __init__(self) -> None:
        self.__headers = get_headers(key="headers")
        self.base_review_url = "https://www.coupang.com/vp/product/reviews"
        self.sd = SaveData()
        self.retries = 5
        self.delay = 2

    def get_product_info(self, prod_code: str) -> tuple:
        """상품 정보 추출"""
        url = f"https://www.coupang.com/vp/products/{prod_code}"
        resp = rq.get(url=url, headers=self.__headers)
        soup = self.get_soup_object(resp=resp)
        
        title = soup.select_one("h1.prod-buy-header__title").text.strip()
        review_count = int(re.sub("[^0-9]", "", soup.select("span.count")[0].text.strip()))
        return title, review_count

    def start(self, url: str) -> None:
        """크롤링 시작"""
        self.sd.create_directory()
        self.__headers["Referer"] = url
        prod_code = self.get_product_code(url=url)

        # 상품 정보 추출
        self.title, review_count = self.get_product_info(prod_code=prod_code)
        review_pages = 9

        # 페이로드 설정
        payloads = [{
            "productId": prod_code,
            "page": page,
            "size": 5,
            "sortBy": "ORDER_SCORE_ASC",
            "ratings": "",
            "q": "",
            "viRoleCode": 2,
            "ratingSummary": True
        } for page in range(1, review_pages + 1)]
        
        # 데이터 크롤링
        for payload in payloads:
            self.fetch(payload=payload)

    def fetch(self, payload: dict) -> None:
        """리뷰 페이지에서 데이터 추출"""
        now_page = payload["page"]
        print(f"\n[INFO] 페이지 {now_page} 크롤링 시작...\n")
        
        for attempt in range(self.retries):
            try:
                with rq.Session() as session:
                    with session.get(
                        url=self.base_review_url, headers=self.__headers, params=payload, timeout=10
                    ) as resp:
                        resp.raise_for_status()
                        soup = self.get_soup_object(resp)
                        articles = soup.select("article.sdp-review__article__list")
                        
                        # 리뷰 데이터 추출
                        for article in articles:
                            dict_data = {
                                "title": self.title,
                                "review_date": article.select_one(
                                    "div.sdp-review__article__list__info__product-info__reg-date"
                                ).text.strip() if article.select_one(
                                    "div.sdp-review__article__list__info__product-info__reg-date") else "-",
                                "user_name": article.select_one(
                                    "span.sdp-review__article__list__info__user__name"
                                ).text.strip() if article.select_one(
                                    "span.sdp-review__article__list__info__user__name") else "-",
                                "rating": int(article.select_one(
                                    "div.sdp-review__article__list__info__product-info__star-orange"
                                ).attrs["data-rating"]) if article.select_one(
                                    "div.sdp-review__article__list__info__product-info__star-orange") else 0,
                                "review_content": re.sub(
                                    "[\n\t]", "", article.select_one(
                                        "div.sdp-review__article__list__review > div"
                                    ).text.strip()) if article.select_one(
                                        "div.sdp-review__article__list__review > div") else "리뷰 내용 없음",
                            }
                            self.sd.save(datas=dict_data)
                            print(dict_data, "\n")
                        time.sleep(1)
                        return
            except RequestException as e:
                print(f"{attempt + 1}번째 시도 실패: {e}")
                time.sleep(self.delay)
        print("최대 재시도 초과. 종료합니다.")
        sys.exit()

    def calculate_total_pages(self, review_counts: int) -> int:
        """리뷰 총 페이지 수 계산"""
        return math.ceil(review_counts / 5)

class SaveData:
    def __init__(self) -> None:
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.append(["제목", "작성일자", "사용자명", "평점", "리뷰 내용"])
        self.row = 2
        self.dir_name = "reviewxisx"

    def create_directory(self) -> None:
        """저장 폴더 생성"""
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

    def save(self, datas: dict) -> None:
        """데이터를 엑셀 파일에 저장"""
        file_name = os.path.join(self.dir_name, f"{datas['title']}.xlsx")
        self.ws.append([datas["title"], datas["review_date"], datas["user_name"], datas["rating"], datas["review_content"]])
        self.wb.save(filename=file_name)

    def __del__(self) -> None:
        self.wb.close()