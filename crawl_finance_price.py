'''
- Data : 2022.11.29
- Author : Jeong Soo Yeon
- Description :  네이버 주식 페이지 내 코스피/코스닥 종목 정보 크롤링
'''
import pandas as pd
from urllib3.util.retry import Retry
from datetime import datetime
import requests
from bs4 import BeautifulSoup as bs


url = "https://finance.naver.com/sise/sise_rise.nhn?sosok={}"
today = datetime.today().strftime("%Y.%m.%d")
lst = ['KOSPI', 'KOSDAQ']
HEADERS = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36", "connection": "close"}

current_url = url.format(0)

r = requests.Session()
resp = r.get(current_url, headers = HEADERS, verify=False)

soup = bs(resp.text, 'lxml', from_encoding = 'utf-8')
aa = soup.find('table', class_='type_2').findAll('tr')

stck_cd_list, vlm_list, crrnt_list, stck_nm_list = [], [], [], []
for a in aa:
    if a.a != None:
        stck_cd_list.append(a.a['href'][-6:]) # stock_code
        stck_nm_list.append(a.a.text) # stock_name
        vlm_list.append(a.findAll('td', class_='number')[3].text) # volume
        crrnt_list.append(a.findAll('td', class_='number')[0].text) # 현재가


df = pd.DataFrame(list(zip(stck_cd_list, stck_nm_list, vlm_list, crrnt_list)), columns=['stock_code', 'stock_name', 'volume', 'current'])
print(df)



