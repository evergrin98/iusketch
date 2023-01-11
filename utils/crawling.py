import os

from bs4 import BeautifulSoup
from newspaper import Article   # new crawling에 특화됨. newspaper3k module
import requests
import pandas as pd
import urllib

def html_parser(html):
    '''
    html을 파싱하는 html parser생성.
    
    soup.select('원하는 정보')  # select('원하는 정보') -->  단 하나만 있더라도, 복수 가능한 형태로 되어있음
    soup.select('태그명')
    soup.select('.클래스명')
    soup.select('상위태그명 > 하위태그명 > 하위태그명')
    soup.select('상위태그명.클래스명 > 하위태그명.클래스명')    # 바로 아래의(자식) 태그를 선택시에는 > 기호를 사용
    soup.select('상위태그명.클래스명 하~위태그명')              # 아래의(자손) 태그를 선택시에는   띄어쓰기 사용
    soup.select('상위태그명 > 바로아래태그명 하~위태그명')     
    soup.select('.클래스명')
    soup.select('#아이디명')                  # 태그는 여러개에 사용 가능하나 아이디는 한번만 사용 가능함! ==> 선택하기 좋음
    soup.select('태그명.클래스명)
    soup.select('#아이디명 > 태그명.클래스명)
    soup.select('태그명[속성1=값1]')
    
    '''
    return BeautifulSoup(html, 'html.parser')


def news_article(url):
    '''
    Article을 사용하여 url을 파싱.
     - 언어가 한국어이므로 language='ko'로 설정해줍니다.
    '''
    article = Article(url, language='ko')
    article.download()
    article.parse()
    
    # 기사의 제목, 내용을 전달.
    return article.title, article.text


def make_request_header():
    '''
    url request를 위한 header를 생성.
    '''
    return {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.90 Safari/537.36'}


def request_get(url, headers=None):
    '''
    url에 request.get을 호출
    '''
    if headers is None:
        headers = make_request_header()
    
    return requests.get(url, headers=headers)


def request_post(url, headers=None, data=None):
    '''
    url에 request.get을 호출
    '''
    if headers is None:
        headers = make_request_header()
    
    return requests.post(url, headers=headers, data=data)


def make_urllist(page_num, code, date):
    '''
    페이지 수, 카테고리, 날짜를 입력값으로 받습니다.
    '''
    urllist= []
    
    for i in range(1, page_num + 1):
        url = 'https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1='+str(code)+'&date='+str(date)+'&page='+str(i)
        headers = make_request_header()
        news = request_get(url, headers)

        # BeautifulSoup의 인스턴스 생성합니다. 파서는 html.parser를 사용합니다.
        soup = BeautifulSoup(news.content, 'html.parser')

        # CASE 1
        news_list = soup.select('.newsflash_body .type06_headline li dl')
        # CASE 2
        news_list.extend(soup.select('.newsflash_body .type06 li dl'))
            
        # 각 뉴스로부터 a 태그인 <a href ='주소'> 에서 '주소'만을 가져옵니다.
        for line in news_list:
            urllist.append(line.a.get('href'))
    
    return urllist


def file_download(url, to_file):
    '''
    url로 부타 파일을 다운받아 to_file로 저장함.
    '''
    urllib.request.urlretrieve(url, to_file)


def gimg_down_from_file(data_file, save_dir, max_count=3000):
    '''
    data_file: 검색 결과 element파일.
        1. google에서 이미지 검색후, 썸네일이 보여지는 상태에서..
        2. F12로 source창을 열어서...
        3. '검색결과' 이어서 나오는 div를 찾아서 element를 복사한다.
        4. 복사된 element를 파일로 저장.

    save_dir: 크롤링한 이미지를 저장할 폴더.
    '''

    with open(data_file, "r") as f:
        tags = f.read()

    soup = BeautifulSoup(tags, 'html.parser')

    img_srcs = []
    for anchor in soup.select('img', limit=max_count):
        src = anchor.get("src")
        if src is not None:
            cls = anchor.get("class")
            if 'rg_i' == cls[0] and 'Q4LuWd' == cls[1]:
                if src.find('png') != -1:
                    pass
                elif src.find('jpg') != -1:
                    pass
                elif src.find('gif') != -1:
                    pass
                elif src.find('jpeg') != -1:
                    pass
                else:
                    img_srcs.append(src)

    print("검색결과 이미지 개수:", len(img_srcs))

    for i, img_src in enumerate(img_srcs):
        to_file = os.path.join(save_dir, f'img{i:04d}.png')
        file_download(img_src, to_file)
        print(f"\rdownload img: {i}/{len(img_srcs)}")

    print("crawling done !")


def yimg_down_from_file(data_file, save_dir, max_count=3000):
    '''
    data_file: 검색 결과 element파일.
        1. yahoo에서 이미지 검색후, 썸네일이 보여지는 상태에서..
        2. F12로 source창을 열어서... 'sres-cntr' 키워드 검색
        3. <section id="results" class="results justify".. > 아래에 있는
        4. <div class='sres-cntr'..> 엘리먼트 선택한후 복사.
        4. 복사된 element를 파일로 저장.

    save_dir: 크롤링한 이미지를 저장할 폴더.
    '''

    with open(data_file, "r") as f:
        tags = f.read()

    soup = BeautifulSoup(tags, 'html.parser')

    img_srcs = []
    for anchor in soup.select('img', limit=max_count):
        src = anchor.get("src")
        if src is not None:
            if src.endswith('300'):
                img_srcs.append(src)

    print("검색결과 이미지 개수:", len(img_srcs))

    for i, img_src in enumerate(img_srcs):
        to_file = os.path.join(save_dir, f'img{i:04d}.png')
        file_download(img_src, to_file)
        print(f"\rdownload img: {i}/{len(img_srcs)} {img_src}")

    print("crawling done !")




if __name__ == "__main__":
    """ 
    main함수.
    """

    base_path = '/home/evergrin/iu/datas'
    element_file = os.path.join(base_path, "crawl_y.txt")

    save_dir = os.path.join(base_path, 'imgs/yahoo')

    gimg_down_from_file(element_file, save_dir)
    #yimg_down_from_file(element_file, save_dir)