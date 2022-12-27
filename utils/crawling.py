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
    
    
if __name__ == "__main__":
    """ 
    main함수.
    """
    import os
    #https://www.google.com/search?q=child+drawing+lion+images+no+color&client=ubuntu&hs=RT5&sxsrf=ALiCzsbkTl2FEX-dIYkkld-XjU9tHsdy_A:1668698620494&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjVwovkwrX7AhV7mlYBHfmMApIQ_AUoAXoECAIQAw&biw=1490&bih=758&dpr=1.25
    # img_url = 'https://www.google.com/search?q=child+drawing+lion+images+no+color&oq=child+drawing+lion+images+no+color&aqs=chrome..69i57j33i160l2.13843j0j15&client=ubuntu&sourceid=chrome&ie=UTF-8'

    # response = request_get(img_url)
    # data = response.content.decode()
    
    base_path = './datas/'
    res_file = os.path.join(base_path, "crawl.txt")
    with open(res_file, "r") as f:
        data = f.read()

    soup = BeautifulSoup(data, 'html.parser')

    img_srcs = []
    for anchor in soup.select('img', limit=3000):
        src = anchor.get("src")
        if src is not None:
            cls = anchor.get("class")
            if 'rg_i' == cls[0] and 'Q4LuWd' == cls[1]:
                if src.find('png') != -1:
                    pass
                    # img_srcs.append(src)
                elif src.find('jpg') != -1:
                    pass
                    # img_srcs.append(src)
                elif src.find('gif') != -1:
                    pass
                    # img_srcs.append(src)
                elif src.find('jpeg') != -1:
                    pass
                    # img_srcs.append(src)
                else:
                    img_srcs.append(src)
                    print(src)
                    # pass
                # if src.startswith("data:image"):
                #     img_srcs.append(src)

            # anchor = anchor.find("img")
            # img_srcs.append(anchor.get("src")+"\n")

    print("검색결과 이미지:", len(img_srcs))
    for i, img_src in enumerate(img_srcs):
        to_file = os.path.join(base_path, 'img%04d.png'%(i))
        file_download(img_src, to_file)

