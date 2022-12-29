VScode 설치
===========

https://code.visualstudio.com/

링크타고 들어가 설치 진행.

![2](https://user-images.githubusercontent.com/112140363/209625787-a5d4bb83-9dcb-4006-a384-5098430c1520.png)

간단히 필요한 것들 예시.

이름: Python

ID: ms-python.python
설명: IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), Jupyter Notebooks, code formatting, refactoring, unit tests, and more.
버전: 2022.20.1
게시자: Microsoft
VS Marketplace 링크: https://marketplace.visualstudio.com/items?itemName=ms-python.python

이름: Pylance

ID: ms-python.vscode-pylance
설명: A performant, feature-rich language server for Python in VS Code
버전: 2022.12.20
게시자: Microsoft
VS Marketplace 링크: https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance
https://demun.github.io/vscode-tutorial/debug/

이름: Korean Language Pack for Visual Studio Code

ID: MS-CEINTL.vscode-language-pack-ko
설명: Language pack extension for Korean
버전: 1.74.12140931
게시자: Microsoft
VS Marketplace 링크: https://marketplace.visualstudio.com/items?itemName=MS-CEINTL.vscode-language-pack-ko
튜토리얼 링크 들어가서 확인

이름: Jupyter

ID: ms-toolsai.jupyter
설명: Jupyter notebook support, interactive programming and computing that supports Intellisense, debugging and more.
버전: 2022.11.1003412109
게시자: Microsoft
VS Marketplace 링크: https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter

필요한 확장 도구들 설치

##GIT 설치


-------------------------------------------------------------------------------------------------------------------------

GIT 을 이용하여 파일 불러오는 방법 2가지

1. GIT clone 
- clone 코드를 이용하여 데이터 불러오는 방법

2. download ZIP
- ZIP 파일을 받아 파일을 옮기는 방법

필자는 2번 방법을 통해 파일을 옮겼다.

* 2번 방법으로 했을 시 유의점
파일을 새로 받아 옮기는 작업이기 때문에 안에 필요한 작업들을 따로 해주어야 한다.
a. 현재 window에서 frame_maker 를 실행하기 위해 같은 폴더안에 py파일들을 모아두어야한다.
b. 파일 내에 문서 내용을 고쳐야한다.
- classes, utils 이 들어간 파일 경로 삭제.

(수작업으로 일일히 해주어야한다는 점이 불편, 파일 경로를 읽지 못해 발생하는 문제인거 같다, 해결하지 못해 수작업을 사용중)

추가 
GIT 기본 명령어 몇가지..

현재 상태 확인 
- git status
 
전체 로그 확인 
- git log
 
git 저장소 생성하기 
- git init
 
저장소 복제 및 다운로드 
- git clone [https: ~~~~ ]
 
저장소에 코드 추가
- git add
- git add *
 
커밋에 파일의 변경 사항을 한번에 모두 포함 
- git add -A
 
커밋 생성
- git commit -m "message"
 
변경 사항 원격 서버 업로드 (push)
- git push origin master

원격 저장소의 변경 내용을 현재 디렉토리로 가져오기 (pull)
- git pull
 
변경 내용을 merge 하기 전에 바뀐 내용 비교
- git diff [브랜치 이름] [다른 브랜치 이름]

!path 설정 ghostscript설정
