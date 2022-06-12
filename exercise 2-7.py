# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:08:39 2022

@author: smpsm
"""

from datetime import datetime

# 경기 결과 입력
place = input("경기가 열린 곳은? ")
time = input("경기가 열린 시간은? ")
opponent = input("상대팀은? ")
goals = input("손흥민은 몇 골을 넣었나요? ")
aids = input("도움은 몇 개인가요? ")
score_me = input("손흥민 팀이 넣은 골 수는? ")
score_you = input("상대 팀이 넣은 골 수는? ")

# 기사 작성하는 곳
news = "\n\n[프리미어 리그 속보(" + str(datetime.now().strftime("%Y-%m-%d %H:%M")) + ")]\n"
news += "손흥민 선수는 " + place + "에서 " + time + "에 열린 경기에 출전하였습니다. "
news += "상대 팀은 " + opponent + "입니다. "

if score_me > score_you:
    news += "손흥민 선수의 팀이 " + score_me + "골을 넣어 " + score_you + "골을 넣은 상대 팀을 이겼습니다. "
elif score_me < score_you:
    news += "손흥민 선수의 팀이 " + score_me + "골을 넣어 " + score_you + "골을 넣은 상대 팀에게 졌습니다. "
else:
    news += "두 팀은 " + score_me + "대" + score_you + "로 비겼습니다. "
    
if int(goals) > 0 and int(aids) > 0:
    news += "손흥민 선수는 " + goals + "골에 도움 " + aids + "개로 승리를 크게 이끌었습니다."
elif int(goals) > 0 and int(aids) == 0:
    news += "손흥민 선수는 " + goals + "골로 승리를 이끌었습니다."
elif int(goals) == 0 and int(aids) > 0:
    news += "손흥민 선수는 골은 없지만 도움 " + aids + "개로 승리하는데 공헌하였습니다."
else:
    news += "아쉽게도 이번 경기에서 손흥민의 발끝은 침묵을 지켰습니다."
    
print(news)
print()



from gtts import gTTS
import googletrans
import playsound

# **** 기존의 음성파일을 반드시 삭제 후 실행시킬 것 ****

# 한국어 음성 저장 및 출력
tts = gTTS(text=news, lang="ko")
tts.save("news_Son.mp3")
playsound.playsound("news_Son.mp3", True)

import os
os.remove('news_Son.mp3')



# 영어 번역
translator = googletrans.Translator()
news_en = translator.translate(news, dest='en').text
print(news_en)


# 영어 음성 저장 및 출력
tts_english = gTTS(text=news_en, lang="en")
tts_english.save("english_news.mp3")
playsound.playsound("english_news.mp3", True)

import os
os.remove('english_news.mp3')