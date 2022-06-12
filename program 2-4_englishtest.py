# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:09:47 2022

@author: smpsm
"""

from gtts import gTTS
import googletrans
import playsound

news_en = "Premier League Breaking News 2022 04 25 13 52 Son Heung min played in 10 games in A The opponent team is 1 The two teams were shifted to one to 1 Son Heung min led the victory with one help in two goals"
print(news_en)

tts_english = gTTS(text=news_en, lang="en")
tts_english.save("test.mp3")

print("생성중..")

playsound.playsound("test.mp3", True)

import os
os.remove('test.mp3')
