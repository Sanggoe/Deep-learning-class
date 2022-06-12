# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:21:32 2022

@author: smpsm
"""

import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='petal_length', y='sepal_width', z='petal_width',
                    color='species')
fig.show(renderer="browser")

'''
먼저 sepal length의 경우, setosa와 versicolor,
versicolor와 virginica 각각이 겹치는 부분이 많이 생겨
분별력이 좋지 않은 편이었다.

그에 비해 sepal length대신 petal length를 추가하여
분포를 출력한 결과, 상대적으로 좀 더 분명하게 구분이 되는 것을
확인할 수 있었다.
versicolor와 virginica가 약간 겹치는 부분이 있긴 하지만,
setosa는 완전히 구분이 있고, 전체적으로 보았을 때도
꽤나 분별력 있는 특성으로 보여진다.
''' 