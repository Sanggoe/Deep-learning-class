# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:21:32 2022

@author: smpsm
"""

import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species')
fig.show(renderer="browser")