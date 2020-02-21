import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.offline as py
from plotly.graph_objs import *
import plotly.graph_objs as go
from datetime import datetime
import numpy as np
import math
import geocoder
import json
from urllib.request import urlopen, quote
import requests
import os
import zipfile
py.init_notebook_mode(connected=True)
pyplt = py.offline.plot


def get_data(url,path):
    '''
    :param url: 数据源下载地址
    :param path: 当前路径
    :return: 下载数据并解压
    '''
    r = requests.get(url)
    zip_name = "master.zip"
    with open(zip_name, "wb") as code:
             code.write(r.content)
    with zipfile.ZipFile(zip_name, 'r') as zip:
        zip.extractall(path)

def get_ll(data,path):
    '''
    :param data: 数据信息
    :param path: 当前路径
    :return: 获取每个城市的经纬度信息
    '''
    city_name = list(data['城市'].unique())
    ll_data = open(path + '/城市经纬度.txt', 'r')
    ll_data1 = ll_data.read()
    dict_name = eval(ll_data1)
    ll_data.close()
    latitude = dict_name[0]
    longitude = dict_name[1]
    ak = '1PKyd56vT0mL72hnZoBR9ZCH0fPvvteA'  # ak
    output = 'json'
    url = 'http://api.map.baidu.com/geocoder/v2/'
    for ci in city_name:
        if not ci in latitude:
            try:
                add = quote(ci)
                uri = url + '?' + 'address=' + add + '&output=' + output + '&ak=' + ak  # 百度地理编码AP
                req = urlopen(uri)
                res = req.read().decode()
                temp = json.loads(res)
                if not bool(temp['status']):
                    # print(temp['result']['location']['lng'], temp['result']['location']['lat'])  # 打印出经纬度
                    lat = temp['result']['location']['lat']
                    lng = temp['result']['location']['lng']
                    latitude[ci] = lat
                    longitude[ci] = lng
                    continue
                else:
                    g = geocoder.arcgis(ci)
                    if g.latlng[0]:
                        latitude[ci] = g.latlng[0]
                        longitude[ci] = g.latlng[1]
                    else:
                        print('无{}经纬度信息'.format(ci))
            except:
                pass
    dict_name[0] = latitude
    dict_name[1] = longitude
    ll_data = open(path + '/城市经纬度.txt', 'w')
    ll_data.write(str(dict_name))
    ll_data.close()
    data['latitude'] = data['城市'].apply(lambda x: latitude[x])
    data['longitude'] = data['城市'].apply(lambda x: longitude[x])
    return data


def data_processing(path):
    '''
    :param path: 数据路径
    :return:对数据进行补齐，去空，时间转化为序列，增加聚类标签（以log5 分级）
    '''
    data = pd.read_csv(path + '/Novel-Coronavirus-Updates-master/Updates_NC.csv', encoding='gbk')
    city_update = data[['省份', '城市']].fillna(method='ffill', axis=1)
    data['城市'] = city_update['城市']
    data.fillna(value=0.0, axis=1, inplace=True)
    index_name = data.loc[(data['城市'] == '地市明细不详')|(data['城市'] == 0)]
    data.drop(index=index_name.index, axis=0, inplace=True)
    data['报道时间'] = data['报道时间'].apply(lambda x : '2020-'+x.replace('月','-').replace('日',''))
    t1 = [datetime.strptime(x, '%Y-%m-%d') for x in list(data['报道时间'])]
    data['时间'] = t1
    data['聚类标签'] = data['新增确诊'].apply(lambda x: 0 if x < 1 else int(math.log(x, 5) + 1))
    data = get_ll(data,path)
    data.to_csv(path + '\聚类后疫情数据.csv', encoding='gbk')

def prvoience_Heatmap(data):
    '''
    :param data: 处理后的数据
    :return: 以省份为单位绘出热力图
    '''
    provience_data = data.groupby(['时间', '省份']).聚类标签.max().reset_index()
    provience = provience_data.sort_values('聚类标签', ascending=False, na_position='first', inplace=False)
    provience = provience['省份'].unique()
    provience_data.sort_values('时间', ascending=True, na_position='first', inplace=True)
    pros_data = pd.DataFrame({'时间': data['时间'].unique()})
    for i in provience:
        date_data = provience_data.loc[(provience_data['省份'] == i), ['时间', '聚类标签']]
        date_data.rename(columns={'聚类标签': i}, inplace=True)
        pros_data = pd.merge(pros_data, date_data, on='时间', how='left')
    pros_data.fillna(value=0, inplace=True, axis=1)
    trace = go.Heatmap(
        z=pros_data.drop(['时间'], axis=1, inplace=False).values.tolist(),
        y=pros_data['时间'],
        x=pros_data.columns[1:],
        colorscale='Reds'
    )
    data3 = [trace]
    layout = go.Layout(
        title='疫情每日新增热力图',
        xaxis=dict(title='地区', ticks=''),
        yaxis=dict(title='时间', ticks=''),
        height=750
    )
    fig = dict(data=data3, layout=layout)
    pyplt(fig, filename='tmp/省级新增人数聚类分级热力图.html')

def hubei_Heatmap(data_new):
    '''
    :param data_new: 处理后的数据
    :return: 绘出湖北各个城市的热力图
    '''
    data_new.sort_values('聚类标签', ascending=True, inplace=True)
    city_list = data_new.loc[data_new['省份'] == '湖北', ['城市']]
    city_list = city_list['城市'].unique()
    cities_data = data_new.groupby(['时间', '城市']).聚类标签.max().reset_index()
    # print(cities_data)
    data = pd.DataFrame({'时间': data_new['时间'].unique()})
    data.sort_values('时间', ascending=True, inplace=True)
    for ci in city_list:
        city_data = cities_data.loc[(cities_data['城市'] == ci), ['时间', '聚类标签']]
        city_data.rename(columns={'聚类标签': ci}, inplace=True)
        data = pd.merge(data, city_data, on='时间', how='left')
    data.fillna(value=0, inplace=True, axis=1)
    trace = go.Heatmap(
        z=data.drop(['时间'], axis=1, inplace=False).T,
        x=data['时间'],
        y=data.columns[1:],
        colorscale='Reds'
    )
    data1 = [trace]
    layout = go.Layout(
        title='湖北疫情每日新增热力图',
        xaxis=dict(title='时间', ticks=''),
        yaxis=dict(title='地区', ticks=''),
        height=750
    )
    fig = dict(data=data1, layout=layout)
    pyplt(fig, filename='tmp/湖北新增人数聚类分级热力图.html')


def mix_3d(data_new):
    '''
    :param data_new: 处理好的数据
    :return: 绘出湖北各城市的3d效果图
    '''
    data_new.sort_values('新增确诊', ascending=True, inplace=True)
    city_list = data_new.loc[data_new['省份'] == '湖北', ['城市']]
    city_list = city_list['城市'].unique()
    cities_data = data_new.groupby(['时间', '城市']).新增确诊.max().reset_index()
    # print(cities_data)
    data = pd.DataFrame({'时间': data_new['时间'].unique()})
    data.sort_values('时间', ascending=True, inplace=True)
    for ci in city_list:
        city_data = cities_data.loc[(cities_data['城市'] == ci), ['时间', '新增确诊']]
        city_data.rename(columns={'新增确诊': ci}, inplace=True)
        data = pd.merge(data, city_data, on='时间', how='left')
    data.fillna(value=0, inplace=True, axis=1)
    marker = [[0,'white'],[5/12000,'#FDF5E6'],[25/12000,'#FF4500'],
              [125/12000,'#FF3030'],[625/12000,'#FF0000'],[1,'#8B0000']]
    #marker = ([[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
    threed = Surface(z=data.values.tolist(),y = data['时间'],
                     x = data.columns, colorscale=marker, showscale=False,
                     name = '湖北连续新增确诊人数三维展示')

    rank_date = data_new['时间'].max()
    cities_rank_data = data_new.loc[(data_new['时间'] == rank_date) &
                                    (data_new['省份'] != '湖北') , ['城市','新增确诊']]
    cities_rank_data = cities_rank_data.groupby('城市').sum()
    cities_rank_data = pd.DataFrame({'城市':list(cities_rank_data.index),
                                     '新增确诊':cities_rank_data.values.reshape(1,cities_rank_data.values.shape[0])[0]})
    print(cities_rank_data)
    cities_rank_data.sort_values('新增确诊', ascending=False, na_position='first', inplace=True)
    cities_rank_data = cities_rank_data[0:10][:]
    print(cities_rank_data)
    cities_rank = go.Bar(x = cities_rank_data['城市'],
                         y = cities_rank_data["新增确诊"],
                         marker=dict(color='#CF1020'),
                         name = '非湖北新增确诊前十地区')

    trace3 = {
        "geo": "geo3",
        "lon": data_new['longitude'],
        "lat": data_new['latitude'],
        "hoverinfo": 'text',
        "marker": {
            "size": 4,
            "opacity": 0.8,
            "color": '#CF1020',
            "colorscale": 'Viridis'
        },
        "mode": "markers",
        "type": "scattergeo"
        #,'name':'发现疫情城市' #修改name位置
    }

    data1 = go.Data([threed,trace3,cities_rank])
    layout = {
        "plot_bgcolor": 'black',
        "paper_bgcolor": 'black',
        "titlefont": {
            "size": 20,
            "family": "Raleway"
        },
        "font": {
            "color": 'white'
        },
        "dragmode": "zoom",
        "geo3": {
            "domain": {
                "x": [0, 0.55],
                "y": [0, 0.9]
            },
            "lakecolor": "rgba(127,205,255,1)",
            "oceancolor": "rgb(6,66,115)",
            "landcolor": 'white',
            "projection": {"type": "orthographic"},
            "scope": "world",
            "showlakes": True,
            "showocean": True,
            "showland": True,
            "bgcolor": 'black'
        },
        "margin": {  #留白大小
            "r": 10, #右
            "t": 25, #上
            "b": 40, #下
            "l": 60  #左
        },
        "scene": {"domain": {  #surface setting
            "x": [0.5, 1],   #surface画布区域
            "y": [0, 0.55]
        },
            "xaxis": {"gridcolor": 'white'},
            "yaxis": {"gridcolor": 'white'},
            "zaxis": {"gridcolor": 'white'}
        },
        "showlegend": True,
        "title": "<br>{} 疫情数据三维可视化".format(rank_date),
        "xaxis": {
            "anchor": "y",
            "domain": [0.6, 0.95]
        },
        "yaxis": {
            "anchor": "x",
            "domain": [0.65, 0.95],
            "showgrid": False
        }
    }

    annotations = {"text": "data from 美数课",
                   "showarrow": False,
                   "xref": "paper",
                   "yref": "paper",
                   "x": 0,
                   "y": 0}

    layout['annotations'] = [annotations]
    fig = go.Figure(data=data1, layout=layout)
    pyplt(fig, filename="tmp/3d-疫情数据可视化.html")

url = 'https://github.com/839-Studio/Novel-Coronavirus-Updates/archive/master.zip'
path = os.getcwd()
print('是否更新数据')
update = str(input('(y/n):'))
if update == 'y':
    get_data(url, path)
    data_processing(path)
data_new = pd.read_csv(path + '\聚类后疫情数据.csv', encoding='gbk')

print('输入：\n1:省级新增人数聚类分级热力图\n2：湖北新增人数聚类分级热力图\n其他:3d混合展示')
num = str(input(':'))
if num == '1':
    prvoience_Heatmap(data_new)
elif num == '2':
    hubei_Heatmap(data_new)
else:
    mix_3d(data_new)