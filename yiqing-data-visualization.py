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
from scipy.optimize import curve_fit
py.init_notebook_mode(connected=True)
pyplt = py.offline.plot

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
            print('更新经纬度：',ci)
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
                    print('百度无{0}经纬度'.format(ci))
                    g = geocoder.arcgis(ci)
                    if g.latlng[0]:
                        latitude[ci] = g.latlng[0]
                        longitude[ci] = g.latlng[1]
                    else:
                        print('geocoder无{0}经纬度'.format(ci))
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
    data = pd.read_csv(path + '/Novel-Coronavirus-Updates-master/Updates_NC.csv', encoding='utf-8')
    city_update = data[['省份', '城市']].fillna(method='ffill', axis=1)
    data['城市'] = city_update['城市']
    data.fillna(value=0.0, axis=1, inplace=True)
    index_name = data.loc[(data['城市'].isin(['地市明细不详','钻石公主号',0]))]  # == '地市明细不详')|(data['城市'] == 0)]
    data.drop(index=index_name.index, axis=0, inplace=True)
    data['报道时间'] = data['报道时间'].apply(lambda x : '2020-'+x.replace('月','-').replace('日',''))
    t1 = [datetime.strptime(x, '%Y-%m-%d') for x in list(data['报道时间'])]
    data['时间'] = t1
    data['聚类标签'] = data['新增确诊'].apply(lambda x: 0 if x < 1 else int(math.log(x, 5) + 1))
    data = get_ll(data,path)
    data.to_csv(path + '\聚类后疫情数据.csv', encoding='gbk')

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
    print('已获取数据')
    data_processing(path)

def prvoience_Heatmap(data):
    '''
    :param data: 处理后的数据
    :return: 以省份为单位绘出热力图
    '''
    provience_data = data.groupby(['时间', '省份']).聚类标签.max().reset_index()
    pros_data = provience_data.pivot(index = '时间',values = '聚类标签',columns = '省份')
    pros_data.fillna(value=0, inplace=True, axis=1)
    trace = go.Heatmap(
        z=pros_data.values.tolist(),
        y=pros_data.index,
        x=pros_data.columns,
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
    city_list = data_new.loc[data_new['省份'] == '湖北', ['时间','城市','聚类标签']]
    city_list = city_list.groupby(['时间', '城市']).聚类标签.max().reset_index()
    data = city_list.pivot(index='时间', values='聚类标签', columns='城市')
    data.fillna(value=0, inplace=True, axis=1)
    trace = go.Heatmap(
        z=data.T,
        x=data.index,
        y=data.columns,
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
    #data_new.sort_values('新增确诊', ascending=True, inplace=True)
    city_list = data_new.loc[data_new['省份'] == '湖北', ['时间', '城市', '新增确诊']]
    city_list = city_list.groupby(['时间', '城市']).新增确诊.max().reset_index()
    city_list.sort_values('新增确诊', ascending=True, inplace=True)
    data = city_list.pivot(index='时间', values='新增确诊', columns='城市')
    data.fillna(value=0, inplace=True, axis=1)
    marker = [[0,'white'],[5/12000,'#FDF5E6'],[25/12000,'#FF4500'],
              [125/12000,'#FF3030'],[625/12000,'#FF0000'],[1,'#8B0000']]
    #marker = ([[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']])
    threed = Surface(z=data.values.tolist(),y = data.index,
                     x = data.columns, colorscale=marker, showscale=False,
                     name = '湖北连续新增确诊人数三维展示')

    rank_date = data_new['时间'].max()
    cities_rank_data = data_new.loc[(data_new['时间'] == rank_date), ['城市','新增确诊']]
    cities_rank_data = cities_rank_data.groupby('城市')['新增确诊'].sum().nlargest(10)
    cities_rank = go.Bar(x = cities_rank_data.index,
                         y = cities_rank_data.values,
                         marker=dict(color='#CF1020'),
                         name = '新增确诊人数前十地区'
                         )

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

    annotations = {"text": "感谢 美数课 对数据的收集整理",
                   "showarrow": False,
                   "xref": "paper",
                   "yref": "paper",
                   "x": 0,
                   "y": 0}

    layout['annotations'] = [annotations]
    fig = go.Figure(data=data1, layout=layout)
    pyplt(fig, filename="tmp/3d-疫情数据可视化.html")

def logistic_increase_function1(t,K,P0,r):
    '''
    使用逻辑增长法计算新增确诊数据
    :param t:
    :param K:
    :param P0:
    :param r:
    :return:
    '''
    t0=0
    # r = r
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    exp_value=np.exp(r*(t-t0))
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)

def logistic_increase_function2(t,K,P0,r):
    '''
    使用逻辑增长法计算新增治愈数据
    :param t:标准化时间序列
    :param K:
    :param P0:
    :param r:参数
    :return: 逻辑增长函数计算出的数据
    '''
    t0=0
    r = 0.148
    # t:time   t0:initial time    P0:initial_value    K:capacity  r:increase_rate
    exp_value=np.exp(r*(t-t0))
    return (K*exp_value*P0)/(K+(exp_value-1)*P0)

def curve( x_train, y_train,n,x,r):
    '''
    :param x_train:标准化日期
    :param y_train:训练目标数据
    :param n:预测未来天数
    :param x:日期列表
    :param r:逻辑增长函数参数
    :return:拟合数据，预测数据
    '''
    if r == 0:
        logistic_increase_function = logistic_increase_function1
    else:
        logistic_increase_function = logistic_increase_function2
    popt, pcov = curve_fit(logistic_increase_function, x_train, y_train)
    # print("K:capacity  P0:initial_value   r:increase_rate   t:time")
    # print(popt)
    # 拟合后预测的P值
    y_predict = logistic_increase_function(x_train, popt[0], popt[1],popt[2])
    y_predict = y_predict.astype(int)
    # 未来预测
    x_test = np.array([x for x in range(len(x), len(x) + n)])
    y_test = logistic_increase_function(x_test, popt[0], popt[1],popt[2])
    y_test = y_test.astype(int)
    return y_predict,y_test

def future_grouth(data):
    '''
    对疫情数据进行预测并可视化
    :param data: 处理好的数据
    :return:预测数据,变化趋势可视化展现
    '''
    chinaarea = '''
        '重庆' '新疆' '山西' '河南' '黑龙江' '台湾' '海南' '陕西'  '吉林' '江苏' '福建' '宁夏' '江西' '辽宁' '河北' '香港' '安徽' '广西' '山东' '湖北' '四川' '浙江' '湖南'  '北京'  '贵州' '广东' '天津' '上海' '云南' '甘肃' '内蒙古'  '澳门' '青海' '西藏' 
     '''
    areaname = chinaarea.replace('\'', '').split(' ')
    data = data.loc[data['省份'].isin(areaname)]  # 只保留国内
    date_list = list(data['时间'].unique())
    x = []
    y = []
    z = []
    while bool(date_list):
        x.append(date_list.pop())
        y.append(data.loc[data['时间'].isin(x)]['新增确诊'].sum())
        z.append(data.loc[data['时间'].isin(x)]['新增出院'].sum())
    x_train = np.array([x for x in range(len(x))])
    y_train = np.array(y)
    z_train = np.array(z)
    n = 25
    num_y = 0
    num_z = 0
    while num_z <= num_y:
        y_predict, y_test = curve(x_train, y_train, n, x, 0)
        z_predict, z_test = curve(x_train, z_train, n, x, 1)
        num_y = y_test[-1]
        num_z = z_test[-1]
        n += 1

    trace1 = go.Scatter(
        x=x,
        y=y,
        name='确诊总数'
    )
    trace2 = go.Scatter(
        x=x,
        y=y_predict,
        name='拟合确诊'
    )
    x1 = pd.date_range(x[-1], periods=n + 1, freq='1d')
    y1 = np.append(y_predict[-1], y_test)
    trace3 = go.Scatter(
        x=x1,
        y=y1,
        name='预测确诊'
    )
    trace4 = go.Scatter(
        x=x,
        y=z,
        name='出院总数'
    )
    trace5 = go.Scatter(
        x=x,
        y=z_predict,
        name='拟合出院'
    )
    z1 = np.append(z_predict[-1], z_test)
    trace6 = go.Scatter(
        x=x1,
        y=z1,
        name='预测出院'
    )
    data3 = [trace1, trace2, trace3,trace4, trace5, trace6]
    layout1 = dict(title='疫情趋势预测',
                   yaxis=dict(zeroline=True),
                   xaxis=dict(zeroline=False))
    fig1 = dict(data=data3, layout=layout1)

    futuere_y = y1[1:] - y1[:-1]
    futuere_z = z1[1:] - z1[:-1]
    trace7 = go.Bar(
        x=x1[1:],
        y=futuere_y,
        name = '新增确诊'
    )
    trace8 = go.Bar(
        x=x1[1:],
        y=futuere_z,
        name = '新增出院'
    )
    data4 = [trace7,trace8]
    fig2 = go.Figure(data=data4)
    pyplt(fig2, filename='tmp/预测数据.html')
    pyplt(fig1, filename='tmp/变化趋势.html')

def main():
    url = 'https://github.com/839-Studio/Novel-Coronavirus-Updates/archive/master.zip'
    path = os.getcwd()
    print('是否更新数据')
    update = str(input('(y/n):'))
    if update == 'y':
        get_data(url, path)
    data_new = pd.read_csv(path + '\聚类后疫情数据.csv', encoding='gbk')
    while True:
        print('''
        输入：\n1:省级新增人数聚类分级热力图
        \n2：湖北新增人数聚类分级热力图
        \n3:3d混合展示
        \n4:新增确诊预测
        ''')
        num = str(input(':'))
        if num == '1':
            prvoience_Heatmap(data_new)
        elif num == '2':
            hubei_Heatmap(data_new)
        elif num == '3':
            mix_3d(data_new)
        elif num == '4':
            future_grouth(data_new)
        else:
            break

main()