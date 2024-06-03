from flask import Blueprint, redirect, jsonify

# Response, request => HTML 응답 요청을 처리하기 위함 
# render_template => HTML 파일을 렌더링
from flask import Flask, render_template, Response, request

import pandas as pd 
import json
import plotly
import plotly.express as px
import numpy as np

bp = Blueprint("main", __name__, url_prefix="/")

# @=>데코레이터 
@bp.route("/")
def index():
    # graph one
    df = px.data.medals_wide()
    fig1 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="WideForm Input")
    
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph Two
    df = px.data.iris()
    fig2 = px.scatter_3d(df, x="sepal_length", y="sepal_width", z="petal_width",
                         color="species", title="Iris")
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template("index.html", title="Home", graph1JSON=graph1JSON, graph2JSON=graph2JSON)

@bp.route('/update_chart')
def update_chart():
    print("Updating chart...")  # 로그 추가 -> python terminal에서 확인
    # 데이터 업데이트 로직 (예시)
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': np.random.randint(1, 10, 3)  # 여기에 새로운 데이터로 변경
    })
    
    new_fig = px.scatter(df, x="x", y="y")
    new_graphJSON = json.dumps(new_fig, cls=plotly.utils.PlotlyJSONEncoder)
    # return jsonify(graph1JSON)
    return jsonify({"new_graphJSON": new_graphJSON, "msg" : "화이팅!"})