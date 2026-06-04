import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px  

st.set_page_config(page_title="信贷风险评分系统", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
explainer = shap.TreeExplainer(model)

st.title("🏦 信贷违约风险评分系统")
st.caption("基于 XGBoost 模型 · AUC = 0.87 · 数据来源：Give Me Some Credit")
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 客户信息输入")
    revolving   = st.slider("信用卡额度使用率", 0.0, 1.5, 0.3, 0.01)
    age         = st.slider("年龄", 18, 100, 45)
    late_30     = st.number_input("30–59天逾期次数", 0, 20, 0)
    late_60     = st.number_input("60–89天逾期次数", 0, 20, 0)
    late_90     = st.number_input("90天以上逾期次数", 0, 20, 0)
    debt_ratio  = st.slider("负债比率", 0.0, 2.0, 0.35, 0.01)
    income      = st.number_input("月收入（元）", 0, 200000, 5400)
    open_credit = st.slider("信用账户数量", 0, 30, 8)
    real_estate = st.slider("房产贷款数量", 0, 10, 1)
    dependents  = st.slider("家庭人口数", 0, 10, 0)

input_df = pd.DataFrame([{
    "RevolvingUtilizationOfUnsecuredLines": revolving,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": income,
    "NumberOfOpenCreditLinesAndLoans": open_credit,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": real_estate,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60,
    "NumberOfDependents": dependents
}])

prob = model.predict_proba(input_df)[0][1]

with col2:
    st.subheader("📊 风险评估结果")
    if prob < 0.15:
        level, icon = "低风险", "🟢"
    elif prob < 0.35:
        level, icon = "中等风险", "🟡"
    else:
        level, icon = "高风险", "🔴"

    st.metric("违约概率", f"{prob:.1%}")
    st.markdown(f"### {icon} {level}")
    st.progress(float(prob))
    st.divider()
    
    st.markdown("**关键影响因素分析**")
    st.caption("🔴 红色=增加违约风险 🟢 绿色=降低违约风险")

    feature_map = {
        "RevolvingUtilizationOfUnsecuredLines": "信用卡额度使用率",
        "age": "年龄",
        "NumberOfTime30-59DaysPastDueNotWorse": "30-59天逾期次数",
        "DebtRatio": "负债比率",
        "MonthlyIncome": "月收入",
        "NumberOfOpenCreditLinesAndLoans": "信用账户数量",
        "NumberOfTimes90DaysLate": "90天以上逾期次数",
        "NumberRealEstateLoansOrLines": "房产贷款数量",
        "NumberOfTime60-89DaysPastDueNotWorse": "60-89天逾期次数",
        "NumberOfDependents": "家庭人口数"
    }

    shap_vals = explainer.shap_values(input_df)[0]
    features  = list(input_df.columns)
    top_idx   = np.argsort(np.abs(shap_vals))[-6:]

    plot_df = pd.DataFrame({
        "特征": [feature_map[features[i]] for i in top_idx],
        "SHAP值": [shap_vals[i] for i in top_idx],
        "影响方向": ["增加风险" if shap_vals[i] > 0 else "降低风险" for i in top_idx]
    })

    fig = px.bar(
        plot_df, 
        x="SHAP值", 
        y="特征", 
        orientation='h',
        color="影响方向",
        color_discrete_map={"增加风险": "#E24B4A", "降低风险": "#1D9E75"}
    )
    
    fig.update_layout(
        showlegend=False, 
        margin=dict(l=0, r=0, t=0, b=0), 
        height=300,
        xaxis_title=None,
        yaxis_title=None
    )
    
    st.plotly_chart(fig, use_container_width=True)
