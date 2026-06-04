import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.express as px

# 1. 页面配置 (更专业的布局)
st.set_page_config(page_title="智能信贷审批工作台", page_icon="🏦", layout="wide")

@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
explainer = shap.TreeExplainer(model)

st.title("🏦 智能信贷审批工作台 (风控版)")
st.caption("基于 XGBoost 算法 · 融合专家业务规则 · 支持可视化归因与沙盘推演")
st.divider()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📋 客户特征输入与沙盘推演")
    
    # 【升级点 3：沙盘推演模式开关】
    sandbox_mode = st.toggle("🧪 开启沙盘推演模式 (动态测算参数调整对风险的影响)", value=False)
    if sandbox_mode:
        st.info("沙盘模式已开启：请滑动下方参数，右侧风控结果将实时重新计算。")

    revolving   = st.slider("信用卡额度使用率", 0.0, 1.5, 0.65, 0.01) # 默认设高一点演示熔断
    age         = st.slider("年龄", 18, 100, 45)
    late_30     = st.number_input("30–59天逾期次数", 0, 20, 0)
    late_60     = st.number_input("60–89天逾期次数", 0, 20, 0)
    late_90     = st.number_input("90天以上逾期次数", 0, 20, 1)
    debt_ratio  = st.slider("负债比率", 0.0, 2.0, 0.35, 0.01)
    income      = st.number_input("月收入（元）", 0, 200000, 5400)
    open_credit = st.slider("信用账户数量", 0, 30, 8)
    real_estate = st.slider("房产贷款数量", 0, 10, 1)
    dependents  = st.slider("家庭人口数", 0, 10, 0)

# 构建预测数据
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

# 模型预测与解释
prob = model.predict_proba(input_df)[0][1]
shap_vals = explainer.shap_values(input_df)[0]

with col2:
    st.subheader("📊 智能风控评估报告")
    
    # 【升级点 1：专家业务规则熔断 (Business Rule Engine)】
    # 无论 AI 算出来是多少，触碰红线直接拒贷
    is_rejected_by_rule = False
    reject_reason = ""
    
    if late_90 >= 2:
        is_rejected_by_rule, reject_reason = True, "历史严重逾期 (90天以上逾期≥2次)"
    elif debt_ratio > 1.2:
        is_rejected_by_rule, reject_reason = True, "负债过高 (负债比率>120%)"
        
    if is_rejected_by_rule:
        st.error(f"### 🔴 触发风控熔断：拒绝授信\n**触发原因**：{reject_reason}")
        st.warning(f"注：该客户机器评估违约概率为 {prob:.1%}，但因触发硬性合规规则，系统自动拦截。")
    else:
        if prob < 0.15:
            level, icon, color = "优质客户 (低风险)", "🟢", "normal"
        elif prob < 0.35:
            level, icon, color = "边缘客户 (中等风险)", "🟡", "off"
        else:
            level, icon, color = "高危客户 (高风险)", "🔴", "inverse"

        st.metric("AI 预测违约概率", f"{prob:.1%}")
        st.markdown(f"### {icon} {level}")
        st.progress(float(prob))

    st.divider()

    # 可视化部分 (保留了没有方块字的 Plotly 图表)
    st.markdown("**🔍 归因分析 (SHAP 模型解释)**")
    
    feature_map = {
        "RevolvingUtilizationOfUnsecuredLines": "信用卡额度使用率",
        "age": "年龄",
        "NumberOfTime30-59DaysPastDueNotWorse": "30-59天逾期",
        "DebtRatio": "负债比率",
        "MonthlyIncome": "月收入",
        "NumberOfOpenCreditLinesAndLoans": "信用账户数",
        "NumberOfTimes90DaysLate": "90天以上逾期",
        "NumberRealEstateLoansOrLines": "房产贷款数",
        "NumberOfTime60-89DaysPastDueNotWorse": "60-89天逾期",
        "NumberOfDependents": "家庭人口数"
    }

    features  = list(input_df.columns)
    top_idx   = np.argsort(np.abs(shap_vals))[-5:] # 取前5大影响因素

    plot_df = pd.DataFrame({
        "特征": [feature_map[features[i]] for i in top_idx],
        "SHAP值": [shap_vals[i] for i in top_idx],
        "影响方向": ["增加风险 ⬆️" if shap_vals[i] > 0 else "降低风险 ⬇️" for i in top_idx]
    })

    fig = px.bar(
        plot_df, x="SHAP值", y="特征", orientation='h', color="影响方向",
        color_discrete_map={"增加风险 ⬆️": "#E24B4A", "降低风险 ⬇️": "#1D9E75"}
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=250, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # 【升级点 2：智能改善建议 (Next Best Action)】
    if not is_rejected_by_rule and prob > 0.15:
        st.divider()
        st.markdown("**💡 智能改善建议 (Next Best Action)**")
        
        # 找到增加风险最多的特征
        max_risk_idx = np.argmax(shap_vals)
        max_risk_feature = features[max_risk_idx]
        
        advice_dict = {
            "RevolvingUtilizationOfUnsecuredLines": "系统检测到该客户**信用卡额度使用率过高**。建议客户结清部分账单，将使用率降至 30% 以下，可显著降低违约评分。",
            "NumberOfTime30-59DaysPastDueNotWorse": "客户近期有**轻微逾期记录**。建议要求客户提供资产证明，或增加抵押物以覆盖风险。",
            "NumberOfTimes90DaysLate": "客户有**严重逾期历史**。属于高危特征，建议转交人工信贷专家进行深度背景调查。",
            "DebtRatio": "客户**负债比率较高**。建议降低本次授信额度，或要求客户合并其他高息贷款。"
        }
        
        advice = advice_dict.get(max_risk_feature, "建议关注客户近期现金流状况，可要求增加担保人。")
        st.info(advice)
