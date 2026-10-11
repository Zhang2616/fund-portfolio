"""
基金风险资产组合优化系统
功能：从雪球/同花顺获取基金数据 → 计算最优风险组合 → 计算最优投资比例 → Web展示
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
import pysnowball as ball
from datetime import datetime, timedelta

# ============ 页面配置 ============
st.set_page_config(page_title="基金组合优化器", page_icon="📊", layout="wide")
st.title("📊 基金风险资产组合优化系统")
st.markdown("基于马科维茨投资组合理论，自动获取基金数据并计算最优配置方案")

# ============ 侧边栏：参数设置 ============
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 基金代码输入（雪球格式）
    fund_codes_input = st.text_area(
        "输入基金代码（每行一个，雪球格式）",
        value="F008975\nF000001\nF110011",
        help="雪球基金代码格式：F+基金代码，如 F008975 表示 008975 基金"
    )
    
    # 无风险利率
    rf_rate = st.number_input(
        "无风险利率（年化，%）", 
        value=2.5, step=0.1, format="%.1f"
    ) / 100
    
    # 数据时间段
    years_back = st.slider("获取历史数据年数", 1, 5, 3)
    
    # 雪球Token
    xq_token = st.text_input(
        "雪球Token（可选）", 
        type="password",
        help="从浏览器Cookie中获取xq_a_token值，部分接口需要"
    )
    
    # 投资者风险厌恶系数
    risk_aversion = st.slider(
        "风险厌恶系数 A", 1.0, 10.0, 3.0, 0.5,
        help="A越大表示越厌恶风险，A=1最激进，A=10最保守"
    )
    
    run_btn = st.button("🚀 开始计算", type="primary", use_container_width=True)


# ============ 数据获取函数 ============
@st.cache_data(ttl=3600)  # 缓存1小时，避免重复请求
def fetch_fund_nav(fund_code, token=None, years=3):
    """
    从雪球获取基金净值历史数据
    
    参数:
        fund_code: 雪球基金代码，如 "F008975"
        token: 雪球API Token
        years: 获取多少年的历史数据
    返回:
        DataFrame，包含日期和单位净值
    """
    if token:
        ball.set_token(f'xq_a_token={token};')
    
    try:
        # 使用pysnowball的fund_nav_history接口获取基金净值
        # 该接口返回基金的历史净值数据
        code = fund_code.replace("F", "")
        result = ball.fund_nav_history(code, page=1, size=years * 250)
        
        if result and 'data' in result:
            nav_data = result['data']
            if 'list' in nav_data:
                df = pd.DataFrame(nav_data['list'])
                # 解析日期和净值字段
                df['date'] = pd.to_datetime(df['date'])
                df = df[['date', 'unit_nav']].rename(
                    columns={'date': '日期', 'unit_nav': '净值'}
                )
                df = df.sort_values('日期').reset_index(drop=True)
                return df
        return None
    except Exception as e:
        st.warning(f"获取基金 {fund_code} 数据失败: {str(e)}")
        return None


def fetch_multiple_funds(fund_codes, token, years):
    """批量获取多只基金的净值数据并合并"""
    all_data = {}
    for code in fund_codes:
        code = code.strip()
        if not code:
            continue
        df = fetch_fund_nav(code, token, years)
        if df is not None and len(df) > 0:
            all_data[code] = df.set_index('日期')['净值']
    
    if not all_data:
        return None
    
    # 合并为统一DataFrame
    combined = pd.DataFrame(all_data)
    combined = combined.dropna()
    return combined


# ============ 组合优化核心函数 ============
def calculate_returns(nav_df):
    """根据净值数据计算日收益率序列"""
    returns = nav_df.pct_change().dropna()
    return returns


def portfolio_stats(weights, mean_returns, cov_matrix):
    """计算组合的期望收益和风险（标准差）"""
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_risk = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )
    return portfolio_return, portfolio_risk


def negative_sharpe(weights, mean_returns, cov_matrix, rf):
    """负夏普比率（用于最小化）"""
    p_ret, p_risk = portfolio_stats(weights, mean_returns, cov_matrix)
    return -(p_ret - rf) / p_risk


def optimize_tangency_portfolio(mean_returns, cov_matrix, rf):
    """
    计算最优风险资产组合（切点组合）
    通过最大化夏普比率得到最优权重 wd 和 we
    """
    n = len(mean_returns)
    init_weights = np.ones(n) / n  # 等权重初始化
    bounds = tuple((0, 1) for _ in range(n))  # 不允许做空
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
    
    result = minimize(
        negative_sharpe, init_weights,
        args=(mean_returns, cov_matrix, rf),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )
    
    if result.success:
        optimal_weights = result.x
        p_return, p_risk = portfolio_stats(
            optimal_weights, mean_returns, cov_matrix
        )
        sharpe = (p_return - rf) / p_risk
        return {
            'weights': optimal_weights,
            'expected_return': p_return,
            'risk': p_risk,
            'sharpe_ratio': sharpe,
            'success': True
        }
    else:
        return {'success': False, 'message': result.message}


def calculate_optimal_allocation(p_return, p_risk, rf, risk_aversion):
    """
    计算完整资产组合中风险资产的最优投资比例 y
    
    公式: y* = (E(rp) - rf) / (A × σp²)
    
    其中:
        E(rp): 风险组合期望收益
        rf:   无风险利率
        A:    风险厌恶系数
        σp:   风险组合标准差
    """
    if p_risk == 0:
        return 0
    
    # 最优风险资产投资比例
    y = (p_return - rf) / (risk_aversion * p_risk ** 2)
    y = np.clip(y, 0, 2)  # 限制在0~200%（允许杠杆但限制上限）
    
    # 完整组合的期望收益和风险
    complete_return = rf + y * (p_return - rf)
    complete_risk = y * p_risk
    
    # 各基金在完整组合中的最终投资比例
    # 如果 y > 1，表示借入无风险资产投资风险资产
    final_weights = y * np.array([p_return, p_risk])  # 占位
    
    return {
        'y': y,
        'complete_return': complete_return,
        'complete_risk': complete_risk,
        'risk_free_weight': 1 - y,
        'is_leveraged': y > 1
    }


def generate_efficient_frontier(mean_returns, cov_matrix, n_points=50):
    """生成有效前沿"""
    n = len(mean_returns)
    results = []
    
    # 先求最小方差组合
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    min_var_result = minimize(
        portfolio_variance, np.ones(n)/n,
        method='SLSQP', bounds=bounds, constraints=constraints
    )
    min_var_risk = np.sqrt(min_var_result.fun)
    min_var_ret, _ = portfolio_stats(
        min_var_result.x, mean_returns, cov_matrix
    )
    
    max_ret = np.max(mean_returns)
    
    for target_ret in np.linspace(min_var_ret, max_ret, n_points):
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.sum(w * mean_returns) - target_ret}
        )
        res = minimize(
            portfolio_variance, np.ones(n)/n,
            method='SLSQP', bounds=bounds, constraints=cons
        )
        if res.success:
            _, risk = portfolio_stats(res.x, mean_returns, cov_matrix)
            results.append({'risk': risk, 'return': target_ret})
    
    return pd.DataFrame(results)


# ============ 主逻辑 ============
if run_btn:
    fund_codes = [c.strip() for c in fund_codes_input.split('\n') if c.strip()]
    
    if len(fund_codes) < 2:
        st.error("请至少输入2只基金代码")
        st.stop()
    
    # --- Step 1: 获取数据 ---
    with st.spinner("正在从雪球获取基金净值数据..."):
        nav_df = fetch_multiple_funds(fund_codes, xq_token, years_back)
    
    if nav_df is None or nav_df.empty:
        st.error("数据获取失败。请检查基金代码是否正确、网络是否可用，或是否需要在Token栏填入雪球Token。")
        st.info("💡 提示：雪球部分接口需要登录Token。获取方法：登录xueqiu.com → F12开发者工具 → Application → Cookies → 复制xq_a_token的值")
        st.stop()
    
    # --- Step 2: 计算收益率和统计量 ---
    returns_df = calculate_returns(nav_df)
    n_assets = len(returns_df.columns)
    
    # 年化处理（假设250个交易日）
    mean_returns_daily = returns_df.mean()
    cov_matrix_daily = returns_df.cov()
    mean_returns = mean_returns_daily * 250  # 年化收益
    cov_matrix = cov_matrix_daily * 250       # 年化协方差
    
    st.success(f"✅ 成功获取 {n_assets} 只基金，共 {len(returns_df)} 个交易日数据")
    
    # --- Step 3: 最优风险资产组合 ---
    result = optimize_tangency_portfolio(mean_returns.values, cov_matrix.values, rf_rate)
    
    if not result['success']:
        st.error(f"优化失败: {result['message']}")
        st.stop()
    
    optimal_weights = result['weights']
    p_return = result['expected_return']
    p_risk = result['risk']
    
    # --- Step 4: 计算最优投资比例 y ---
    alloc = calculate_optimal_allocation(p_return, p_risk, rf_rate, risk_aversion)
    y = alloc['y']
    
    # 各基金在完整组合中的最终比例
    final_fund_weights = optimal_weights * y
    
    # ============ 结果展示 ============
    st.markdown("---")
    
    # --- 关键指标卡片 ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("风险组合期望收益 E(rp)", f"{p_return:.2%}")
    with col2:
        st.metric("风险组合标准差 σp", f"{p_risk:.2%}")
    with col3:
        st.metric("夏普比率", f"{result['sharpe_ratio']:.3f}")
    with col4:
        st.metric("最优风险资产投资比例 y*", f"{y:.2%}")
    
    st.markdown("---")
    
    # --- 两列布局 ---
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("📋 各基金权重分配")
        
        # 最优风险组合中的权重
        weight_df = pd.DataFrame({
            '基金代码': list(returns_df.columns),
            '最优风险组合权重 (wd/we)': optimal_weights,
            f'完整组合最终比例 (×y={y:.2f})': final_fund_weights
        })
        weight_df['最优风险组合权重 (wd/we)'] = weight_df['最优风险组合权重 (wd/we)'].apply(
            lambda x: f"{x:.2%}"
        )
        weight_df[f'完整组合最终比例 (×y={y:.2f})'] = weight_df[f'完整组合最终比例 (×y={y:.2f})'].apply(
            lambda x: f"{x:.2%}"
        )
        st.dataframe(weight_df, use_container_width=True, hide_index=True)
        
        # 完整组合构成
        st.subheader("🏦 完整资产组合构成")
        complete_df = pd.DataFrame({
            '资产': ['风险资产组合 P', '无风险资产'],
            '投资比例': [f"{y:.2%}", f"{1-y:.2%}"]
        })
        st.dataframe(complete_df, use_container_width=True, hide_index=True)
        
        if y > 1:
            st.info(f"⚠️ y > 1，表示借入无风险资产加杠杆投资。实际投资中需考虑融资成本。")
        
        # 完整组合指标
        st.subheader("📈 完整组合指标")
        st.write(f"- 完整组合期望收益: **{alloc['complete_return']:.2%}**")
        st.write(f"- 完整组合风险(标准差): **{alloc['complete_risk']:.2%}**")
    
    with right_col:
        st.subheader("📊 有效前沿与资本配置线")
        
        # 生成有效前沿
        ef_df = generate_efficient_frontier(mean_returns.values, cov_matrix.values)
        
        fig = go.Figure()
        
        # 有效前沿
        fig.add_trace(go.Scatter(
            x=ef_df['risk'], y=ef_df['return'],
            mode='lines', name='有效前沿',
            line=dict(color='blue', width=2)
        ))
        
        # 最优风险组合点
        fig.add_trace(go.Scatter(
            x=[p_risk], y=[p_return],
            mode='markers', name=f'最优风险组合 P',
            marker=dict(color='red', size=14, symbol='star')
        ))
        
        # 资本配置线（CAL）
        x_range = np.linspace(0, p_risk * 2.5, 50)
        cal_y = rf_rate + result['sharpe_ratio'] * x_range
        fig.add_trace(go.Scatter(
            x=x_range, y=cal_y,
            mode='lines', name='资本配置线 (CAL)',
            line=dict(color='green', dash='dash')
        ))
        
        # 无风险资产点
        fig.add_trace(go.Scatter(
            x=[0], y=[rf_rate],
            mode='markers+text', name='无风险资产',
            marker=dict(color='orange', size=12),
            text=[f'rf={rf_rate:.1%}'], textposition='top right'
        ))
        
        # 完整组合点
        fig.add_trace(go.Scatter(
            x=[alloc['complete_risk']], y=[alloc['complete_return']],
            mode='markers', name='完整组合 C',
            marker=dict(color='purple', size=12, symbol='diamond')
        ))
        
        fig.update_layout(
            xaxis_title='风险（标准差）',
            yaxis_title='期望收益',
            xaxis_tickformat='.1%',
            yaxis_tickformat='.1%',
            height=500,
            legend=dict(yanchor="top",  x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # --- 数据表格 ---
    st.markdown("---")
    st.subheader("📉 基金净值走势")
    st.line_chart(nav_df)

else:
    st.info("👈 在左侧输入基金代码和参数，点击「开始计算」启动分析")
    st.markdown("""
    ### 使用说明
    1. **基金代码**：使用雪球格式，`F` + 基金代码，例如 `F008975`
    2. **无风险利率**：参考当前国债或货币基金收益率
    3. **风险厌恶系数 A**：A越大越保守，建议范围2~5
    4. 如有需要，在Token栏填入雪球Token以获取更完整的净值数据
    """)


# ============ 同花顺数据源备选方案 ============
def fetch_from_ths(fund_codes, api_key):
    """
    备选方案：从同花顺iFind API获取基金净值
    需要先在同花顺开放平台申请API Key
    
    使用方法：
    1. 访问 https://fuyao.aicubes.cn/admin/ 申请API Key
    2. 设置环境变量 HITHINK_FINANCE_API_KEY
    3. 调用此函数替代 fetch_multiple_funds
    """
    import requests
    
    headers = {"Authorization": f"Bearer {api_key}"}
    all_data = {}
    
    for code in fund_codes:
        ths_code = code.replace("F", "") + ".OF"  # 转为同花顺格式
        url = f"https://fuyao.aicubes.cn/api/fund/nav?thscode={ths_code}"
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            data = resp.json()
            if data.get('status') == 'ok':
                df = pd.DataFrame(data['data'])
                df['date'] = pd.to_datetime(df['date'])
                all_data[code] = df.set_index('date')['nav']
        except Exception as e:
            st.warning(f"同花顺接口获取 {code} 失败: {e}")
    
    if all_data:
        return pd.DataFrame(all_data).dropna()
    return None
