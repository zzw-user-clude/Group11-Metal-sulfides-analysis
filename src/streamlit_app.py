"""
金属硫化物材料交互式数据探索Web应用
使用Streamlit构建
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="金属硫化物材料分析",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    """加载数据"""
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    
    binary_df = pd.read_csv(data_dir / 'binary_metal_sulfides_20260115_200723.csv')
    ternary_df = pd.read_csv(data_dir / 'ternary_metal_sulfides_20260115_201330.csv')
    
    # 清理列名
    binary_df.columns = binary_df.columns.str.replace('\ufeff', '')
    ternary_df.columns = ternary_df.columns.str.replace('\ufeff', '')
    
    # 添加类型标签
    binary_df['material_type'] = 'Binary'
    ternary_df['material_type'] = 'Ternary'
    
    # 合并数据
    combined_df = pd.concat([binary_df, ternary_df], ignore_index=True)
    
    return binary_df, ternary_df, combined_df


def main():
    """主函数"""
    
    # 标题
    st.title("🔬 金属硫化物材料数据分析平台")
    st.markdown("---")
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        binary_df, ternary_df, combined_df = load_data()
    
    # 侧边栏
    st.sidebar.title("📊 导航")
    page = st.sidebar.radio(
        "选择页面",
        ["数据概览", "稳定性分析", "电子性质", "晶体结构", "磁性分析", "数据探索"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **数据集信息**
        - 二元硫化物: 726种
        - 三元硫化物: 995种
        - 总计: 1,721种材料
        
        **数据来源**  
        Materials Project Database
        """
    )
    
    # 根据选择显示不同页面
    if page == "数据概览":
        show_overview(binary_df, ternary_df, combined_df)
    elif page == "稳定性分析":
        show_stability_analysis(binary_df, ternary_df, combined_df)
    elif page == "电子性质":
        show_electronic_properties(binary_df, ternary_df, combined_df)
    elif page == "晶体结构":
        show_crystal_structure(combined_df)
    elif page == "磁性分析":
        show_magnetic_analysis(combined_df)
    elif page == "数据探索":
        show_data_explorer(combined_df)


def show_overview(binary_df, ternary_df, combined_df):
    """数据概览页面"""
    st.header("📈 数据概览")
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总材料数", f"{len(combined_df)}")
    with col2:
        stable_count = combined_df['is_stable'].sum()
        st.metric("稳定材料", f"{stable_count}", f"{stable_count/len(combined_df)*100:.1f}%")
    with col3:
        magnetic_count = combined_df['is_magnetic'].sum()
        st.metric("磁性材料", f"{magnetic_count}", f"{magnetic_count/len(combined_df)*100:.1f}%")
    with col4:
        avg_bandgap = combined_df['band_gap'].mean()
        st.metric("平均带隙", f"{avg_bandgap:.2f} eV")
    
    st.markdown("---")
    
    # 材料类型分布
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("材料类型分布")
        type_counts = combined_df['material_type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("基础统计")
        stats_data = {
            '指标': ['稳定材料数', '磁性材料数', '平均带隙 (eV)', '平均密度 (g/cm³)'],
            '二元硫化物': [
                f"{binary_df['is_stable'].sum()} ({binary_df['is_stable'].sum()/len(binary_df)*100:.1f}%)",
                f"{binary_df['is_magnetic'].sum()} ({binary_df['is_magnetic'].sum()/len(binary_df)*100:.1f}%)",
                f"{binary_df['band_gap'].mean():.3f}",
                f"{binary_df['density'].mean():.3f}"
            ],
            '三元硫化物': [
                f"{ternary_df['is_stable'].sum()} ({ternary_df['is_stable'].sum()/len(ternary_df)*100:.1f}%)",
                f"{ternary_df['is_magnetic'].sum()} ({ternary_df['is_magnetic'].sum()/len(ternary_df)*100:.1f}%)",
                f"{ternary_df['band_gap'].mean():.3f}",
                f"{ternary_df['density'].mean():.3f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)


def show_stability_analysis(binary_df, ternary_df, combined_df):
    """稳定性分析页面"""
    st.header("⚖️ 材料稳定性分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("稳定vs非稳定材料")
        stability_data = combined_df.groupby(['material_type', 'is_stable']).size().unstack()
        fig = px.bar(stability_data, barmode='group', 
                    labels={'value': 'Count', 'material_type': 'Material Type'},
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("能量高于凸包分布")
        data_filtered = combined_df[combined_df['energy_above_hull'] < 0.5]
        fig = px.box(data_filtered, x='material_type', y='energy_above_hull',
                    color='material_type', color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # 稳定性分类
    st.subheader("稳定性分类统计")
    
    def classify_stability(energy):
        if energy == 0:
            return 'Stable (E=0)'
        elif energy < 0.05:
            return 'Near-stable'
        elif energy < 0.2:
            return 'Metastable'
        else:
            return 'Unstable'
    
    combined_df['stability_class'] = combined_df['energy_above_hull'].apply(classify_stability)
    stability_counts = combined_df.groupby(['material_type', 'stability_class']).size().unstack(fill_value=0)
    
    fig = px.bar(stability_counts, barmode='stack',
                labels={'value': 'Count', 'material_type': 'Material Type'})
    st.plotly_chart(fig, use_container_width=True)


def show_electronic_properties(binary_df, ternary_df, combined_df):
    """电子性质页面"""
    st.header("⚡ 电子性质分析")
    
    # 带隙分布
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("带隙分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=binary_df['band_gap'], name='Binary', 
                                  opacity=0.7, marker_color='#FF6B6B', nbinsx=50))
        fig.add_trace(go.Histogram(x=ternary_df['band_gap'], name='Ternary', 
                                  opacity=0.7, marker_color='#4ECDC4', nbinsx=50))
        fig.update_layout(barmode='overlay', xaxis_title='Band Gap (eV)', 
                         yaxis_title='Count', xaxis_range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("带隙分类")
        def classify_bandgap(bg):
            if bg == 0:
                return 'Metal'
            elif bg < 1.0:
                return 'Narrow-gap'
            elif bg < 3.0:
                return 'Semiconductor'
            else:
                return 'Wide-gap'
        
        combined_df['bandgap_class'] = combined_df['band_gap'].apply(classify_bandgap)
        bandgap_counts = combined_df.groupby(['material_type', 'bandgap_class']).size().unstack(fill_value=0)
        
        fig = px.bar(bandgap_counts, barmode='stack',
                    labels={'value': 'Count', 'material_type': 'Material Type'})
        st.plotly_chart(fig, use_container_width=True)
    
    # 带隙vs形成能散点图
    st.subheader("带隙 vs 形成能")
    fig = px.scatter(combined_df, x='band_gap', y='formation_energy_per_atom',
                    color='material_type', hover_data=['formula_pretty'],
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
    fig.update_layout(xaxis_range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)


def show_crystal_structure(combined_df):
    """晶体结构页面"""
    st.header("🔷 晶体结构分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("晶系分布")
        crystal_counts = combined_df['crystal_system'].value_counts()
        fig = px.bar(x=crystal_counts.index, y=crystal_counts.values,
                    labels={'x': 'Crystal System', 'y': 'Count'},
                    color=crystal_counts.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 空间群")
        top_space_groups = combined_df['space_group_symbol'].value_counts().head(10)
        fig = px.bar(x=top_space_groups.values, y=top_space_groups.index,
                    orientation='h', labels={'x': 'Count', 'y': 'Space Group'},
                    color=top_space_groups.values, color_continuous_scale='Teal')
        st.plotly_chart(fig, use_container_width=True)
    
    # 晶系与带隙关系
    st.subheader("不同晶系的带隙分布")
    fig = px.box(combined_df, x='crystal_system', y='band_gap',
                color='crystal_system')
    fig.update_layout(showlegend=False, yaxis_range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)


def show_magnetic_analysis(combined_df):
    """磁性分析页面"""
    st.header("🧲 磁性材料分析")
    
    magnetic_df = combined_df[combined_df['is_magnetic'] == True]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("磁性材料总数", len(magnetic_df))
    with col2:
        st.metric("磁性材料比例", f"{len(magnetic_df)/len(combined_df)*100:.2f}%")
    with col3:
        avg_moment = magnetic_df['total_magnetization'].mean()
        st.metric("平均总磁矩", f"{avg_moment:.2f} μB")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("磁序类型分布")
        ordering_counts = magnetic_df['ordering'].value_counts()
        fig = px.pie(values=ordering_counts.values, names=ordering_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("磁性 vs 非磁性材料带隙")
        magnetic_data = combined_df[combined_df['is_magnetic'] == True]['band_gap']
        non_magnetic_data = combined_df[combined_df['is_magnetic'] == False]['band_gap']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=magnetic_data, name='Magnetic', 
                                  opacity=0.7, marker_color='#FD79A8', nbinsx=40))
        fig.add_trace(go.Histogram(x=non_magnetic_data, name='Non-magnetic', 
                                  opacity=0.7, marker_color='#74B9FF', nbinsx=40))
        fig.update_layout(barmode='overlay', xaxis_title='Band Gap (eV)', 
                         yaxis_title='Count', xaxis_range=[0, 5])
        st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(combined_df):
    """数据探索页面"""
    st.header("🔍 交互式数据探索")
    
    # 筛选器
    st.sidebar.subheader("数据筛选")
    
    material_type = st.sidebar.multiselect(
        "材料类型",
        options=combined_df['material_type'].unique(),
        default=combined_df['material_type'].unique()
    )
    
    stability = st.sidebar.radio(
        "稳定性",
        options=['全部', '仅稳定', '仅非稳定']
    )
    
    bandgap_range = st.sidebar.slider(
        "带隙范围 (eV)",
        min_value=0.0,
        max_value=float(combined_df['band_gap'].max()),
        value=(0.0, 5.0),
        step=0.1
    )
    
    # 应用筛选
    filtered_df = combined_df[combined_df['material_type'].isin(material_type)]
    
    if stability == '仅稳定':
        filtered_df = filtered_df[filtered_df['is_stable'] == True]
    elif stability == '仅非稳定':
        filtered_df = filtered_df[filtered_df['is_stable'] == False]
    
    filtered_df = filtered_df[
        (filtered_df['band_gap'] >= bandgap_range[0]) & 
        (filtered_df['band_gap'] <= bandgap_range[1])
    ]
    
    st.info(f"筛选后材料数量: {len(filtered_df)}")
    
    # 显示数据表
    st.subheader("数据表")
    display_columns = ['material_id', 'formula_pretty', 'formation_energy_per_atom', 
                      'energy_above_hull', 'is_stable', 'band_gap', 'density', 
                      'crystal_system', 'material_type']
    st.dataframe(filtered_df[display_columns].head(100), use_container_width=True)
    
    # 自定义散点图
    st.subheader("自定义散点图")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_axis = st.selectbox("X轴", ['band_gap', 'formation_energy_per_atom', 'density', 
                                     'volume', 'energy_above_hull'])
    with col2:
        y_axis = st.selectbox("Y轴", ['formation_energy_per_atom', 'band_gap', 'density', 
                                     'volume', 'energy_above_hull'])
    with col3:
        color_by = st.selectbox("颜色分组", ['material_type', 'is_stable', 'is_magnetic', 
                                         'crystal_system'])
    
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                    hover_data=['formula_pretty'])
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
