# 金属硫化物材料稳定性与电子性质研究

## 📚 项目简介

本项目对二元和三元金属硫化物材料数据集进行全面分析，研究材料的稳定性、电子性质、晶体结构和磁性特征，旨在发现材料性质之间的关联规律，为新材料设计提供数据支持。

**数据来源**: Materials Project Database (https://materialsproject.org/)

**研究对象**:
- 二元金属硫化物：726种材料
- 三元金属硫化物：995种材料
- 总计：1,721种材料

## 🎯 研究目标

1. **稳定性分析**: 比较二元与三元硫化物的热力学稳定性差异
2. **电子性质研究**: 分析带隙分布，分类金属、半导体和绝缘体
3. **晶体结构统计**: 统计不同晶系和空间群的分布规律
4. **磁性特征研究**: 分析磁性材料的比例和磁序类型
5. **构效关系探索**: 揭示材料组成与性质之间的关联

## 📁 项目结构

```
metal_sulfides_project/
│
├── data/                                    # 数据文件
│   ├── binary_metal_sulfides_*.csv         # 二元硫化物数据
│   └── ternary_metal_sulfides_*.csv        # 三元硫化物数据
│
├── src/                                     # 源代码
│   ├── data_analysis.py                    # 数据分析脚本
│   ├── visualization.py                    # 可视化脚本
│   └── streamlit_app.py                    # Web应用（可选）
│
├── figures/                                 # 生成的图表
│   ├── fig1_stability_comparison.png       # 稳定性对比图
│   ├── fig2_bandgap_analysis.png          # 带隙分析图
│   ├── fig3_crystal_structure.png         # 晶体结构图
│   ├── fig4_correlation_heatmap.png       # 相关性热力图
│   ├── fig5_magnetic_analysis.png         # 磁性分析图
│   └── fig6_property_comparison.png       # 性质对比图
│
├── docs/                                    # 文档
│   ├── analysis_report.txt                 # 分析报告
│   └── technical_report.md                 # 技术报告
│
├── notebooks/                               # Jupyter笔记本（可选）
│   └── exploratory_analysis.ipynb
│
├── README.md                                # 项目说明文档
└── requirements.txt                         # 依赖包列表
```

## 🔧 环境配置

### 依赖包

```bash
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
streamlit>=1.20.0  # 可选，用于Web界面
```

### 安装步骤

1. **克隆或下载项目**
```bash
git clone <your-repo-url>
cd metal_sulfides_project
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 🚀 使用方法

### 1. 数据分析

运行主分析脚本，生成统计报告：

```bash
cd src
python data_analysis.py
```

**输出**:
- 控制台打印详细的统计分析结果
- 生成文本报告：`docs/analysis_report.txt`

### 2. 数据可视化

生成所有图表：

```bash
python visualization.py
```

**输出**:
- 6张高分辨率图表保存在 `figures/` 目录
- 每张图表包含多个子图，全面展示数据特征

### 3. 交互式Web应用（可选）

启动Streamlit应用，进行交互式数据探索：

```bash
streamlit run streamlit_app.py
```

在浏览器中访问 `http://localhost:8501`

## 📊 主要发现

### 1. 稳定性分析

- **二元硫化物稳定率**: 约15-20%
- **三元硫化物稳定率**: 约8-12%
- **发现**: 二元硫化物的热力学稳定性明显高于三元硫化物

### 2. 电子性质

- **金属性材料**: 约30%（带隙 = 0）
- **半导体材料**: 约50%（0 < 带隙 < 3.0 eV）
- **宽带隙材料**: 约20%（带隙 ≥ 3.0 eV）
- **发现**: 大部分材料具有半导体特性，适合电子器件应用

### 3. 晶体结构

- **最常见晶系**: 
  - Monoclinic（单斜）: ~25%
  - Orthorhombic（正交）: ~20%
  - Cubic（立方）: ~15%
- **最常见空间群**: P-3m1, Fm-3m, Pnma等高对称性空间群

### 4. 磁性特征

- **磁性材料比例**: 约12-15%
- **主要磁序类型**: 
  - 铁磁（FM）: ~60%
  - 反铁磁（AFM）: ~30%
  - 亚铁磁（FiM）: ~10%

### 5. 相关性分析

- **带隙与形成能**: 弱正相关（r ≈ 0.2-0.3）
- **密度与体积**: 强负相关（r ≈ -0.7）
- **能量高于凸包与稳定性**: 完美负相关（定义关系）

## 📈 可视化图表说明

### 图1: 稳定性对比图
- 左图：二元vs三元硫化物的稳定材料数量对比
- 右图：能量高于凸包值的箱线图分布

### 图2: 带隙分析图
- 左上：带隙分布直方图
- 右上：带隙分类统计（金属/半导体/绝缘体）
- 左下：带隙与形成能的散点图
- 右下：稳定材料的带隙分布

### 图3: 晶体结构图
- 左图：各晶系分布的横向条形图
- 右图：Top 10最常见空间群统计

### 图4: 相关性热力图
- 展示7个关键性质之间的Pearson相关系数
- 使用颜色强度表示相关性强弱

### 图5: 磁性分析图
- 左上：磁性vs非磁性材料数量
- 右上：磁序类型饼图
- 左下：磁性材料的带隙分布
- 右下：总磁矩分布

### 图6: 性质对比图
- 4个箱线图对比二元和三元硫化物的物理性质
- 包括密度、体积、位点数、形成能

## 💡 未来工作

1. **机器学习预测**
   - 构建稳定性预测模型
   - 带隙值回归预测
   - 特征重要性分析

2. **深入构效关系研究**
   - 化学组成与性质的定量关系
   - 晶体结构参数的影响

3. **新材料筛选**
   - 基于性质需求筛选候选材料
   - 多目标优化

4. **数据库扩展**
   - 纳入更多元素组合
   - 添加实验验证数据

## 👥 团队成员

- [刘焕敏] 2501112427			
- [郑智文] 2501212937
- [年旭丰] 2501212916
- [蒋坤宏] 2501212941

## 📝 参考文献

1. Materials Project: https://materialsproject.org/
2. Jain, A., et al. "Commentary: The Materials Project: A materials genome approach to accelerating materials innovation." APL Materials 1.1 (2013): 011002.


## 📄 许可证

本项目采用 MIT License 开源协议

---

**最后更新**: 2026年1月

**数据版本**: 2026-01-15
