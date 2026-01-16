"""
金属硫化物材料数据可视化脚本
生成各类统计图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")
sns.set_palette("Set2")


class MetalSulfideVisualizer:
    """金属硫化物数据可视化器"""
    
    def __init__(self, binary_path, ternary_path, output_dir):
        """初始化"""
        self.binary_df = pd.read_csv(binary_path)
        self.ternary_df = pd.read_csv(ternary_path)
        
        # 清理列名
        self.binary_df.columns = self.binary_df.columns.str.replace('\ufeff', '')
        self.ternary_df.columns = self.ternary_df.columns.str.replace('\ufeff', '')
        
        # 添加类型标签
        self.binary_df['material_type'] = 'Binary'
        self.ternary_df['material_type'] = 'Ternary'
        
        # 合并数据
        self.combined_df = pd.concat([self.binary_df, self.ternary_df], ignore_index=True)
        
        # 输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"可视化输出目录: {self.output_dir}")
    
    def plot_stability_comparison(self):
        """绘制稳定性对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 稳定vs非稳定材料数量
        stability_counts = self.combined_df.groupby(['material_type', 'is_stable']).size().unstack()
        stability_counts.plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Stability Comparison: Binary vs Ternary Sulfides', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Material Type', fontsize=12)
        axes[0].set_ylabel('Number of Materials', fontsize=12)
        axes[0].legend(['Unstable', 'Stable'], title='Stability')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 子图2: 能量高于凸包分布
        data_to_plot = [
            self.binary_df[self.binary_df['energy_above_hull'] < 0.5]['energy_above_hull'],
            self.ternary_df[self.ternary_df['energy_above_hull'] < 0.5]['energy_above_hull']
        ]
        bp = axes[1].boxplot(data_to_plot, labels=['Binary', 'Ternary'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
            patch.set_facecolor(color)
        axes[1].set_title('Energy Above Hull Distribution', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Energy Above Hull (eV/atom)', fontsize=12)
        axes[1].set_xlabel('Material Type', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_stability_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已生成: fig1_stability_comparison.png")
    
    def plot_bandgap_analysis(self):
        """绘制带隙分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 带隙分布直方图
        axes[0, 0].hist([self.binary_df['band_gap'], self.ternary_df['band_gap']], 
                       bins=50, label=['Binary', 'Ternary'], alpha=0.7, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_xlabel('Band Gap (eV)', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].set_title('Band Gap Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(-0.5, 5)
        axes[0, 0].grid(alpha=0.3)
        
        # 子图2: 带隙类别统计
        def classify_bandgap(bg):
            if bg == 0:
                return 'Metal'
            elif bg < 1.0:
                return 'Narrow-gap'
            elif bg < 3.0:
                return 'Semiconductor'
            else:
                return 'Wide-gap'
        
        self.combined_df['bandgap_class'] = self.combined_df['band_gap'].apply(classify_bandgap)
        bandgap_counts = self.combined_df.groupby(['material_type', 'bandgap_class']).size().unstack(fill_value=0)
        bandgap_counts.plot(kind='bar', ax=axes[0, 1], stacked=True)
        axes[0, 1].set_title('Band Gap Classification', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Material Type', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].legend(title='Band Gap Type', bbox_to_anchor=(1.05, 1))
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
        
        # 子图3: 带隙 vs 形成能散点图
        for mtype, color in zip(['Binary', 'Ternary'], ['#FF6B6B', '#4ECDC4']):
            data = self.combined_df[self.combined_df['material_type'] == mtype]
            axes[1, 0].scatter(data['band_gap'], data['formation_energy_per_atom'], 
                             alpha=0.5, s=30, label=mtype, color=color)
        axes[1, 0].set_xlabel('Band Gap (eV)', fontsize=11)
        axes[1, 0].set_ylabel('Formation Energy (eV/atom)', fontsize=11)
        axes[1, 0].set_title('Band Gap vs Formation Energy', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xlim(-0.5, 5)
        
        # 子图4: 稳定材料的带隙分布
        stable_data = self.combined_df[self.combined_df['is_stable'] == True]
        for mtype, color in zip(['Binary', 'Ternary'], ['#FF6B6B', '#4ECDC4']):
            data = stable_data[stable_data['material_type'] == mtype]['band_gap']
            axes[1, 1].hist(data, bins=30, alpha=0.7, label=f'{mtype} (Stable)', color=color)
        axes[1, 1].set_xlabel('Band Gap (eV)', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('Band Gap Distribution (Stable Materials Only)', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_xlim(-0.5, 5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_bandgap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已生成: fig2_bandgap_analysis.png")
    
    def plot_crystal_structure(self):
        """绘制晶体结构分析图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 子图1: 晶系分布
        crystal_counts = self.combined_df.groupby(['material_type', 'crystal_system']).size().unstack(fill_value=0)
        crystal_counts.T.plot(kind='barh', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Crystal System Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Count', fontsize=12)
        axes[0].set_ylabel('Crystal System', fontsize=12)
        axes[0].legend(title='Material Type')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 子图2: Top 10 空间群
        top_space_groups = self.combined_df['space_group_symbol'].value_counts().head(10)
        axes[1].barh(range(len(top_space_groups)), top_space_groups.values, color='#95E1D3')
        axes[1].set_yticks(range(len(top_space_groups)))
        axes[1].set_yticklabels(top_space_groups.index)
        axes[1].set_xlabel('Count', fontsize=12)
        axes[1].set_title('Top 10 Space Groups', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_crystal_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已生成: fig3_crystal_structure.png")
    
    def plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        # 选择数值特征
        numeric_features = ['formation_energy_per_atom', 'energy_above_hull', 
                          'band_gap', 'total_magnetization', 'density', 
                          'volume', 'nsites']
        
        # 计算相关性矩阵
        corr_matrix = self.combined_df[numeric_features].corr()
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Property Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已生成: fig4_correlation_heatmap.png")
    
    def plot_magnetic_analysis(self):
        """绘制磁性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 磁性vs非磁性材料比例
        magnetic_counts = self.combined_df.groupby(['material_type', 'is_magnetic']).size().unstack()
        magnetic_counts.plot(kind='bar', ax=axes[0, 0], color=['#FFEAA7', '#DFE6E9'])
        axes[0, 0].set_title('Magnetic vs Non-magnetic Materials', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Material Type', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].legend(['Non-magnetic', 'Magnetic'])
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 子图2: 磁序类型分布
        magnetic_df = self.combined_df[self.combined_df['is_magnetic'] == True]
        ordering_counts = magnetic_df['ordering'].value_counts()
        axes[0, 1].pie(ordering_counts.values, labels=ordering_counts.index, autopct='%1.1f%%',
                      colors=sns.color_palette("Set2"))
        axes[0, 1].set_title('Magnetic Ordering Types', fontsize=12, fontweight='bold')
        
        # 子图3: 磁性材料的带隙分布
        magnetic_data = self.combined_df[self.combined_df['is_magnetic'] == True]['band_gap']
        non_magnetic_data = self.combined_df[self.combined_df['is_magnetic'] == False]['band_gap']
        axes[1, 0].hist([magnetic_data, non_magnetic_data], bins=40, 
                       label=['Magnetic', 'Non-magnetic'], alpha=0.7,
                       color=['#FD79A8', '#74B9FF'])
        axes[1, 0].set_xlabel('Band Gap (eV)', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].set_title('Band Gap Distribution by Magnetism', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xlim(-0.5, 5)
        
        # 子图4: 总磁矩分布
        magnetic_moment = self.combined_df[self.combined_df['total_magnetization'] > 0]['total_magnetization']
        axes[1, 1].hist(magnetic_moment, bins=50, color='#A29BFE', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Total Magnetization (μB)', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title('Total Magnetization Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_xlim(0, 10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_magnetic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已生成: fig5_magnetic_analysis.png")
    
    def plot_property_comparison(self):
        """绘制性质对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        properties = [
            ('density', 'Density (g/cm³)', axes[0, 0]),
            ('volume', 'Volume (Ų)', axes[0, 1]),
            ('nsites', 'Number of Sites', axes[1, 0]),
            ('formation_energy_per_atom', 'Formation Energy (eV/atom)', axes[1, 1])
        ]
        
        for prop, label, ax in properties:
            data_to_plot = [
                self.binary_df[prop].dropna(),
                self.ternary_df[prop].dropna()
            ]
            bp = ax.boxplot(data_to_plot, labels=['Binary', 'Ternary'], patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
                patch.set_facecolor(color)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig6_property_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ 已生成: fig6_property_comparison.png")
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("\n开始生成可视化图表...")
        print("="*60)
        
        self.plot_stability_comparison()
        self.plot_bandgap_analysis()
        self.plot_crystal_structure()
        self.plot_correlation_heatmap()
        self.plot_magnetic_analysis()
        self.plot_property_comparison()
        
        print("="*60)
        print(f"✓ 所有图表已生成完毕！保存位置: {self.output_dir}\n")


def main():
    """主函数"""
    # 设置路径
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    output_dir = project_dir / 'figures'
    
    binary_path = data_dir / 'binary_metal_sulfides_20260115_200723.csv'
    ternary_path = data_dir / 'ternary_metal_sulfides_20260115_201330.csv'
    
    # 创建可视化器并生成所有图表
    visualizer = MetalSulfideVisualizer(binary_path, ternary_path, output_dir)
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()
