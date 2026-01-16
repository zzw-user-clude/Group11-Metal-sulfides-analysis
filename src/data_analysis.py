"""
金属硫化物材料数据分析脚本
分析二元和三元金属硫化物的稳定性和电子性质
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class MetalSulfideAnalyzer:
    """金属硫化物数据分析器"""
    
    def __init__(self, binary_path, ternary_path):
        """初始化分析器"""
        print("正在加载数据...")
        self.binary_df = pd.read_csv(binary_path)
        self.ternary_df = pd.read_csv(ternary_path)
        
        # 清理列名（移除BOM）
        self.binary_df.columns = self.binary_df.columns.str.replace('\ufeff', '')
        self.ternary_df.columns = self.ternary_df.columns.str.replace('\ufeff', '')
        
        # 添加类型标签
        self.binary_df['material_type'] = 'Binary'
        self.ternary_df['material_type'] = 'Ternary'
        
        # 合并数据集
        self.combined_df = pd.concat([self.binary_df, self.ternary_df], ignore_index=True)
        
        print(f"✓ 二元硫化物: {len(self.binary_df)} 条记录")
        print(f"✓ 三元硫化物: {len(self.ternary_df)} 条记录")
        print(f"✓ 总计: {len(self.combined_df)} 条记录\n")
    
    def basic_statistics(self):
        """基础统计分析"""
        print("="*60)
        print("1. 基础统计信息")
        print("="*60)
        
        stats = []
        
        # 稳定性统计
        binary_stable = self.binary_df['is_stable'].sum()
        ternary_stable = self.ternary_df['is_stable'].sum()
        
        stats.append({
            '指标': '稳定材料数量',
            '二元硫化物': f"{binary_stable} ({binary_stable/len(self.binary_df)*100:.1f}%)",
            '三元硫化物': f"{ternary_stable} ({ternary_stable/len(self.ternary_df)*100:.1f}%)"
        })
        
        # 磁性材料统计
        binary_magnetic = self.binary_df['is_magnetic'].sum()
        ternary_magnetic = self.ternary_df['is_magnetic'].sum()
        
        stats.append({
            '指标': '磁性材料数量',
            '二元硫化物': f"{binary_magnetic} ({binary_magnetic/len(self.binary_df)*100:.1f}%)",
            '三元硫化物': f"{ternary_magnetic} ({ternary_magnetic/len(self.ternary_df)*100:.1f}%)"
        })
        
        # 平均带隙
        binary_bandgap = self.binary_df['band_gap'].mean()
        ternary_bandgap = self.ternary_df['band_gap'].mean()
        
        stats.append({
            '指标': '平均带隙 (eV)',
            '二元硫化物': f"{binary_bandgap:.3f}",
            '三元硫化物': f"{ternary_bandgap:.3f}"
        })
        
        # 平均形成能
        binary_formation = self.binary_df['formation_energy_per_atom'].mean()
        ternary_formation = self.ternary_df['formation_energy_per_atom'].mean()
        
        stats.append({
            '指标': '平均形成能 (eV/atom)',
            '二元硫化物': f"{binary_formation:.3f}",
            '三元硫化物': f"{ternary_formation:.3f}"
        })
        
        # 平均密度
        binary_density = self.binary_df['density'].mean()
        ternary_density = self.ternary_df['density'].mean()
        
        stats.append({
            '指标': '平均密度 (g/cm³)',
            '二元硫化物': f"{binary_density:.3f}",
            '三元硫化物': f"{ternary_density:.3f}"
        })
        
        stats_df = pd.DataFrame(stats)
        print(stats_df.to_string(index=False))
        print()
        
        return stats_df
    
    def analyze_stability(self):
        """稳定性分析"""
        print("="*60)
        print("2. 材料稳定性分析")
        print("="*60)
        
        # 按能量高于凸包值分类
        def classify_stability(energy):
            if energy == 0:
                return 'Stable (E=0)'
            elif energy < 0.05:
                return 'Near-stable (E<0.05)'
            elif energy < 0.2:
                return 'Metastable (0.05≤E<0.2)'
            else:
                return 'Unstable (E≥0.2)'
        
        self.combined_df['stability_class'] = self.combined_df['energy_above_hull'].apply(classify_stability)
        
        stability_stats = self.combined_df.groupby(['material_type', 'stability_class']).size().unstack(fill_value=0)
        print(stability_stats)
        print()
        
        return stability_stats
    
    def analyze_bandgap(self):
        """带隙分析"""
        print("="*60)
        print("3. 电子性质（带隙）分析")
        print("="*60)
        
        # 带隙分类
        def classify_bandgap(bg):
            if bg == 0:
                return 'Metal (Eg=0)'
            elif bg < 1.0:
                return 'Narrow-gap (Eg<1.0)'
            elif bg < 3.0:
                return 'Semiconductor (1.0≤Eg<3.0)'
            else:
                return 'Wide-gap (Eg≥3.0)'
        
        self.combined_df['bandgap_class'] = self.combined_df['band_gap'].apply(classify_bandgap)
        
        bandgap_stats = self.combined_df.groupby(['material_type', 'bandgap_class']).size().unstack(fill_value=0)
        print(bandgap_stats)
        print()
        
        # 统计不同带隙类型的平均形成能
        print("不同带隙类型的平均形成能:")
        formation_by_bandgap = self.combined_df.groupby('bandgap_class')['formation_energy_per_atom'].mean()
        print(formation_by_bandgap.sort_values())
        print()
        
        return bandgap_stats
    
    def analyze_crystal_structure(self):
        """晶体结构分析"""
        print("="*60)
        print("4. 晶体结构分析")
        print("="*60)
        
        # 晶系统计
        crystal_stats = self.combined_df.groupby(['material_type', 'crystal_system']).size().unstack(fill_value=0)
        print("各晶系分布:")
        print(crystal_stats)
        print()
        
        # 最常见的空间群
        print("Top 10 最常见空间群:")
        top_space_groups = self.combined_df['space_group_symbol'].value_counts().head(10)
        print(top_space_groups)
        print()
        
        return crystal_stats
    
    def correlation_analysis(self):
        """相关性分析"""
        print("="*60)
        print("5. 性质相关性分析")
        print("="*60)
        
        # 选择数值型特征
        numeric_features = ['formation_energy_per_atom', 'energy_above_hull', 
                          'band_gap', 'total_magnetization', 'density', 
                          'volume', 'nsites']
        
        corr_matrix = self.combined_df[numeric_features].corr()
        
        print("关键相关性系数 (|r| > 0.3):")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3:
                    print(f"{corr_matrix.columns[i]:30s} <-> {corr_matrix.columns[j]:30s}: {corr_val:6.3f}")
        print()
        
        return corr_matrix
    
    def magnetic_analysis(self):
        """磁性分析"""
        print("="*60)
        print("6. 磁性材料分析")
        print("="*60)
        
        magnetic_df = self.combined_df[self.combined_df['is_magnetic'] == True]
        
        print(f"磁性材料总数: {len(magnetic_df)}")
        print(f"磁性材料比例: {len(magnetic_df)/len(self.combined_df)*100:.2f}%\n")
        
        # 磁序统计
        print("磁序类型分布:")
        magnetic_ordering = magnetic_df['ordering'].value_counts()
        print(magnetic_ordering)
        print()
        
        # 磁性材料的平均性质
        print("磁性 vs 非磁性材料的平均性质对比:")
        comparison = pd.DataFrame({
            '磁性材料': magnetic_df[['band_gap', 'formation_energy_per_atom', 'density']].mean(),
            '非磁性材料': self.combined_df[self.combined_df['is_magnetic'] == False][['band_gap', 'formation_energy_per_atom', 'density']].mean()
        })
        print(comparison)
        print()
        
        return magnetic_df
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("分析总结报告")
        print("="*60 + "\n")
        
        report = []
        
        # 1. 数据集概况
        report.append("【数据集概况】")
        report.append(f"• 二元硫化物: {len(self.binary_df)} 种材料")
        report.append(f"• 三元硫化物: {len(self.ternary_df)} 种材料")
        report.append(f"• 总计: {len(self.combined_df)} 种材料\n")
        
        # 2. 稳定性发现
        stable_binary = self.binary_df['is_stable'].sum()
        stable_ternary = self.ternary_df['is_stable'].sum()
        report.append("【稳定性发现】")
        report.append(f"• 二元硫化物稳定率: {stable_binary/len(self.binary_df)*100:.1f}%")
        report.append(f"• 三元硫化物稳定率: {stable_ternary/len(self.ternary_df)*100:.1f}%")
        if stable_binary/len(self.binary_df) > stable_ternary/len(self.ternary_df):
            report.append("• 发现：二元硫化物稳定性更高\n")
        else:
            report.append("• 发现：三元硫化物稳定性更高\n")
        
        # 3. 电子性质
        metal_count = (self.combined_df['band_gap'] == 0).sum()
        semiconductor_count = ((self.combined_df['band_gap'] > 0) & (self.combined_df['band_gap'] < 3.0)).sum()
        report.append("【电子性质】")
        report.append(f"• 金属性材料: {metal_count} ({metal_count/len(self.combined_df)*100:.1f}%)")
        report.append(f"• 半导体材料: {semiconductor_count} ({semiconductor_count/len(self.combined_df)*100:.1f}%)")
        report.append(f"• 平均带隙: {self.combined_df['band_gap'].mean():.3f} eV\n")
        
        # 4. 磁性
        magnetic_count = self.combined_df['is_magnetic'].sum()
        report.append("【磁性特征】")
        report.append(f"• 磁性材料: {magnetic_count} ({magnetic_count/len(self.combined_df)*100:.1f}%)")
        report.append(f"• 非磁性材料: {len(self.combined_df)-magnetic_count} ({(len(self.combined_df)-magnetic_count)/len(self.combined_df)*100:.1f}%)\n")
        
        # 5. 晶体结构
        top_crystal = self.combined_df['crystal_system'].value_counts().head(3)
        report.append("【晶体结构】")
        report.append(f"• 最常见晶系:")
        for crystal, count in top_crystal.items():
            report.append(f"  - {crystal}: {count} ({count/len(self.combined_df)*100:.1f}%)")
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text


def main():
    """主函数"""
    # 设置路径
    data_dir = Path(__file__).parent.parent / 'data'
    binary_path = data_dir / 'binary_metal_sulfides_20260115_200723.csv'
    ternary_path = data_dir / 'ternary_metal_sulfides_20260115_201330.csv'
    
    # 创建分析器
    analyzer = MetalSulfideAnalyzer(binary_path, ternary_path)
    
    # 执行分析
    analyzer.basic_statistics()
    analyzer.analyze_stability()
    analyzer.analyze_bandgap()
    analyzer.analyze_crystal_structure()
    analyzer.correlation_analysis()
    analyzer.magnetic_analysis()
    
    # 生成总结报告
    report = analyzer.generate_summary_report()
    
    # 保存报告
    report_path = data_dir.parent / 'docs' / 'analysis_report.txt'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ 报告已保存至: {report_path}")


if __name__ == '__main__':
    main()
