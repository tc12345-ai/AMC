"""
AMC Visualization
AMC可视化模块 - 图表绘制
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class AMCPlotter:
    """
    AMC可视化绘图器
    
    生成以下图表:
    1. 吞吐量 vs SNR
    2. 频谱效率 vs SNR
    3. BLER vs SNR
    4. MCS选择 vs SNR
    5. 策略对比
    """
    
    # 策略颜色和样式
    STRATEGY_STYLES = {
        'conservative': {'color': '#2ecc71', 'linestyle': '-', 'label': '保守策略'},
        'aggressive': {'color': '#e74c3c', 'linestyle': '-', 'label': '激进策略'},
        'target_bler': {'color': '#3498db', 'linestyle': '-', 'label': '目标BLER策略'},
    }
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        初始化绘图器
        
        Args:
            figsize: 图形尺寸
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = 'seaborn-v0_8-whitegrid'
    
    def _setup_style(self):
        """设置绘图样式"""
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
    
    def plot_throughput_comparison(self, 
                                    simulation_result,
                                    include_shannon: bool = True,
                                    ax: plt.Axes = None) -> Figure:
        """
        绘制吞吐量对比图
        
        Args:
            simulation_result: 仿真结果
            include_shannon: 是否包含Shannon容量
            ax: 可选的Axes对象
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        snr = simulation_result.snr_range
        
        # 绘制Shannon容量
        if include_shannon and simulation_result.shannon_capacity is not None:
            ax.plot(snr, simulation_result.shannon_capacity, 
                   'k--', linewidth=1.5, alpha=0.7, label='Shannon容量')
        
        # 绘制各策略
        for name, result in simulation_result.strategy_results.items():
            style = self.STRATEGY_STYLES.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax.plot(snr, result['throughput_mbps'],
                   color=style['color'], linestyle=style['linestyle'],
                   linewidth=2, label=style['label'])
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('吞吐量 (Mbps)', fontsize=12)
        ax.set_title('吞吐量 vs SNR - 策略对比', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([snr[0], snr[-1]])
        ax.set_ylim(bottom=0)
        
        fig.tight_layout()
        return fig
    
    def plot_spectral_efficiency(self,
                                  simulation_result,
                                  ax: plt.Axes = None) -> Figure:
        """
        绘制频谱效率对比图
        
        Args:
            simulation_result: 仿真结果
            ax: 可选的Axes对象
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        snr = simulation_result.snr_range
        
        # 绘制各策略
        for name, result in simulation_result.strategy_results.items():
            style = self.STRATEGY_STYLES.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax.plot(snr, result['effective_spectral_efficiency'],
                   color=style['color'], linestyle=style['linestyle'],
                   linewidth=2, label=style['label'])
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('频谱效率 (bits/s/Hz)', fontsize=12)
        ax.set_title('有效频谱效率 vs SNR', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([snr[0], snr[-1]])
        ax.set_ylim(bottom=0)
        
        fig.tight_layout()
        return fig
    
    def plot_bler_comparison(self,
                              simulation_result,
                              log_scale: bool = True,
                              ax: plt.Axes = None) -> Figure:
        """
        绘制BLER对比图
        
        Args:
            simulation_result: 仿真结果
            log_scale: 是否使用对数刻度
            ax: 可选的Axes对象
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        snr = simulation_result.snr_range
        
        # 绘制各策略
        for name, result in simulation_result.strategy_results.items():
            style = self.STRATEGY_STYLES.get(name, {'color': 'gray', 'linestyle': '-', 'label': name})
            ax.plot(snr, result['bler'],
                   color=style['color'], linestyle=style['linestyle'],
                   linewidth=2, label=style['label'])
        
        # 绘制目标BLER参考线
        target_bler = simulation_result.config.target_bler
        ax.axhline(y=target_bler, color='gray', linestyle=':', 
                   alpha=0.7, label=f'目标BLER ({target_bler*100:.0f}%)')
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('BLER', fontsize=12)
        ax.set_title('BLER vs SNR', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([snr[0], snr[-1]])
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim([1e-4, 1])
        else:
            ax.set_ylim([0, 1])
        
        fig.tight_layout()
        return fig
    
    def plot_mcs_selection(self,
                            simulation_result,
                            strategy_name: str = 'target_bler',
                            ax: plt.Axes = None) -> Figure:
        """
        绘制MCS选择图
        
        Args:
            simulation_result: 仿真结果
            strategy_name: 策略名称
            ax: 可选的Axes对象
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        snr = simulation_result.snr_range
        result = simulation_result.strategy_results.get(strategy_name)
        
        if result is None:
            return fig
        
        mcs = result['selected_mcs']
        
        # 使用阶梯图
        ax.step(snr, mcs, where='post', linewidth=2, color='#9b59b6')
        ax.fill_between(snr, mcs, step='post', alpha=0.3, color='#9b59b6')
        
        # 标记切换点
        thresholds = simulation_result.thresholds.get(strategy_name, [])
        for t in thresholds:
            ax.axvline(x=t.snr_threshold, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('MCS Index', fontsize=12)
        style = self.STRATEGY_STYLES.get(strategy_name, {'label': strategy_name})
        ax.set_title(f'MCS选择 vs SNR ({style["label"]})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([snr[0], snr[-1]])
        ax.set_ylim(bottom=0)
        
        fig.tight_layout()
        return fig
    
    def plot_bler_curves(self,
                          mcs_table,
                          bler_model,
                          snr_range: np.ndarray,
                          ax: plt.Axes = None) -> Figure:
        """
        绘制各MCS的BLER曲线
        
        Args:
            mcs_table: MCS表
            bler_model: BLER模型
            snr_range: SNR范围
            ax: 可选的Axes对象
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        # 颜色映射
        cmap = plt.cm.viridis
        n_mcs = len(mcs_table)
        
        for i, entry in enumerate(mcs_table):
            color = cmap(i / n_mcs)
            bler = bler_model.calculate_bler(snr_range, entry.index)
            ax.semilogy(snr_range, bler, color=color, linewidth=1.5,
                       label=f'MCS {entry.index} ({entry.modulation})')
        
        # 参考线
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='BLER=10%')
        ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='BLER=1%')
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('BLER', fontsize=12)
        ax.set_title('BLER曲线 - 各MCS', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([snr_range[0], snr_range[-1]])
        ax.set_ylim([1e-4, 1])
        
        fig.tight_layout()
        return fig
    
    def plot_harq_analysis(self,
                            harq_analysis: Dict,
                            ax: plt.Axes = None) -> Figure:
        """
        绘制HARQ分析图
        
        Args:
            harq_analysis: HARQ分析结果
            ax: 可选的Axes对象
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        else:
            fig = ax.get_figure()
            axes = [ax, ax.twinx()]
        
        snr = harq_analysis['snr_db']
        
        # 左图: BLER对比
        axes[0].semilogy(snr, harq_analysis['initial_bler'], 'b-', 
                        linewidth=2, label='初始BLER')
        axes[0].semilogy(snr, harq_analysis['residual_bler'], 'g-', 
                        linewidth=2, label='残余BLER (HARQ后)')
        axes[0].set_xlabel('SNR (dB)', fontsize=12)
        axes[0].set_ylabel('BLER', fontsize=12)
        axes[0].set_title('HARQ BLER改善', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([1e-6, 1])
        
        # 右图: 平均传输次数
        axes[1].plot(snr, harq_analysis['avg_transmissions'], 'r-', linewidth=2)
        axes[1].set_xlabel('SNR (dB)', fontsize=12)
        axes[1].set_ylabel('平均传输次数', fontsize=12)
        axes[1].set_title('HARQ平均传输次数', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([1, 5])
        
        fig.tight_layout()
        return fig
    
    def plot_all(self, simulation_result, save_path: str = None) -> List[Figure]:
        """
        绘制所有图表
        
        Args:
            simulation_result: 仿真结果
            save_path: 保存路径（可选）
            
        Returns:
            Figure列表
        """
        figures = []
        
        # 吞吐量对比
        fig1 = self.plot_throughput_comparison(simulation_result)
        figures.append(('throughput', fig1))
        
        # 频谱效率
        fig2 = self.plot_spectral_efficiency(simulation_result)
        figures.append(('spectral_efficiency', fig2))
        
        # BLER对比
        fig3 = self.plot_bler_comparison(simulation_result)
        figures.append(('bler', fig3))
        
        # MCS选择
        fig4 = self.plot_mcs_selection(simulation_result)
        figures.append(('mcs_selection', fig4))
        
        # HARQ分析
        if simulation_result.harq_analysis:
            fig5 = self.plot_harq_analysis(simulation_result.harq_analysis)
            figures.append(('harq', fig5))
        
        # 保存
        if save_path:
            for name, fig in figures:
                fig.savefig(f"{save_path}/{name}.png", dpi=150, bbox_inches='tight')
        
        return [f for _, f in figures]
    
    def create_dashboard(self, simulation_result) -> Figure:
        """
        创建仪表板（多子图）
        
        Args:
            simulation_result: 仿真结果
            
        Returns:
            Figure对象
        """
        self._setup_style()
        
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        
        # 2x2布局
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        
        # 绘制各子图
        self.plot_throughput_comparison(simulation_result, ax=ax1)
        self.plot_spectral_efficiency(simulation_result, ax=ax2)
        self.plot_bler_comparison(simulation_result, ax=ax3)
        self.plot_mcs_selection(simulation_result, ax=ax4)
        
        fig.suptitle('AMC仿真结果仪表板', fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        
        return fig
