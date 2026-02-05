#!/usr/bin/env python3
"""
AMC Strategy Simulation and Throughput Evaluation Software
AMC策略仿真与吞吐评估软件

自适应调制编码（AMC）策略仿真软件，支持5G/4G链路自适应。

功能特性:
- 支持4G LTE / 5G NR MCS表
- BLER性能曲线建模（Sigmoid逼近）
- 门限自动搜索（二分法）
- 多策略对比：保守/激进/目标BLER
- 吞吐量与频谱效率评估
- HARQ简单模型（Chase Combining）
- 完整的GUI界面
- 结果导出（CSV/图表）

Usage:
    python main.py           # 启动GUI
    python main.py --cli     # 命令行模式

Author: AMC Simulator
Version: 1.0.0
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_gui():
    """启动图形界面"""
    from gui.main_window import run_app
    run_app()


def run_cli(args):
    """命令行模式"""
    from simulation.simulator import AMCSimulator, SimulationConfig
    from visualization.plots import AMCPlotter
    
    print("=" * 60)
    print("AMC策略仿真与吞吐评估软件")
    print("Adaptive Modulation and Coding Simulation")
    print("=" * 60)
    
    # 配置
    config = SimulationConfig(
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_step=args.snr_step,
        target_bler=args.target_bler,
        bandwidth_mhz=args.bandwidth,
        mcs_table_name=args.mcs_table,
        enable_harq=args.harq,
        harq_max_retx=args.harq_retx,
        harq_gain_db=args.harq_gain
    )
    
    print(f"\n配置:")
    print(f"  SNR范围: {config.snr_min} ~ {config.snr_max} dB (步长: {config.snr_step} dB)")
    print(f"  MCS表: {config.mcs_table_name}")
    print(f"  目标BLER: {config.target_bler*100:.1f}%")
    print(f"  带宽: {config.bandwidth_mhz} MHz")
    print(f"  HARQ: {'启用' if config.enable_harq else '禁用'}")
    
    # 运行仿真
    print("\n正在运行仿真...")
    simulator = AMCSimulator(config)
    result = simulator.run()
    
    print("仿真完成！\n")
    
    # 显示策略对比
    print("策略性能对比:")
    print("-" * 70)
    comparison = simulator.compare_strategies()
    print(f"{'策略':<20} {'平均吞吐量':>12} {'峰值吞吐量':>12} {'平均BLER':>10} {'平均SE':>10}")
    print("-" * 70)
    
    strategy_names = {
        'conservative': '保守策略',
        'aggressive': '激进策略',
        'target_bler': '目标BLER策略'
    }
    
    for name, data in comparison.items():
        print(f"{strategy_names.get(name, name):<20} "
              f"{data['avg_throughput_mbps']:>10.2f} Mbps "
              f"{data['peak_throughput_mbps']:>10.2f} Mbps "
              f"{data['avg_bler']:>10.2%} "
              f"{data['avg_spectral_efficiency']:>8.3f}")
    
    print("-" * 70)
    
    # 显示门限表
    print("\nMCS切换门限表 (目标BLER策略):")
    print("-" * 55)
    print(f"{'MCS':>4} {'调制':>8} {'频谱效率':>12} {'SNR门限':>12} {'BLER':>10}")
    print("-" * 55)
    
    for t in simulator.get_threshold_table('target_bler'):
        print(f"{t['MCS']:>4} {t['Modulation']:>8} {t['SE (bits/s/Hz)']:>12} "
              f"{t['SNR Threshold (dB)']:>12} {t['BLER']:>10}")
    
    print("-" * 55)
    
    # 导出结果
    if args.output:
        simulator.export_results(args.output)
        print(f"\n结果已导出到: {args.output}")
    
    # 绘图
    if args.plot:
        print("\n生成图表...")
        plotter = AMCPlotter()
        
        import matplotlib.pyplot as plt
        
        # 创建仪表板
        fig = plotter.create_dashboard(result)
        
        if args.save_plots:
            fig.savefig(args.save_plots, dpi=150, bbox_inches='tight')
            print(f"图表已保存到: {args.save_plots}")
        else:
            plt.show()


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description='AMC策略仿真与吞吐评估软件',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--cli', action='store_true',
                        help='使用命令行模式（默认启动GUI）')
    
    # SNR参数
    parser.add_argument('--snr-min', type=float, default=-5,
                        help='最小SNR (dB), 默认: -5')
    parser.add_argument('--snr-max', type=float, default=30,
                        help='最大SNR (dB), 默认: 30')
    parser.add_argument('--snr-step', type=float, default=0.5,
                        help='SNR步长 (dB), 默认: 0.5')
    
    # MCS参数
    parser.add_argument('--mcs-table', type=str, default='LTE',
                        choices=['LTE', 'NR_Table1', 'NR_Table2'],
                        help='MCS表选择, 默认: LTE')
    parser.add_argument('--target-bler', type=float, default=0.10,
                        help='目标BLER, 默认: 0.10')
    parser.add_argument('--bandwidth', type=float, default=20,
                        help='带宽 (MHz), 默认: 20')
    
    # HARQ参数
    parser.add_argument('--harq', action='store_true',
                        help='启用HARQ')
    parser.add_argument('--harq-retx', type=int, default=4,
                        help='HARQ最大重传次数, 默认: 4')
    parser.add_argument('--harq-gain', type=float, default=3.0,
                        help='HARQ合并增益 (dB), 默认: 3.0')
    
    # 输出参数
    parser.add_argument('--output', '-o', type=str,
                        help='输出CSV文件路径')
    parser.add_argument('--plot', action='store_true',
                        help='生成图表')
    parser.add_argument('--save-plots', type=str,
                        help='保存图表到文件')
    
    args = parser.parse_args()
    
    if args.cli:
        run_cli(args)
    else:
        print("=" * 60)
        print("AMC策略仿真与吞吐评估软件")
        print("Adaptive Modulation and Coding Simulation")  
        print("=" * 60)
        print("\n启动图形界面...\n")
        run_gui()


if __name__ == '__main__':
    main()
