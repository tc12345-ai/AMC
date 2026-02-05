"""
AMC Simulator
AMC仿真引擎 - 核心仿真逻辑
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from config.mcs_tables import MCSTable, get_mcs_table, AVAILABLE_MCS_TABLES
    from core.bler_model import AWGNBLERModel
    from core.throughput import ThroughputCalculator, ThroughputResult
    from core.threshold_search import ThresholdSearcher, ThresholdResult
    from core.harq import HARQModel, HARQConfig
    from strategies.conservative import ConservativeStrategy
    from strategies.aggressive import AggressiveStrategy
    from strategies.target_bler import TargetBLERStrategy
except ImportError:
    from ..config.mcs_tables import MCSTable, get_mcs_table, AVAILABLE_MCS_TABLES
    from ..core.bler_model import AWGNBLERModel
    from ..core.throughput import ThroughputCalculator, ThroughputResult
    from ..core.threshold_search import ThresholdSearcher, ThresholdResult
    from ..core.harq import HARQModel, HARQConfig
    from ..strategies.conservative import ConservativeStrategy
    from ..strategies.aggressive import AggressiveStrategy
    from ..strategies.target_bler import TargetBLERStrategy


@dataclass
class SimulationConfig:
    """仿真配置"""
    snr_min: float = -5.0           # 最小SNR (dB)
    snr_max: float = 30.0           # 最大SNR (dB)
    snr_step: float = 0.5           # SNR步长 (dB)
    target_bler: float = 0.10       # 目标BLER
    bandwidth_mhz: float = 20.0     # 带宽 (MHz)
    mcs_table_name: str = "LTE"     # MCS表名称
    enable_harq: bool = False       # 是否启用HARQ
    harq_max_retx: int = 4          # HARQ最大重传次数
    harq_gain_db: float = 3.0       # HARQ合并增益 (dB)


@dataclass
class SimulationResult:
    """仿真结果"""
    config: SimulationConfig
    snr_range: np.ndarray
    
    # 各策略结果
    strategy_results: Dict[str, Dict] = field(default_factory=dict)
    
    # 门限信息
    thresholds: Dict[str, List[ThresholdResult]] = field(default_factory=dict)
    
    # Shannon容量
    shannon_capacity: np.ndarray = None
    
    # HARQ分析（如果启用）
    harq_analysis: Dict = None


class AMCSimulator:
    """
    AMC仿真器
    
    主要功能:
    1. 配置MCS表和BLER模型
    2. 运行多策略对比仿真
    3. 计算吞吐量和频谱效率
    4. 分析HARQ性能
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        初始化仿真器
        
        Args:
            config: 仿真配置
        """
        self.config = config or SimulationConfig()
        
        # 初始化组件
        self.mcs_table: MCSTable = None
        self.bler_model = AWGNBLERModel()
        self.throughput_calc = ThroughputCalculator(self.config.bandwidth_mhz)
        self.threshold_searcher = ThresholdSearcher()
        self.harq_model = HARQModel()
        
        # 策略
        self.strategies: Dict[str, any] = {}
        
        # 结果
        self.result: SimulationResult = None
    
    def setup(self):
        """设置仿真环境"""
        # 加载MCS表
        self.mcs_table = get_mcs_table(self.config.mcs_table_name)
        
        # 设置BLER模型参数
        self.bler_model.set_params_from_mcs_table(self.mcs_table, target_bler=0.1)
        
        # 设置吞吐量计算器
        self.throughput_calc.set_bandwidth(self.config.bandwidth_mhz)
        
        # 设置HARQ模型
        self.harq_model.set_config(
            max_retx=self.config.harq_max_retx,
            gain_db=self.config.harq_gain_db,
            enable=self.config.enable_harq
        )
        
        # 创建策略
        self._create_strategies()
    
    def _create_strategies(self):
        """创建所有策略"""
        self.strategies = {
            'conservative': ConservativeStrategy(target_bler=0.01, margin_db=3.0),
            'aggressive': AggressiveStrategy(target_bler=0.20, margin_db=0.0),
            'target_bler': TargetBLERStrategy(target_bler=self.config.target_bler, margin_db=0.0)
        }
        
        # 设置策略
        for strategy in self.strategies.values():
            strategy.setup(self.mcs_table, self.bler_model, self.threshold_searcher)
    
    def set_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 重新设置
        self.setup()
    
    def run(self) -> SimulationResult:
        """
        运行仿真
        
        Returns:
            仿真结果
        """
        if self.mcs_table is None:
            self.setup()
        
        # 生成SNR范围
        snr_range = np.arange(
            self.config.snr_min,
            self.config.snr_max + self.config.snr_step,
            self.config.snr_step
        )
        
        # 初始化结果
        self.result = SimulationResult(
            config=self.config,
            snr_range=snr_range
        )
        
        # 计算Shannon容量
        self.result.shannon_capacity = self.throughput_calc.calculate_theoretical_max_throughput(snr_range)
        
        # 运行各策略
        for name, strategy in self.strategies.items():
            # 评估策略
            eval_result = strategy.evaluate(snr_range)
            
            # 计算吞吐量
            throughput = np.zeros(len(snr_range))
            for i, snr in enumerate(snr_range):
                se = eval_result['effective_spectral_efficiency'][i]
                
                if self.config.enable_harq:
                    initial_bler = eval_result['bler'][i]
                    tp, _ = self.harq_model.calculate_effective_throughput(
                        eval_result['spectral_efficiency'][i],
                        initial_bler,
                        self.config.bandwidth_mhz * 1e6
                    )
                    throughput[i] = tp
                else:
                    throughput[i] = self.config.bandwidth_mhz * se
            
            eval_result['throughput_mbps'] = throughput
            self.result.strategy_results[name] = eval_result
            
            # 保存门限
            self.result.thresholds[name] = self.threshold_searcher.search_all_thresholds(
                self.mcs_table, self.bler_model,
                strategy.target_bler, strategy.margin_db
            )
        
        # HARQ分析
        if self.config.enable_harq:
            self.result.harq_analysis = self._analyze_harq(snr_range)
        
        return self.result
    
    def _analyze_harq(self, snr_range: np.ndarray) -> Dict:
        """分析HARQ性能"""
        # 选择中间的MCS进行分析
        mid_mcs = len(self.mcs_table) // 2
        mcs_entry = list(self.mcs_table)[mid_mcs]
        
        analysis = self.harq_model.analyze_harq_performance(
            snr_range, self.bler_model.calculate_bler, mcs_entry.index
        )
        analysis['mcs_index'] = mcs_entry.index
        analysis['mcs_name'] = mcs_entry.modulation
        
        return analysis
    
    def compare_strategies(self) -> Dict:
        """
        对比各策略性能
        
        Returns:
            对比结果字典
        """
        if self.result is None:
            self.run()
        
        comparison = {}
        
        for name, result in self.result.strategy_results.items():
            # 平均吞吐量
            avg_throughput = np.mean(result['throughput_mbps'])
            
            # 峰值吞吐量
            peak_throughput = np.max(result['throughput_mbps'])
            
            # 平均BLER
            avg_bler = np.mean(result['bler'])
            
            # 频谱效率
            avg_se = np.mean(result['effective_spectral_efficiency'])
            
            comparison[name] = {
                'avg_throughput_mbps': avg_throughput,
                'peak_throughput_mbps': peak_throughput,
                'avg_bler': avg_bler,
                'avg_spectral_efficiency': avg_se
            }
        
        return comparison
    
    def get_threshold_table(self, strategy_name: str = 'target_bler') -> List[Dict]:
        """
        获取门限表
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            门限表数据
        """
        if self.result is None or strategy_name not in self.result.thresholds:
            return []
        
        thresholds = self.result.thresholds[strategy_name]
        
        return [
            {
                'MCS': t.mcs_index,
                'Modulation': t.modulation,
                'SE (bits/s/Hz)': f"{t.spectral_efficiency:.2f}",
                'SNR Threshold (dB)': f"{t.snr_threshold:.1f}",
                'BLER': f"{t.actual_bler:.1%}"
            }
            for t in thresholds
        ]
    
    def export_results(self, filename: str):
        """
        导出结果到CSV
        
        Args:
            filename: 输出文件名
        """
        if self.result is None:
            return
        
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入头部
            headers = ['SNR (dB)', 'Shannon Capacity (Mbps)']
            for name in self.result.strategy_results:
                headers.extend([
                    f'{name} - Throughput (Mbps)',
                    f'{name} - SE (bits/s/Hz)',
                    f'{name} - BLER',
                    f'{name} - MCS'
                ])
            writer.writerow(headers)
            
            # 写入数据
            for i, snr in enumerate(self.result.snr_range):
                row = [f"{snr:.1f}", f"{self.result.shannon_capacity[i]:.2f}"]
                
                for name, result in self.result.strategy_results.items():
                    row.extend([
                        f"{result['throughput_mbps'][i]:.2f}",
                        f"{result['effective_spectral_efficiency'][i]:.3f}",
                        f"{result['bler'][i]:.4f}",
                        str(result['selected_mcs'][i])
                    ])
                
                writer.writerow(row)
        
        print(f"结果已导出到: {filename}")
    
    @staticmethod
    def get_available_mcs_tables() -> List[str]:
        """获取可用的MCS表列表"""
        return list(AVAILABLE_MCS_TABLES.keys())
