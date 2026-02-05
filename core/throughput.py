"""
Throughput Calculator
吞吐量计算模块
"""

import numpy as np
from typing import Union, List, Tuple
from dataclasses import dataclass


@dataclass
class ThroughputResult:
    """吞吐量计算结果"""
    snr_db: np.ndarray              # SNR数组 (dB)
    throughput: np.ndarray          # 吞吐量 (Mbps)
    spectral_efficiency: np.ndarray # 频谱效率 (bits/s/Hz)
    selected_mcs: np.ndarray        # 选择的MCS索引
    bler: np.ndarray                # 实际BLER


class ThroughputCalculator:
    """
    吞吐量计算器
    
    计算公式:
    Throughput = Bandwidth * SE * (1 - BLER)
    
    其中SE是频谱效率，考虑HARQ重传开销。
    """
    
    def __init__(self, bandwidth_mhz: float = 20.0):
        """
        初始化计算器
        
        Args:
            bandwidth_mhz: 带宽 (MHz)
        """
        self.bandwidth_mhz = bandwidth_mhz
        self.bandwidth_hz = bandwidth_mhz * 1e6
    
    def set_bandwidth(self, bandwidth_mhz: float):
        """设置带宽"""
        self.bandwidth_mhz = bandwidth_mhz
        self.bandwidth_hz = bandwidth_mhz * 1e6
    
    def calculate_instantaneous_throughput(self, 
                                           spectral_efficiency: float,
                                           bler: float) -> float:
        """
        计算瞬时吞吐量
        
        Args:
            spectral_efficiency: 频谱效率 (bits/s/Hz)
            bler: 块错误率
            
        Returns:
            吞吐量 (Mbps)
        """
        # 有效频谱效率 = SE * (1 - BLER)
        effective_se = spectral_efficiency * (1 - bler)
        # 吞吐量 = 带宽 * 有效频谱效率
        throughput_bps = self.bandwidth_hz * effective_se
        return throughput_bps / 1e6  # 转换为Mbps
    
    def calculate_throughput_with_harq(self,
                                       spectral_efficiency: float,
                                       initial_bler: float,
                                       harq_gain_db: float = 3.0,
                                       max_retx: int = 4) -> Tuple[float, float]:
        """
        计算考虑HARQ的吞吐量
        
        Args:
            spectral_efficiency: 频谱效率 (bits/s/Hz)
            initial_bler: 初始BLER
            harq_gain_db: 每次重传的SNR增益 (dB)
            max_retx: 最大重传次数
            
        Returns:
            (吞吐量 Mbps, 平均传输次数)
        """
        # 简化HARQ模型：每次重传BLER按比例下降
        # 假设Chase Combining增益
        
        # 计算残余BLER（经过所有重传后）
        residual_bler = initial_bler
        total_bler_reduction = 1
        
        for i in range(max_retx):
            # 每次重传后BLER降低（简化模型）
            # 实际上HARQ合并会提供SNR增益
            reduction_factor = 10 ** (-harq_gain_db / 10)
            residual_bler *= reduction_factor
            if residual_bler < 1e-6:
                break
        
        # 计算平均传输次数
        # E[transmissions] ≈ 1 + BLER + BLER^2 + ... (几何级数)
        if initial_bler < 0.99:
            avg_transmissions = 1 / (1 - initial_bler)
            avg_transmissions = min(avg_transmissions, max_retx + 1)
        else:
            avg_transmissions = max_retx + 1
        
        # 有效频谱效率需要除以平均传输次数
        effective_se = spectral_efficiency * (1 - residual_bler) / avg_transmissions
        throughput_bps = self.bandwidth_hz * effective_se
        
        return throughput_bps / 1e6, avg_transmissions
    
    def calculate_throughput_curve(self,
                                   snr_db: np.ndarray,
                                   mcs_selection_func,
                                   se_func,
                                   bler_func,
                                   use_harq: bool = False,
                                   harq_params: dict = None) -> ThroughputResult:
        """
        计算吞吐量曲线
        
        Args:
            snr_db: SNR数组 (dB)
            mcs_selection_func: MCS选择函数，输入SNR返回MCS索引
            se_func: 频谱效率函数，输入MCS索引返回SE
            bler_func: BLER函数，输入(SNR, MCS)返回BLER
            use_harq: 是否使用HARQ
            harq_params: HARQ参数字典
            
        Returns:
            ThroughputResult对象
        """
        snr_db = np.atleast_1d(snr_db)
        n_points = len(snr_db)
        
        throughput = np.zeros(n_points)
        spectral_efficiency = np.zeros(n_points)
        selected_mcs = np.zeros(n_points, dtype=int)
        bler = np.zeros(n_points)
        
        harq_params = harq_params or {}
        harq_gain = harq_params.get('gain_db', 3.0)
        max_retx = harq_params.get('max_retx', 4)
        
        for i, snr in enumerate(snr_db):
            # 选择MCS
            mcs = mcs_selection_func(snr)
            selected_mcs[i] = mcs
            
            # 获取频谱效率
            se = se_func(mcs)
            spectral_efficiency[i] = se
            
            # 计算BLER
            current_bler = bler_func(snr, mcs)
            bler[i] = current_bler
            
            # 计算吞吐量
            if use_harq:
                tp, _ = self.calculate_throughput_with_harq(
                    se, current_bler, harq_gain, max_retx
                )
            else:
                tp = self.calculate_instantaneous_throughput(se, current_bler)
            
            throughput[i] = tp
        
        return ThroughputResult(
            snr_db=snr_db,
            throughput=throughput,
            spectral_efficiency=spectral_efficiency * (1 - bler),  # 有效频谱效率
            selected_mcs=selected_mcs,
            bler=bler
        )
    
    def calculate_theoretical_max_throughput(self, snr_db: np.ndarray) -> np.ndarray:
        """
        计算Shannon容量极限
        
        Args:
            snr_db: SNR数组 (dB)
            
        Returns:
            理论最大吞吐量 (Mbps)
        """
        snr_linear = 10 ** (snr_db / 10)
        # Shannon容量: C = B * log2(1 + SNR)
        capacity_bps_hz = np.log2(1 + snr_linear)
        capacity_mbps = self.bandwidth_hz * capacity_bps_hz / 1e6
        return capacity_mbps
