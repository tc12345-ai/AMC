"""
HARQ Model
HARQ简单模型 - Chase Combining
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class HARQConfig:
    """HARQ配置"""
    max_retransmissions: int = 4    # 最大重传次数
    combining_gain_db: float = 3.0  # 每次合并的SNR增益 (dB)
    enable: bool = True             # 是否启用HARQ


@dataclass
class HARQResult:
    """HARQ仿真结果"""
    initial_bler: float         # 初始BLER
    residual_bler: float        # 残余BLER（经过所有重传后）
    avg_transmissions: float    # 平均传输次数
    throughput_gain: float      # 吞吐量增益比例


class HARQModel:
    """
    HARQ模型
    
    实现Chase Combining的简化模型：
    - 每次重传后合并所有接收信号
    - SNR按次数累加（线性域）
    - BLER根据合并后的等效SNR重新计算
    """
    
    def __init__(self, config: HARQConfig = None):
        """
        初始化HARQ模型
        
        Args:
            config: HARQ配置
        """
        self.config = config or HARQConfig()
    
    def set_config(self, max_retx: int = 4, gain_db: float = 3.0, enable: bool = True):
        """设置HARQ参数"""
        self.config = HARQConfig(
            max_retransmissions=max_retx,
            combining_gain_db=gain_db,
            enable=enable
        )
    
    def calculate_combined_snr(self, initial_snr_db: float, num_transmissions: int) -> float:
        """
        计算合并后的等效SNR
        
        Chase Combining: SNR_combined = N * SNR_initial (线性域)
        
        Args:
            initial_snr_db: 初始SNR (dB)
            num_transmissions: 传输次数
            
        Returns:
            合并后的SNR (dB)
        """
        if num_transmissions <= 0:
            return initial_snr_db
        
        # 转换到线性域
        snr_linear = 10 ** (initial_snr_db / 10)
        
        # Chase Combining增益
        combined_snr_linear = snr_linear * num_transmissions
        
        # 转回dB
        return 10 * np.log10(combined_snr_linear)
    
    def calculate_residual_bler(self, 
                                 initial_bler: float,
                                 bler_func=None,
                                 snr_db: float = None,
                                 mcs_index: int = None) -> Tuple[float, float]:
        """
        计算残余BLER（经过最大重传后）
        
        Args:
            initial_bler: 初始BLER
            bler_func: BLER计算函数（可选，用于更精确计算）
            snr_db: 初始SNR (dB)
            mcs_index: MCS索引
            
        Returns:
            (残余BLER, 平均传输次数)
        """
        if not self.config.enable:
            return initial_bler, 1.0
        
        max_tx = self.config.max_retransmissions + 1  # 包括初始传输
        
        if bler_func is not None and snr_db is not None and mcs_index is not None:
            # 使用精确的BLER函数计算
            return self._calculate_with_bler_func(
                bler_func, snr_db, mcs_index, max_tx
            )
        else:
            # 使用简化模型
            return self._calculate_simplified(initial_bler, max_tx)
    
    def _calculate_simplified(self, initial_bler: float, max_tx: int) -> Tuple[float, float]:
        """简化的HARQ模型"""
        # 简化假设：每次重传后BLER按固定比例下降
        gain_factor = 10 ** (-self.config.combining_gain_db / 10)
        
        residual_bler = initial_bler
        prob_success_by_tx = []
        
        current_bler = initial_bler
        for tx in range(max_tx):
            prob_success_by_tx.append((1 - current_bler) if tx == 0 
                                      else residual_bler * (1 - current_bler))
            
            if tx > 0:
                residual_bler *= current_bler
            
            # 下次传输的BLER更低（由于合并增益）
            current_bler *= gain_factor
            current_bler = max(current_bler, 1e-10)
        
        # 最终残余BLER
        final_residual = residual_bler * current_bler
        
        # 计算平均传输次数
        # E[tx] = sum(i * P(success at i-th attempt))
        avg_tx = 0
        cumulative_success = 0
        current_fail_prob = 1.0
        
        for tx in range(1, max_tx + 1):
            bler_at_tx = initial_bler * (gain_factor ** (tx - 1))
            success_prob = current_fail_prob * (1 - bler_at_tx)
            avg_tx += tx * success_prob
            cumulative_success += success_prob
            current_fail_prob *= bler_at_tx
        
        # 如果还没成功，加上最后的传输
        avg_tx += (max_tx + 1) * current_fail_prob
        
        return final_residual, min(avg_tx, max_tx)
    
    def _calculate_with_bler_func(self, bler_func, snr_db: float, 
                                   mcs_index: int, max_tx: int) -> Tuple[float, float]:
        """使用精确BLER函数的计算"""
        avg_tx = 0
        cumulative_fail = 1.0
        
        for tx in range(1, max_tx + 1):
            # 计算合并后的SNR
            combined_snr = self.calculate_combined_snr(snr_db, tx)
            
            # 计算该次传输后的BLER
            bler_after_tx = bler_func(combined_snr, mcs_index)
            
            # 这次成功的概率
            success_prob = cumulative_fail * (1 - bler_after_tx)
            avg_tx += tx * success_prob
            
            # 更新累积失败概率
            cumulative_fail *= bler_after_tx
        
        # 残余BLER就是最后的累积失败概率
        residual_bler = cumulative_fail
        
        # 如果一直失败到最后
        avg_tx += max_tx * cumulative_fail
        
        return residual_bler, avg_tx
    
    def calculate_effective_throughput(self,
                                        spectral_efficiency: float,
                                        initial_bler: float,
                                        bandwidth_hz: float = 20e6) -> Tuple[float, HARQResult]:
        """
        计算考虑HARQ的有效吞吐量
        
        Args:
            spectral_efficiency: 频谱效率 (bits/s/Hz)
            initial_bler: 初始BLER
            bandwidth_hz: 带宽 (Hz)
            
        Returns:
            (有效吞吐量 Mbps, HARQ结果)
        """
        residual_bler, avg_tx = self.calculate_residual_bler(initial_bler)
        
        # 有效频谱效率 = SE * (1 - residual_BLER) / avg_transmissions
        effective_se = spectral_efficiency * (1 - residual_bler) / avg_tx
        
        # 计算吞吐量增益
        baseline_tp = spectral_efficiency * (1 - initial_bler)
        throughput_gain = effective_se / baseline_tp if baseline_tp > 0 else 0
        
        throughput_mbps = bandwidth_hz * effective_se / 1e6
        
        result = HARQResult(
            initial_bler=initial_bler,
            residual_bler=residual_bler,
            avg_transmissions=avg_tx,
            throughput_gain=throughput_gain
        )
        
        return throughput_mbps, result
    
    def analyze_harq_performance(self, 
                                  snr_range: np.ndarray,
                                  bler_func,
                                  mcs_index: int) -> dict:
        """
        分析HARQ性能
        
        Args:
            snr_range: SNR范围 (dB)
            bler_func: BLER计算函数
            mcs_index: MCS索引
            
        Returns:
            性能分析字典
        """
        n_points = len(snr_range)
        
        initial_bler = np.zeros(n_points)
        residual_bler = np.zeros(n_points)
        avg_transmissions = np.zeros(n_points)
        
        for i, snr in enumerate(snr_range):
            initial_bler[i] = bler_func(snr, mcs_index)
            res_bler, avg_tx = self.calculate_residual_bler(
                initial_bler[i], bler_func, snr, mcs_index
            )
            residual_bler[i] = res_bler
            avg_transmissions[i] = avg_tx
        
        return {
            'snr_db': snr_range,
            'initial_bler': initial_bler,
            'residual_bler': residual_bler,
            'avg_transmissions': avg_transmissions,
            'bler_reduction': initial_bler / np.maximum(residual_bler, 1e-10)
        }
