"""
Threshold Search Algorithm
门限自动搜索算法 - 二分法查找满足目标BLER的SNR门限
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ThresholdResult:
    """门限搜索结果"""
    mcs_index: int          # MCS索引
    snr_threshold: float    # SNR门限 (dB)
    actual_bler: float      # 实际BLER值
    modulation: str         # 调制方式
    spectral_efficiency: float  # 频谱效率


class ThresholdSearcher:
    """
    门限搜索器
    
    使用二分法搜索满足目标BLER的SNR门限。
    """
    
    def __init__(self, 
                 snr_range: Tuple[float, float] = (-10.0, 35.0),
                 precision: float = 0.1,
                 max_iterations: int = 50):
        """
        初始化搜索器
        
        Args:
            snr_range: SNR搜索范围 (min, max) dB
            precision: 搜索精度 (dB)
            max_iterations: 最大迭代次数
        """
        self.snr_min, self.snr_max = snr_range
        self.precision = precision
        self.max_iterations = max_iterations
    
    def binary_search_threshold(self, 
                                 bler_func,
                                 target_bler: float,
                                 mcs_index: int) -> Tuple[float, float]:
        """
        二分法搜索SNR门限
        
        Args:
            bler_func: BLER计算函数，输入(snr, mcs_index)返回BLER
            target_bler: 目标BLER
            mcs_index: MCS索引
            
        Returns:
            (SNR门限, 实际BLER值)
        """
        low = self.snr_min
        high = self.snr_max
        
        # 检查边界条件
        bler_low = bler_func(low, mcs_index)
        bler_high = bler_func(high, mcs_index)
        
        # 如果在最小SNR时BLER已经低于目标，返回最小SNR
        if bler_low <= target_bler:
            return low, bler_low
        
        # 如果在最大SNR时BLER仍然高于目标，返回最大SNR
        if bler_high > target_bler:
            return high, bler_high
        
        # 二分法搜索
        for _ in range(self.max_iterations):
            mid = (low + high) / 2
            bler_mid = bler_func(mid, mcs_index)
            
            if abs(high - low) < self.precision:
                return mid, bler_mid
            
            if bler_mid > target_bler:
                # BLER太高，需要更高的SNR
                low = mid
            else:
                # BLER足够低，可以尝试更低的SNR
                high = mid
        
        # 返回最终结果
        final_snr = (low + high) / 2
        return final_snr, bler_func(final_snr, mcs_index)
    
    def search_all_thresholds(self,
                               mcs_table,
                               bler_model,
                               target_bler: float,
                               margin_db: float = 0.0) -> List[ThresholdResult]:
        """
        搜索所有MCS的门限
        
        Args:
            mcs_table: MCS表对象
            bler_model: BLER模型对象
            target_bler: 目标BLER
            margin_db: 额外裕量 (dB)
            
        Returns:
            门限结果列表
        """
        results = []
        
        for entry in mcs_table:
            # 搜索门限
            snr_threshold, actual_bler = self.binary_search_threshold(
                bler_model.calculate_bler,
                target_bler,
                entry.index
            )
            
            # 添加裕量
            snr_threshold += margin_db
            
            results.append(ThresholdResult(
                mcs_index=entry.index,
                snr_threshold=snr_threshold,
                actual_bler=actual_bler,
                modulation=entry.modulation,
                spectral_efficiency=entry.spectral_efficiency
            ))
        
        return results
    
    def get_switching_thresholds(self, threshold_results: List[ThresholdResult]) -> Dict[int, float]:
        """
        获取MCS切换门限字典
        
        Args:
            threshold_results: 门限搜索结果列表
            
        Returns:
            {mcs_index: snr_threshold}字典
        """
        return {r.mcs_index: r.snr_threshold for r in threshold_results}
    
    def validate_thresholds(self, 
                            thresholds: Dict[int, float],
                            bler_model,
                            target_bler: float) -> List[Dict]:
        """
        验证门限的正确性
        
        Args:
            thresholds: 门限字典
            bler_model: BLER模型
            target_bler: 目标BLER
            
        Returns:
            验证结果列表
        """
        validation_results = []
        
        for mcs_index, snr in thresholds.items():
            actual_bler = bler_model.calculate_bler(snr, mcs_index)
            is_valid = actual_bler <= target_bler * 1.1  # 允许10%误差
            
            validation_results.append({
                'mcs_index': mcs_index,
                'snr_threshold': snr,
                'actual_bler': actual_bler,
                'target_bler': target_bler,
                'is_valid': is_valid,
                'error_ratio': actual_bler / target_bler
            })
        
        return validation_results
    
    def optimize_thresholds_for_throughput(self,
                                           mcs_table,
                                           bler_model,
                                           snr_range: np.ndarray,
                                           initial_thresholds: Dict[int, float]) -> Dict[int, float]:
        """
        优化门限以最大化吞吐量
        
        使用梯度下降或网格搜索优化门限。
        
        Args:
            mcs_table: MCS表
            bler_model: BLER模型
            snr_range: SNR范围
            initial_thresholds: 初始门限
            
        Returns:
            优化后的门限
        """
        # 简化实现：使用网格微调
        optimized = dict(initial_thresholds)
        step_size = 0.5  # dB
        
        for mcs_index in optimized:
            best_snr = optimized[mcs_index]
            best_throughput = self._evaluate_throughput_at_threshold(
                mcs_table, bler_model, snr_range, optimized, mcs_index
            )
            
            # 向上向下各搜索几个点
            for delta in [-2, -1, -0.5, 0.5, 1, 2]:
                test_snr = optimized[mcs_index] + delta
                test_thresholds = dict(optimized)
                test_thresholds[mcs_index] = test_snr
                
                tp = self._evaluate_throughput_at_threshold(
                    mcs_table, bler_model, snr_range, test_thresholds, mcs_index
                )
                
                if tp > best_throughput:
                    best_throughput = tp
                    best_snr = test_snr
            
            optimized[mcs_index] = best_snr
        
        return optimized
    
    def _evaluate_throughput_at_threshold(self, mcs_table, bler_model, 
                                          snr_range, thresholds, mcs_index) -> float:
        """评估特定门限下的总吞吐量"""
        total_tp = 0
        entry = mcs_table.get_entry(mcs_index)
        if entry is None:
            return 0
        
        threshold = thresholds.get(mcs_index, float('inf'))
        
        for snr in snr_range:
            if snr >= threshold:
                bler = bler_model.calculate_bler(snr, mcs_index)
                tp = entry.spectral_efficiency * (1 - bler)
                total_tp += tp
        
        return total_tp
