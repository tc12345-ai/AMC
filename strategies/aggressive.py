"""
Aggressive AMC Strategy
激进策略 - 高目标BLER，最大化频谱效率
"""

from .base_strategy import AMCStrategy, StrategyConfig


class AggressiveStrategy(AMCStrategy):
    """
    激进AMC策略
    
    特点:
    - 目标BLER = 20%（允许较高错误率）
    - 无SNR裕量
    - 最大化频谱效率
    - 依赖HARQ来恢复错误
    - 适用于有HARQ支持且延迟容忍的场景
    """
    
    def __init__(self, target_bler: float = 0.20, margin_db: float = 0.0):
        """
        初始化激进策略
        
        Args:
            target_bler: 目标BLER，默认20%
            margin_db: SNR裕量，默认0dB
        """
        config = StrategyConfig(
            name="激进策略 (Aggressive)",
            target_bler=target_bler,
            margin_db=margin_db,
            description="高目标BLER，最大化频谱效率，依赖HARQ"
        )
        super().__init__(config)
    
    def select_mcs(self, snr_db: float) -> int:
        """
        激进的MCS选择
        
        在门限附近也倾向于选择更高的MCS，
        最大化频谱效率。
        """
        if not self._sorted_mcs:
            return 0
        
        selected_mcs = self._sorted_mcs[0][1]
        
        for threshold, mcs in self._sorted_mcs:
            # 激进策略：接近门限就可以选择
            # 使用较小的安全余量
            if snr_db >= threshold - 0.5:  # 允许0.5dB的负余量
                selected_mcs = mcs
            else:
                break
        
        return selected_mcs
    
    def select_mcs_optimistic(self, snr_db: float) -> int:
        """
        更激进的MCS选择（乐观模式）
        
        只要BLER在可接受范围内，就选择更高的MCS
        """
        if not self._sorted_mcs:
            return 0
        
        best_mcs = self._sorted_mcs[0][1]
        max_acceptable_bler = 0.30  # 可接受的最高BLER
        
        for _, mcs in self._sorted_mcs:
            bler = self.get_bler(snr_db, mcs)
            if bler <= max_acceptable_bler:
                best_mcs = mcs
        
        return best_mcs
