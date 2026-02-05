"""
Conservative AMC Strategy
保守策略 - 低目标BLER + 额外裕量
"""

from .base_strategy import AMCStrategy, StrategyConfig


class ConservativeStrategy(AMCStrategy):
    """
    保守AMC策略
    
    特点:
    - 目标BLER = 1%（低错误率）
    - 额外3dB SNR裕量
    - 优先保证可靠性，牺牲部分吞吐量
    - 适用于对延迟敏感或不支持HARQ的场景
    """
    
    def __init__(self, target_bler: float = 0.01, margin_db: float = 3.0):
        """
        初始化保守策略
        
        Args:
            target_bler: 目标BLER，默认1%
            margin_db: SNR裕量，默认3dB
        """
        config = StrategyConfig(
            name="保守策略 (Conservative)",
            target_bler=target_bler,
            margin_db=margin_db,
            description="低目标BLER + 额外裕量，优先保证可靠性"
        )
        super().__init__(config)
    
    def select_mcs(self, snr_db: float) -> int:
        """
        保守的MCS选择
        
        选择满足门限且频谱效率最高的MCS，
        但在边界情况下倾向于选择更低的MCS。
        """
        if not self._sorted_mcs:
            return 0
        
        selected_mcs = self._sorted_mcs[0][1]  # 默认最低MCS
        
        for threshold, mcs in self._sorted_mcs:
            # 保守策略：SNR必须明显高于门限
            if snr_db >= threshold:
                selected_mcs = mcs
            else:
                break
        
        return selected_mcs
