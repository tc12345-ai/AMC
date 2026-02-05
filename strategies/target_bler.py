"""
Target BLER AMC Strategy
目标BLER策略 - 精确控制目标BLER
"""

from .base_strategy import AMCStrategy, StrategyConfig


class TargetBLERStrategy(AMCStrategy):
    """
    目标BLER策略
    
    特点:
    - 用户指定目标BLER（默认10%）
    - 无额外裕量
    - 在可靠性和吞吐量之间取得平衡
    - 适用于大多数通用场景
    """
    
    def __init__(self, target_bler: float = 0.10, margin_db: float = 0.0):
        """
        初始化目标BLER策略
        
        Args:
            target_bler: 目标BLER，默认10%
            margin_db: SNR裕量，默认0dB
        """
        self._user_target_bler = target_bler
        
        config = StrategyConfig(
            name=f"目标BLER策略 ({target_bler*100:.1f}%)",
            target_bler=target_bler,
            margin_db=margin_db,
            description=f"精确控制BLER为{target_bler*100:.1f}%"
        )
        super().__init__(config)
    
    def select_mcs(self, snr_db: float) -> int:
        """
        标准MCS选择
        
        选择满足目标BLER的最高MCS
        """
        if not self._sorted_mcs:
            return 0
        
        selected_mcs = self._sorted_mcs[0][1]
        
        for threshold, mcs in self._sorted_mcs:
            if snr_db >= threshold:
                selected_mcs = mcs
            else:
                break
        
        return selected_mcs
    
    def select_mcs_with_hysteresis(self, snr_db: float, 
                                    current_mcs: int,
                                    hysteresis_db: float = 1.0) -> int:
        """
        带迟滞的MCS选择（避免频繁切换）
        
        Args:
            snr_db: 当前SNR
            current_mcs: 当前使用的MCS
            hysteresis_db: 迟滞值 (dB)
            
        Returns:
            新的MCS索引
        """
        if not self._sorted_mcs:
            return 0
        
        # 找到当前MCS的门限
        current_threshold = self.thresholds.get(current_mcs, float('-inf'))
        
        # 检查是否需要升级（切换到更高的MCS）
        for threshold, mcs in self._sorted_mcs:
            if mcs > current_mcs:
                # 升级需要高于门限 + 迟滞
                if snr_db >= threshold + hysteresis_db:
                    return mcs
        
        # 检查是否需要降级（切换到更低的MCS）
        for threshold, mcs in reversed(self._sorted_mcs):
            if mcs < current_mcs:
                # 降级在低于门限 - 迟滞时触发
                if snr_db < self.thresholds.get(current_mcs, 0) - hysteresis_db:
                    return self.select_mcs(snr_db)
        
        return current_mcs
    
    def set_target_bler(self, target_bler: float):
        """
        更新目标BLER
        
        Args:
            target_bler: 新的目标BLER
        """
        self._user_target_bler = target_bler
        self.config = StrategyConfig(
            name=f"目标BLER策略 ({target_bler*100:.1f}%)",
            target_bler=target_bler,
            margin_db=self.config.margin_db,
            description=f"精确控制BLER为{target_bler*100:.1f}%"
        )
        
        # 重新计算门限
        if self.mcs_table and self.bler_model:
            self._calculate_thresholds()
            self._build_sorted_mcs()
