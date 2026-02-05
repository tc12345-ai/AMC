"""
Base AMC Strategy
AMC策略基类
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str                    # 策略名称
    target_bler: float          # 目标BLER
    margin_db: float = 0.0      # SNR裕量 (dB)
    description: str = ""       # 策略描述


class AMCStrategy(ABC):
    """
    AMC策略抽象基类
    
    定义MCS选择的接口和通用功能。
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化策略
        
        Args:
            config: 策略配置
        """
        self.config = config
        self.mcs_table = None
        self.bler_model = None
        self.thresholds: Dict[int, float] = {}
        self.threshold_results = []
        self._sorted_mcs: List[Tuple[float, int]] = []
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def target_bler(self) -> float:
        return self.config.target_bler
    
    @property
    def margin_db(self) -> float:
        return self.config.margin_db
    
    def setup(self, mcs_table, bler_model, threshold_searcher=None):
        """
        设置MCS表和BLER模型
        
        Args:
            mcs_table: MCS表对象
            bler_model: BLER模型对象
            threshold_searcher: 门限搜索器（可选）
        """
        self.mcs_table = mcs_table

        # 为每个策略创建独立BLER模型副本，避免策略间参数互相覆盖
        self.bler_model = copy.deepcopy(bler_model)

        # 初始化BLER模型参数
        self.bler_model.set_params_from_mcs_table(mcs_table, target_bler=self.target_bler)
        
        # 计算门限
        self._calculate_thresholds(threshold_searcher)
        
        # 构建排序的MCS列表（按门限从低到高）
        self._build_sorted_mcs()
    
    def _calculate_thresholds(self, threshold_searcher=None):
        """计算所有MCS的SNR门限"""
        if threshold_searcher is None:
            try:
                from core.threshold_search import ThresholdSearcher
            except ImportError:
                from ..core.threshold_search import ThresholdSearcher
            threshold_searcher = ThresholdSearcher()
        
        results = threshold_searcher.search_all_thresholds(
            self.mcs_table,
            self.bler_model,
            self.target_bler,
            self.margin_db
        )
        
        self.threshold_results = results
        self.thresholds = {r.mcs_index: r.snr_threshold for r in results}
    
    def _build_sorted_mcs(self):
        """构建按门限排序的MCS列表"""
        self._sorted_mcs = sorted(
            [(self.thresholds[e.index], e.index) for e in self.mcs_table],
            key=lambda x: x[0]
        )
    
    @abstractmethod
    def select_mcs(self, snr_db: float) -> int:
        """
        根据SNR选择MCS
        
        Args:
            snr_db: 当前SNR (dB)
            
        Returns:
            选择的MCS索引
        """
        pass
    
    def get_thresholds(self) -> Dict[int, float]:
        """获取所有MCS的切换门限"""
        return dict(self.thresholds)

    def get_threshold_results(self):
        """获取包含调制方式/BLER等元数据的完整门限结果"""
        return list(self.threshold_results)
    
    def get_spectral_efficiency(self, mcs_index: int) -> float:
        """获取MCS的频谱效率"""
        entry = self.mcs_table.get_entry(mcs_index)
        return entry.spectral_efficiency if entry else 0.0
    
    def get_bler(self, snr_db: float, mcs_index: int) -> float:
        """获取BLER"""
        return self.bler_model.calculate_bler(snr_db, mcs_index)
    
    def evaluate(self, snr_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        在SNR范围内评估策略
        
        Args:
            snr_range: SNR数组 (dB)
            
        Returns:
            评估结果字典
        """
        n_points = len(snr_range)
        
        selected_mcs = np.zeros(n_points, dtype=int)
        spectral_efficiency = np.zeros(n_points)
        bler = np.zeros(n_points)
        effective_se = np.zeros(n_points)
        
        for i, snr in enumerate(snr_range):
            mcs = self.select_mcs(snr)
            selected_mcs[i] = mcs
            
            se = self.get_spectral_efficiency(mcs)
            spectral_efficiency[i] = se
            
            b = self.get_bler(snr, mcs)
            bler[i] = b
            
            effective_se[i] = se * (1 - b)
        
        return {
            'snr_db': snr_range,
            'selected_mcs': selected_mcs,
            'spectral_efficiency': spectral_efficiency,
            'bler': bler,
            'effective_spectral_efficiency': effective_se,
            'strategy_name': self.name
        }
    
    def get_switching_points(self) -> List[Tuple[float, int, int]]:
        """
        获取MCS切换点
        
        Returns:
            切换点列表: [(snr, from_mcs, to_mcs), ...]
        """
        if not self._sorted_mcs:
            return []
        
        switching_points = []
        prev_mcs = self._sorted_mcs[0][1]
        
        for snr, mcs in self._sorted_mcs[1:]:
            switching_points.append((snr, prev_mcs, mcs))
            prev_mcs = mcs
        
        return switching_points
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', target_bler={self.target_bler}, margin={self.margin_db}dB)"
