"""
BLER Performance Model
BLER性能曲线模型 - 使用Sigmoid函数逼近AWGN性能
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List
from dataclasses import dataclass


@dataclass
class BLERModelParams:
    """BLER模型参数"""
    snr_threshold: float  # BLER=0.5时的SNR (dB)
    slope: float = 1.5    # 曲线斜率因子


class BLERModel(ABC):
    """BLER模型抽象基类"""
    
    @abstractmethod
    def calculate_bler(self, snr_db: Union[float, np.ndarray], mcs_index: int) -> Union[float, np.ndarray]:
        """
        计算给定SNR下的BLER
        
        Args:
            snr_db: SNR值 (dB)，可以是标量或数组
            mcs_index: MCS索引
            
        Returns:
            BLER值，范围[0, 1]
        """
        pass
    
    @abstractmethod
    def find_snr_for_bler(self, target_bler: float, mcs_index: int) -> float:
        """
        查找达到目标BLER的SNR值
        
        Args:
            target_bler: 目标BLER值
            mcs_index: MCS索引
            
        Returns:
            SNR值 (dB)
        """
        pass


class AWGNBLERModel(BLERModel):
    """
    AWGN信道BLER模型
    
    使用Sigmoid函数逼近:
    BLER(SNR) = 1 / (1 + exp(slope * (SNR - SNR_threshold)))
    
    其中SNR_threshold是BLER=0.5的点，slope控制曲线陡峭程度
    """
    
    def __init__(self, mcs_params: dict = None):
        """
        初始化BLER模型
        
        Args:
            mcs_params: MCS索引到模型参数的映射
                        格式: {mcs_index: BLERModelParams}
        """
        self.mcs_params = mcs_params or {}
        
        # 默认参数：基于MCS索引估算
        self._default_snr_thresholds = self._generate_default_thresholds()
    
    def _generate_default_thresholds(self) -> dict:
        """生成默认的SNR门限（基于经验公式）"""
        thresholds = {}
        for mcs in range(32):
            # 简化经验公式：每个MCS大约增加1dB
            thresholds[mcs] = -5 + mcs * 1.0
        return thresholds
    
    def set_mcs_params(self, mcs_index: int, snr_threshold: float, slope: float = 1.5):
        """
        设置特定MCS的模型参数
        
        Args:
            mcs_index: MCS索引
            snr_threshold: BLER=0.5时的SNR门限
            slope: 斜率因子（值越大，曲线越陡峭）
        """
        self.mcs_params[mcs_index] = BLERModelParams(snr_threshold, slope)
    
    def set_params_from_mcs_table(self, mcs_table, target_bler: float = 0.1, slope: float = 1.5):
        """
        从MCS表设置参数
        
        Args:
            mcs_table: MCS表对象
            target_bler: MCS表中SNR门限对应的BLER值
            slope: 斜率因子
        """
        for entry in mcs_table:
            # 将目标BLER对应的SNR转换为BLER=0.5的SNR
            # BLER = 1/(1+exp(slope*(SNR-SNR_50)))
            # 解出 SNR_50 = SNR - ln((1-BLER)/BLER) / slope
            offset = np.log((1 - target_bler) / target_bler) / slope
            snr_50 = entry.snr_threshold - offset
            self.mcs_params[entry.index] = BLERModelParams(snr_50, slope)
    
    def _get_params(self, mcs_index: int) -> BLERModelParams:
        """获取MCS参数"""
        if mcs_index in self.mcs_params:
            return self.mcs_params[mcs_index]
        # 使用默认值
        return BLERModelParams(
            self._default_snr_thresholds.get(mcs_index, 0),
            slope=1.5
        )
    
    def calculate_bler(self, snr_db: Union[float, np.ndarray], mcs_index: int) -> Union[float, np.ndarray]:
        """计算BLER"""
        params = self._get_params(mcs_index)
        
        # Sigmoid函数
        exponent = params.slope * (snr_db - params.snr_threshold)
        
        # 数值稳定性处理
        exponent = np.clip(exponent, -50, 50)
        
        bler = 1.0 / (1.0 + np.exp(exponent))
        
        # 限制范围
        return np.clip(bler, 1e-6, 1.0 - 1e-6)
    
    def find_snr_for_bler(self, target_bler: float, mcs_index: int) -> float:
        """查找达到目标BLER的SNR"""
        params = self._get_params(mcs_index)
        
        # 从Sigmoid公式反解SNR:
        # target_bler = 1/(1+exp(slope*(SNR-SNR_th)))
        # SNR = SNR_th - ln((1-target_bler)/target_bler) / slope
        
        if target_bler <= 0 or target_bler >= 1:
            raise ValueError("目标BLER必须在(0, 1)范围内")
        
        offset = np.log((1 - target_bler) / target_bler) / params.slope
        return params.snr_threshold - offset
    
    def calculate_bler_all_mcs(self, snr_db: Union[float, np.ndarray], 
                                mcs_indices: List[int]) -> np.ndarray:
        """
        计算所有MCS的BLER
        
        Args:
            snr_db: SNR数组
            mcs_indices: MCS索引列表
            
        Returns:
            BLER矩阵，shape=(len(mcs_indices), len(snr_db))
        """
        snr_db = np.atleast_1d(snr_db)
        bler_matrix = np.zeros((len(mcs_indices), len(snr_db)))
        
        for i, mcs in enumerate(mcs_indices):
            bler_matrix[i, :] = self.calculate_bler(snr_db, mcs)
        
        return bler_matrix


class LookupTableBLERModel(BLERModel):
    """
    查表BLER模型 - 使用预计算的BLER曲线并进行插值
    """
    
    def __init__(self):
        self.lookup_tables = {}  # {mcs_index: (snr_array, bler_array)}
    
    def load_table(self, mcs_index: int, snr_array: np.ndarray, bler_array: np.ndarray):
        """加载查找表"""
        self.lookup_tables[mcs_index] = (np.array(snr_array), np.array(bler_array))
    
    def calculate_bler(self, snr_db: Union[float, np.ndarray], mcs_index: int) -> Union[float, np.ndarray]:
        """通过插值计算BLER"""
        if mcs_index not in self.lookup_tables:
            raise ValueError(f"未找到MCS {mcs_index}的查找表")
        
        snr_table, bler_table = self.lookup_tables[mcs_index]
        return np.interp(snr_db, snr_table, bler_table)
    
    def find_snr_for_bler(self, target_bler: float, mcs_index: int) -> float:
        """通过插值查找SNR"""
        if mcs_index not in self.lookup_tables:
            raise ValueError(f"未找到MCS {mcs_index}的查找表")
        
        snr_table, bler_table = self.lookup_tables[mcs_index]
        # 反向插值（BLER是递减的）
        return np.interp(target_bler, bler_table[::-1], snr_table[::-1])
