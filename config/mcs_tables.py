"""
MCS Tables for 4G LTE and 5G NR
MCS表定义模块 - 支持4G/5G标准表和自定义表
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class MCSEntry:
    """MCS条目定义"""
    index: int              # MCS索引
    modulation: str         # 调制方式: QPSK, 16QAM, 64QAM, 256QAM
    modulation_order: int   # 调制阶数: 2, 4, 6, 8
    code_rate: float        # 码率
    spectral_efficiency: float  # 频谱效率 (bits/s/Hz)
    snr_threshold: float    # BLER=10%时的SNR门限 (dB) - 初始估计值
    
    def __post_init__(self):
        """计算频谱效率（如果未提供）"""
        if self.spectral_efficiency == 0:
            self.spectral_efficiency = self.modulation_order * self.code_rate


class MCSTable:
    """MCS表管理类"""
    
    def __init__(self, name: str, entries: List[MCSEntry]):
        self.name = name
        self.entries = sorted(entries, key=lambda x: x.index)
        self._build_lookup()
    
    def _build_lookup(self):
        """构建索引查找表"""
        self.index_map = {e.index: e for e in self.entries}
    
    def get_entry(self, index: int) -> Optional[MCSEntry]:
        """根据索引获取MCS条目"""
        return self.index_map.get(index)
    
    def get_all_entries(self) -> List[MCSEntry]:
        """获取所有MCS条目"""
        return self.entries
    
    def get_spectral_efficiencies(self) -> np.ndarray:
        """获取所有频谱效率"""
        return np.array([e.spectral_efficiency for e in self.entries])
    
    def get_snr_thresholds(self) -> np.ndarray:
        """获取所有SNR门限"""
        return np.array([e.snr_threshold for e in self.entries])
    
    def __len__(self):
        return len(self.entries)
    
    def __iter__(self):
        return iter(self.entries)


# ============================================================
# 4G LTE MCS Table (based on 3GPP TS 36.213)
# ============================================================

LTE_MCS_TABLE = MCSTable(
    name="4G LTE MCS Table",
    entries=[
        MCSEntry(0,  "QPSK",   2, 0.1172, 0.2344,  -6.7),
        MCSEntry(1,  "QPSK",   2, 0.1533, 0.3066,  -5.5),
        MCSEntry(2,  "QPSK",   2, 0.1885, 0.3770,  -4.4),
        MCSEntry(3,  "QPSK",   2, 0.2451, 0.4902,  -3.2),
        MCSEntry(4,  "QPSK",   2, 0.3008, 0.6016,  -2.0),
        MCSEntry(5,  "QPSK",   2, 0.3701, 0.7402,  -0.7),
        MCSEntry(6,  "QPSK",   2, 0.4385, 0.8770,   0.5),
        MCSEntry(7,  "QPSK",   2, 0.5137, 1.0273,   1.7),
        MCSEntry(8,  "QPSK",   2, 0.5879, 1.1758,   2.9),
        MCSEntry(9,  "QPSK",   2, 0.6631, 1.3262,   4.1),
        MCSEntry(10, "16QAM", 4, 0.3320, 1.3281,   4.3),
        MCSEntry(11, "16QAM", 4, 0.3691, 1.4766,   5.3),
        MCSEntry(12, "16QAM", 4, 0.4238, 1.6953,   6.5),
        MCSEntry(13, "16QAM", 4, 0.4785, 1.9141,   7.7),
        MCSEntry(14, "16QAM", 4, 0.5400, 2.1602,   9.0),
        MCSEntry(15, "16QAM", 4, 0.6016, 2.4063,  10.3),
        MCSEntry(16, "16QAM", 4, 0.6426, 2.5703,  11.2),
        MCSEntry(17, "64QAM", 6, 0.4277, 2.5664,  11.3),
        MCSEntry(18, "64QAM", 6, 0.4551, 2.7305,  12.0),
        MCSEntry(19, "64QAM", 6, 0.5049, 3.0293,  13.3),
        MCSEntry(20, "64QAM", 6, 0.5537, 3.3223,  14.6),
        MCSEntry(21, "64QAM", 6, 0.6016, 3.6094,  15.8),
        MCSEntry(22, "64QAM", 6, 0.6504, 3.9023,  17.1),
        MCSEntry(23, "64QAM", 6, 0.7021, 4.2129,  18.5),
        MCSEntry(24, "64QAM", 6, 0.7539, 4.5234,  19.8),
        MCSEntry(25, "64QAM", 6, 0.8008, 4.8047,  21.0),
        MCSEntry(26, "64QAM", 6, 0.8525, 5.1152,  22.3),
        MCSEntry(27, "64QAM", 6, 0.8887, 5.3320,  23.3),
        MCSEntry(28, "64QAM", 6, 0.9258, 5.5547,  24.3),
    ]
)


# ============================================================
# 5G NR MCS Table 1 (based on 3GPP TS 38.214, Table 5.1.3.1-1)
# ============================================================

NR_MCS_TABLE_1 = MCSTable(
    name="5G NR MCS Table 1 (64QAM)",
    entries=[
        MCSEntry(0,  "QPSK",   2, 0.1172, 0.2344,  -6.7),
        MCSEntry(1,  "QPSK",   2, 0.1885, 0.3770,  -4.7),
        MCSEntry(2,  "QPSK",   2, 0.3008, 0.6016,  -2.3),
        MCSEntry(3,  "QPSK",   2, 0.4385, 0.8770,   0.2),
        MCSEntry(4,  "QPSK",   2, 0.5879, 1.1758,   2.5),
        MCSEntry(5,  "16QAM", 4, 0.3691, 1.4766,   5.0),
        MCSEntry(6,  "16QAM", 4, 0.4785, 1.9141,   7.4),
        MCSEntry(7,  "16QAM", 4, 0.6016, 2.4063,  10.0),
        MCSEntry(8,  "64QAM", 6, 0.4551, 2.7305,  11.8),
        MCSEntry(9,  "64QAM", 6, 0.5537, 3.3223,  14.3),
        MCSEntry(10, "64QAM", 6, 0.6504, 3.9023,  16.8),
        MCSEntry(11, "64QAM", 6, 0.7539, 4.5234,  19.5),
        MCSEntry(12, "64QAM", 6, 0.8525, 5.1152,  22.0),
        MCSEntry(13, "64QAM", 6, 0.9258, 5.5547,  24.0),
    ]
)


# ============================================================
# 5G NR MCS Table 2 (256QAM, based on 3GPP TS 38.214)
# ============================================================

NR_MCS_TABLE_2 = MCSTable(
    name="5G NR MCS Table 2 (256QAM)",
    entries=[
        MCSEntry(0,  "QPSK",    2, 0.1172, 0.2344,  -6.7),
        MCSEntry(1,  "QPSK",    2, 0.2451, 0.4902,  -3.5),
        MCSEntry(2,  "QPSK",    2, 0.3770, 0.7539,  -0.8),
        MCSEntry(3,  "16QAM",   4, 0.2549, 1.0195,   1.5),
        MCSEntry(4,  "16QAM",   4, 0.3770, 1.5078,   5.2),
        MCSEntry(5,  "16QAM",   4, 0.5137, 2.0547,   8.3),
        MCSEntry(6,  "64QAM",   6, 0.3770, 2.2617,   9.5),
        MCSEntry(7,  "64QAM",   6, 0.4902, 2.9414,  12.8),
        MCSEntry(8,  "64QAM",   6, 0.6162, 3.6973,  16.0),
        MCSEntry(9,  "256QAM",  8, 0.5000, 4.0000,  17.5),
        MCSEntry(10, "256QAM",  8, 0.5527, 4.4219,  19.2),
        MCSEntry(11, "256QAM",  8, 0.6250, 5.0000,  21.5),
        MCSEntry(12, "256QAM",  8, 0.7109, 5.6875,  24.0),
        MCSEntry(13, "256QAM",  8, 0.7773, 6.2188,  26.0),
        MCSEntry(14, "256QAM",  8, 0.8477, 6.7813,  28.0),
        MCSEntry(15, "256QAM",  8, 0.9258, 7.4063,  30.5),
    ]
)


# ============================================================
# 可用MCS表字典
# ============================================================

AVAILABLE_MCS_TABLES: Dict[str, MCSTable] = {
    "LTE": LTE_MCS_TABLE,
    "NR_Table1": NR_MCS_TABLE_1,
    "NR_Table2": NR_MCS_TABLE_2,
}


def get_mcs_table(name: str) -> MCSTable:
    """获取指定名称的MCS表"""
    if name not in AVAILABLE_MCS_TABLES:
        raise ValueError(f"未知的MCS表: {name}. 可用: {list(AVAILABLE_MCS_TABLES.keys())}")
    return AVAILABLE_MCS_TABLES[name]


def create_custom_mcs_table(name: str, entries_data: List[Dict]) -> MCSTable:
    """
    创建自定义MCS表
    
    Args:
        name: 表名称
        entries_data: MCS条目数据列表，每个条目包含:
            - index: MCS索引
            - modulation: 调制方式
            - modulation_order: 调制阶数
            - code_rate: 码率
            - snr_threshold: SNR门限（可选）
    
    Returns:
        MCSTable: 自定义MCS表
    """
    entries = []
    for data in entries_data:
        mod_order = data["modulation_order"]
        code_rate = data["code_rate"]
        se = mod_order * code_rate
        
        entries.append(MCSEntry(
            index=data["index"],
            modulation=data["modulation"],
            modulation_order=mod_order,
            code_rate=code_rate,
            spectral_efficiency=se,
            snr_threshold=data.get("snr_threshold", 0)
        ))
    
    return MCSTable(name, entries)
