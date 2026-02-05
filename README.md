# AMC Strategy Simulation and Throughput Evaluation

一个用于 **自适应调制编码（AMC）策略仿真** 的 Python 项目，支持 4G/5G 常见 MCS 表，能够在不同 SNR 条件下对多种链路自适应策略进行吞吐与 BLER 对比评估。

## 功能特性

- 支持 LTE / NR 的 MCS 表配置
- 基于 Sigmoid 的 AWGN BLER 近似建模
- MCS 切换门限自动搜索（按目标 BLER）
- 多策略对比：保守 / 激进 / 目标 BLER
- 吞吐量与频谱效率评估
- 可选 HARQ（Chase Combining）简化分析
- CLI 与 GUI 两种运行方式
- 结果导出（CSV）与图表可视化

## 项目结构

```text
AMC/
├── main.py                  # 程序入口（CLI/GUI）
├── config/                  # MCS 表与配置
├── core/                    # BLER、门限搜索、吞吐、HARQ 等核心模型
├── strategies/              # AMC 策略实现
├── simulation/              # 仿真编排与结果汇总
├── visualization/           # 绘图逻辑
└── gui/                     # 图形界面
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

### 1) GUI 模式（默认）

```bash
python main.py
```

### 2) CLI 模式

```bash
python main.py --cli
```

### 常用参数示例

```bash
python main.py --cli \
  --snr-min -5 --snr-max 30 --snr-step 0.5 \
  --mcs-table LTE \
  --target-bler 0.10 \
  --bandwidth 20 \
  --harq --harq-retx 4 --harq-gain 3.0 \
  --output result.csv \
  --plot --save-plots dashboard.png
```

## 主要参数说明

- `--snr-min / --snr-max / --snr-step`：SNR 扫描范围
- `--mcs-table`：MCS 表选择（`LTE` / `NR_Table1` / `NR_Table2`）
- `--target-bler`：目标 BLER（默认 0.10）
- `--bandwidth`：带宽（MHz）
- `--harq`：启用 HARQ 简化模型
- `--output`：导出 CSV 文件
- `--plot` / `--save-plots`：显示或保存图表

## 结果解读建议

- **平均吞吐量**：总体频谱利用能力
- **峰值吞吐量**：高 SNR 场景上限
- **平均 BLER**：可靠性水平
- **平均频谱效率（SE）**：调制编码效率

一般来说：
- 保守策略 BLER 更低，吞吐偏稳健
- 激进策略吞吐更高，但 BLER 偏高
- 目标 BLER 策略在两者之间折中

## 开发与检查

可使用以下命令进行基础检查：

```bash
python -m compileall core strategies simulation main.py
python main.py --cli --snr-min 0 --snr-max 2 --snr-step 1
```

## 许可证

当前仓库未显式声明 LICENSE；如需开源发布，建议补充许可证文件（如 MIT / Apache-2.0）。
