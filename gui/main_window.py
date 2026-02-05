"""
AMC Simulator Main Window
AMC仿真器主窗口 - PyQt6 GUI
"""

import sys
import numpy as np
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox,
        QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QMessageBox,
        QFileDialog, QStatusBar, QProgressBar, QFrame, QGridLayout, QSpinBox,
        QDoubleSpinBox, QSlider
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont, QIcon, QPalette, QColor
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox,
        QTabWidget, QTableWidget, QTableWidgetItem, QSplitter, QMessageBox,
        QFileDialog, QStatusBar, QProgressBar, QFrame, QGridLayout, QSpinBox,
        QDoubleSpinBox, QSlider
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    PYQT_VERSION = 5
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

try:
    from simulation.simulator import AMCSimulator, SimulationConfig
    from visualization.plots import AMCPlotter
    from config.mcs_tables import AVAILABLE_MCS_TABLES
except ImportError:
    from ..simulation.simulator import AMCSimulator, SimulationConfig
    from ..visualization.plots import AMCPlotter
    from ..config.mcs_tables import AVAILABLE_MCS_TABLES


class SimulationWorker(QThread):
    """仿真工作线程"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
    
    def run(self):
        try:
            self.progress.emit(30)
            result = self.simulator.run()
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AMCMainWindow(QMainWindow):
    """AMC仿真器主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.simulator = AMCSimulator()
        self.plotter = AMCPlotter()
        self.result = None
        self.worker = None
        
        self.init_ui()
        self.apply_style()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("AMC策略仿真与吞吐评估软件")
        self.setMinimumSize(1400, 900)
        
        # 中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板（参数设置）
        left_panel = self.create_left_panel()
        left_panel.setMaximumWidth(380)
        left_panel.setMinimumWidth(350)
        
        # 右侧面板（结果展示）
        right_panel = self.create_right_panel()
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 1020])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪")
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self) -> QWidget:
        """创建左侧参数面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # === SNR配置 ===
        snr_group = QGroupBox("📊  SNR配置")
        snr_layout = QGridLayout(snr_group)
        
        snr_layout.addWidget(QLabel("最小SNR (dB):"), 0, 0)
        self.snr_min_spin = QDoubleSpinBox()
        self.snr_min_spin.setRange(-20, 10)
        self.snr_min_spin.setValue(-5)
        self.snr_min_spin.setSingleStep(1)
        snr_layout.addWidget(self.snr_min_spin, 0, 1)
        
        snr_layout.addWidget(QLabel("最大SNR (dB):"), 1, 0)
        self.snr_max_spin = QDoubleSpinBox()
        self.snr_max_spin.setRange(10, 50)
        self.snr_max_spin.setValue(30)
        self.snr_max_spin.setSingleStep(1)
        snr_layout.addWidget(self.snr_max_spin, 1, 1)
        
        snr_layout.addWidget(QLabel("步长 (dB):"), 2, 0)
        self.snr_step_spin = QDoubleSpinBox()
        self.snr_step_spin.setRange(0.1, 5)
        self.snr_step_spin.setValue(0.5)
        self.snr_step_spin.setSingleStep(0.1)
        snr_layout.addWidget(self.snr_step_spin, 2, 1)
        
        layout.addWidget(snr_group)
        
        # === MCS表配置 ===
        mcs_group = QGroupBox("📋  MCS表配置")
        mcs_layout = QGridLayout(mcs_group)
        
        mcs_layout.addWidget(QLabel("MCS表:"), 0, 0)
        self.mcs_table_combo = QComboBox()
        self.mcs_table_combo.addItems(list(AVAILABLE_MCS_TABLES.keys()))
        mcs_layout.addWidget(self.mcs_table_combo, 0, 1)
        
        mcs_layout.addWidget(QLabel("目标BLER (%):"), 1, 0)
        self.target_bler_spin = QDoubleSpinBox()
        self.target_bler_spin.setRange(0.1, 50)
        self.target_bler_spin.setValue(10)
        self.target_bler_spin.setSingleStep(1)
        self.target_bler_spin.setSuffix(" %")
        mcs_layout.addWidget(self.target_bler_spin, 1, 1)
        
        mcs_layout.addWidget(QLabel("带宽 (MHz):"), 2, 0)
        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(1, 100)
        self.bandwidth_spin.setValue(20)
        self.bandwidth_spin.setSingleStep(5)
        mcs_layout.addWidget(self.bandwidth_spin, 2, 1)
        
        layout.addWidget(mcs_group)
        
        # === HARQ配置 ===
        harq_group = QGroupBox("🔄  HARQ配置")
        harq_layout = QGridLayout(harq_group)
        
        self.harq_enable_check = QCheckBox("启用HARQ")
        self.harq_enable_check.setChecked(False)
        self.harq_enable_check.stateChanged.connect(self.on_harq_toggle)
        harq_layout.addWidget(self.harq_enable_check, 0, 0, 1, 2)
        
        harq_layout.addWidget(QLabel("最大重传次数:"), 1, 0)
        self.harq_retx_spin = QSpinBox()
        self.harq_retx_spin.setRange(1, 8)
        self.harq_retx_spin.setValue(4)
        self.harq_retx_spin.setEnabled(False)
        harq_layout.addWidget(self.harq_retx_spin, 1, 1)
        
        harq_layout.addWidget(QLabel("合并增益 (dB):"), 2, 0)
        self.harq_gain_spin = QDoubleSpinBox()
        self.harq_gain_spin.setRange(1, 10)
        self.harq_gain_spin.setValue(3)
        self.harq_gain_spin.setEnabled(False)
        harq_layout.addWidget(self.harq_gain_spin, 2, 1)
        
        layout.addWidget(harq_group)
        
        # === 策略对比 ===
        strategy_group = QGroupBox("⚡  策略选择")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.conservative_check = QCheckBox("保守策略 (BLER=1%, +3dB裕量)")
        self.conservative_check.setChecked(True)
        strategy_layout.addWidget(self.conservative_check)
        
        self.aggressive_check = QCheckBox("激进策略 (BLER=20%)")
        self.aggressive_check.setChecked(True)
        strategy_layout.addWidget(self.aggressive_check)
        
        self.target_bler_check = QCheckBox("目标BLER策略 (自定义)")
        self.target_bler_check.setChecked(True)
        strategy_layout.addWidget(self.target_bler_check)
        
        layout.addWidget(strategy_group)
        
        # === 控制按钮 ===
        btn_layout = QVBoxLayout()
        
        self.run_btn = QPushButton("▶  运行仿真")
        self.run_btn.setMinimumHeight(45)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f6dad;
            }
        """)
        self.run_btn.clicked.connect(self.run_simulation)
        btn_layout.addWidget(self.run_btn)
        
        self.export_btn = QPushButton("💾  导出结果")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        btn_layout.addWidget(self.export_btn)
        
        self.save_plots_btn = QPushButton("🖼  保存图表")
        self.save_plots_btn.setMinimumHeight(40)
        self.save_plots_btn.setEnabled(False)
        self.save_plots_btn.clicked.connect(self.save_plots)
        btn_layout.addWidget(self.save_plots_btn)
        
        layout.addLayout(btn_layout)
        
        # 弹性空间
        layout.addStretch()
        
        # 信息标签
        info_label = QLabel("💡 提示: 点击运行仿真后,\n可在右侧查看结果图表和门限表")
        info_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        info_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        if PYQT_VERSION == 6:
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """创建右侧结果面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标签页
        self.tab_widget = QTabWidget()
        
        # 吞吐量标签页
        self.throughput_tab = QWidget()
        throughput_layout = QVBoxLayout(self.throughput_tab)
        self.throughput_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.throughput_toolbar = NavigationToolbar(self.throughput_canvas, self)
        throughput_layout.addWidget(self.throughput_toolbar)
        throughput_layout.addWidget(self.throughput_canvas)
        self.tab_widget.addTab(self.throughput_tab, "📈 吞吐量")
        
        # 频谱效率标签页
        self.se_tab = QWidget()
        se_layout = QVBoxLayout(self.se_tab)
        self.se_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.se_toolbar = NavigationToolbar(self.se_canvas, self)
        se_layout.addWidget(self.se_toolbar)
        se_layout.addWidget(self.se_canvas)
        self.tab_widget.addTab(self.se_tab, "📊 频谱效率")
        
        # BLER标签页
        self.bler_tab = QWidget()
        bler_layout = QVBoxLayout(self.bler_tab)
        self.bler_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.bler_toolbar = NavigationToolbar(self.bler_canvas, self)
        bler_layout.addWidget(self.bler_toolbar)
        bler_layout.addWidget(self.bler_canvas)
        self.tab_widget.addTab(self.bler_tab, "📉 BLER")
        
        # MCS选择标签页
        self.mcs_tab = QWidget()
        mcs_layout = QVBoxLayout(self.mcs_tab)
        self.mcs_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.mcs_toolbar = NavigationToolbar(self.mcs_canvas, self)
        mcs_layout.addWidget(self.mcs_toolbar)
        mcs_layout.addWidget(self.mcs_canvas)
        self.tab_widget.addTab(self.mcs_tab, "🎯 MCS选择")
        
        # HARQ分析标签页
        self.harq_tab = QWidget()
        harq_layout = QVBoxLayout(self.harq_tab)
        self.harq_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.harq_toolbar = NavigationToolbar(self.harq_canvas, self)
        harq_layout.addWidget(self.harq_toolbar)
        harq_layout.addWidget(self.harq_canvas)
        self.tab_widget.addTab(self.harq_tab, "🔄 HARQ分析")
        
        # 门限表标签页
        self.threshold_tab = QWidget()
        threshold_layout = QVBoxLayout(self.threshold_tab)
        
        # 策略选择下拉框
        threshold_header = QHBoxLayout()
        threshold_header.addWidget(QLabel("选择策略:"))
        self.threshold_strategy_combo = QComboBox()
        self.threshold_strategy_combo.addItems(["target_bler", "conservative", "aggressive"])
        self.threshold_strategy_combo.currentTextChanged.connect(self.update_threshold_table)
        threshold_header.addWidget(self.threshold_strategy_combo)
        threshold_header.addStretch()
        threshold_layout.addLayout(threshold_header)
        
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(5)
        self.threshold_table.setHorizontalHeaderLabels([
            "MCS", "调制方式", "频谱效率", "SNR门限 (dB)", "BLER"
        ])
        self.threshold_table.horizontalHeader().setStretchLastSection(True)
        threshold_layout.addWidget(self.threshold_table)
        self.tab_widget.addTab(self.threshold_tab, "📋 门限表")
        
        # 性能对比标签页
        self.comparison_tab = QWidget()
        comparison_layout = QVBoxLayout(self.comparison_tab)
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(5)
        self.comparison_table.setHorizontalHeaderLabels([
            "策略", "平均吞吐量 (Mbps)", "峰值吞吐量 (Mbps)", "平均BLER", "平均频谱效率"
        ])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        comparison_layout.addWidget(self.comparison_table)
        self.tab_widget.addTab(self.comparison_tab, "⚖ 策略对比")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def apply_style(self):
        """应用样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dcdde1;
                border-radius: 8px;
                margin-top: 12px;
                padding: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: #2c3e50;
            }
            QTabWidget::pane {
                border: 1px solid #dcdde1;
                border-radius: 8px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #ecf0f1;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: white;
                font-weight: bold;
            }
            QPushButton {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 8px 15px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d5dbdb;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
            QTableWidget {
                border: 1px solid #dcdde1;
                border-radius: 4px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
    
    def on_harq_toggle(self, state):
        """HARQ开关切换"""
        # PyQt6使用枚举值, PyQt5使用整数
        if PYQT_VERSION == 6:
            enabled = state == Qt.CheckState.Checked.value
        else:
            enabled = state == Qt.Checked
        
        self.harq_retx_spin.setEnabled(enabled)
        self.harq_gain_spin.setEnabled(enabled)
    
    def run_simulation(self):
        """运行仿真"""
        # 获取配置
        config = SimulationConfig(
            snr_min=self.snr_min_spin.value(),
            snr_max=self.snr_max_spin.value(),
            snr_step=self.snr_step_spin.value(),
            target_bler=self.target_bler_spin.value() / 100,
            bandwidth_mhz=self.bandwidth_spin.value(),
            mcs_table_name=self.mcs_table_combo.currentText(),
            enable_harq=self.harq_enable_check.isChecked(),
            harq_max_retx=self.harq_retx_spin.value(),
            harq_gain_db=self.harq_gain_spin.value()
        )
        
        # 更新仿真器配置
        self.simulator = AMCSimulator(config)
        self.simulator.setup()
        
        # 禁用按钮
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ 仿真中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.statusbar.showMessage("正在运行仿真...")
        
        # 启动工作线程
        self.worker = SimulationWorker(self.simulator)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()
    
    def on_simulation_finished(self, result):
        """仿真完成"""
        self.result = result
        
        # 更新图表
        self.update_plots()
        
        # 更新门限表
        self.update_threshold_table()
        
        # 更新对比表
        self.update_comparison_table()
        
        # 恢复按钮
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  运行仿真")
        self.export_btn.setEnabled(True)
        self.save_plots_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage("仿真完成！")
    
    def on_simulation_error(self, error_msg):
        """仿真错误"""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  运行仿真")
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage("仿真出错")
        QMessageBox.critical(self, "仿真错误", f"仿真过程中发生错误:\n{error_msg}")
    
    def update_plots(self):
        """更新所有图表"""
        if self.result is None:
            return
        
        # 吞吐量图
        self.throughput_canvas.figure.clear()
        ax = self.throughput_canvas.figure.add_subplot(111)
        self.plotter.plot_throughput_comparison(self.result, ax=ax)
        self.throughput_canvas.draw()
        
        # 频谱效率图
        self.se_canvas.figure.clear()
        ax = self.se_canvas.figure.add_subplot(111)
        self.plotter.plot_spectral_efficiency(self.result, ax=ax)
        self.se_canvas.draw()
        
        # BLER图
        self.bler_canvas.figure.clear()
        ax = self.bler_canvas.figure.add_subplot(111)
        self.plotter.plot_bler_comparison(self.result, ax=ax)
        self.bler_canvas.draw()
        
        # MCS选择图
        self.mcs_canvas.figure.clear()
        ax = self.mcs_canvas.figure.add_subplot(111)
        self.plotter.plot_mcs_selection(self.result, ax=ax)
        self.mcs_canvas.draw()
        
        # HARQ分析图
        if self.result.harq_analysis:
            self.harq_canvas.figure.clear()
            axes = self.harq_canvas.figure.subplots(1, 2)
            snr = self.result.harq_analysis['snr_db']
            
            axes[0].semilogy(snr, self.result.harq_analysis['initial_bler'], 'b-', 
                            linewidth=2, label='初始BLER')
            axes[0].semilogy(snr, self.result.harq_analysis['residual_bler'], 'g-', 
                            linewidth=2, label='残余BLER')
            axes[0].set_xlabel('SNR (dB)')
            axes[0].set_ylabel('BLER')
            axes[0].set_title('HARQ BLER改善')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(snr, self.result.harq_analysis['avg_transmissions'], 'r-', linewidth=2)
            axes[1].set_xlabel('SNR (dB)')
            axes[1].set_ylabel('平均传输次数')
            axes[1].set_title('HARQ平均传输次数')
            axes[1].grid(True, alpha=0.3)
            
            self.harq_canvas.figure.tight_layout()
            self.harq_canvas.draw()
    
    def update_threshold_table(self):
        """更新门限表"""
        if self.result is None:
            return
        
        strategy = self.threshold_strategy_combo.currentText()
        table_data = self.simulator.get_threshold_table(strategy)
        
        self.threshold_table.setRowCount(len(table_data))
        
        for row, data in enumerate(table_data):
            self.threshold_table.setItem(row, 0, QTableWidgetItem(str(data['MCS'])))
            self.threshold_table.setItem(row, 1, QTableWidgetItem(data['Modulation']))
            self.threshold_table.setItem(row, 2, QTableWidgetItem(data['SE (bits/s/Hz)']))
            self.threshold_table.setItem(row, 3, QTableWidgetItem(data['SNR Threshold (dB)']))
            self.threshold_table.setItem(row, 4, QTableWidgetItem(data['BLER']))
        
        self.threshold_table.resizeColumnsToContents()
    
    def update_comparison_table(self):
        """更新策略对比表"""
        if self.result is None:
            return
        
        comparison = self.simulator.compare_strategies()
        
        self.comparison_table.setRowCount(len(comparison))
        
        strategy_names = {
            'conservative': '保守策略',
            'aggressive': '激进策略',
            'target_bler': '目标BLER策略'
        }
        
        for row, (name, data) in enumerate(comparison.items()):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(strategy_names.get(name, name)))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(f"{data['avg_throughput_mbps']:.2f}"))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(f"{data['peak_throughput_mbps']:.2f}"))
            self.comparison_table.setItem(row, 3, QTableWidgetItem(f"{data['avg_bler']:.2%}"))
            self.comparison_table.setItem(row, 4, QTableWidgetItem(f"{data['avg_spectral_efficiency']:.3f}"))
        
        self.comparison_table.resizeColumnsToContents()
    
    def export_results(self):
        """导出结果到CSV"""
        if self.result is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "amc_results.csv", "CSV文件 (*.csv)"
        )
        
        if filename:
            self.simulator.export_results(filename)
            QMessageBox.information(self, "导出成功", f"结果已导出到:\n{filename}")
    
    def save_plots(self):
        """保存所有图表"""
        if self.result is None:
            return
        
        folder = QFileDialog.getExistingDirectory(self, "选择保存目录")
        
        if folder:
            self.plotter.plot_all(self.result, save_path=folder)
            QMessageBox.information(self, "保存成功", f"图表已保存到:\n{folder}")


def run_app():
    """运行应用程序"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = AMCMainWindow()
    window.show()
    
    if PYQT_VERSION == 6:
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())


if __name__ == '__main__':
    run_app()
