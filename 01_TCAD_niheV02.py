import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ====== 高级参数配置 ======
class Config:
    # 文件设置
    
    file_path = r"C:\Users\Zuspi\Desktop\Source\TCAD_trant\file"  # 替换为您的Windows路径
    file_name = "01_00_simrawdata_buquan.xlsx"      # 替换为您的文件名
    
    # 分段参数
    D1 = 500.0   # 第一段结束深度 (nm)
    D2 = 1500.0  # 第二段结束深度 (nm)
    D3 = 3000.0  # 第三段结束深度 (nm)
    step1 = 50.0  # 第一段步长 (nm)
    step2 = 100.0 # 第二段步长 (nm)
    step3 = 200.0 # 第三段步长 (nm)
    
    # 数据处理参数
    detection_limit = 1e12  # 检测限 (atoms/cm³)
    min_valid_depth = 10.0  # 最小有效深度 (nm)
    max_depth = 3000.0      # 最大处理深度 (nm)
    
    # 平滑参数
    min_smooth_window = 11  # 最小平滑窗口
    max_smooth_window = 101 # 最大平滑窗口
    smooth_order = 3        # 平滑多项式阶数
    
    # 拟合参数
    fit_samples = 15       # 每段采样点数
    min_fit_points = 5     # 最小拟合点数
    max_poly_order = 3     # 最大多项式阶数
    min_segment_length = 5.0  # 最小分段长度(nm)
    
    # 输出控制
    save_plots = True      # 是否保存拟合效果图
    plot_path = os.path.join(file_path, "拟合效果图")  # 效果图保存路径

# ====== 高级平滑与拟合函数 ======
def adaptive_smoothing(depth, conc, min_window=11, max_window=101, order=3):
    """自适应平滑函数，避免边缘效应"""
    # 计算平均步长
    diff = np.diff(depth)
    valid_diff = diff[diff > 0]
    avg_step = np.mean(valid_diff) if len(valid_diff) > 0 else 1.0
    
    # 根据数据密度计算窗口大小
    window_size = int(np.clip(20 / avg_step, min_window, max_window))
    window_size = window_size + 1 if window_size % 2 == 0 else window_size
    
    try:
        # 执行平滑
        smoothed = savgol_filter(conc, window_length=window_size, polyorder=order)
        # 避免负值
        smoothed = np.maximum(smoothed, 0)
        return smoothed, window_size
    except:
        # 平滑失败时返回原始数据
        return conc, 0

def piecewise_fit(x, y, max_order=3):
    """分段拟合函数，自动选择最佳多项式阶数"""
    best_order = 1
    best_r2 = -np.inf
    best_coeffs = None
    
    # 确保有足够点
    valid_idx = ~np.isnan(y) & ~np.isinf(y) & (y > 0)
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    
    if len(y_valid) < 2:
        # 点不足时使用最后一个有效值
        return np.array([np.mean(y_valid) if len(y_valid) > 0 else 0]), 0, 0
    
    # 尝试不同阶数
    for order in range(1, max_order + 1):
        if len(y_valid) <= order:
            continue
            
        try:
            coeffs = np.polyfit(x_valid, y_valid, deg=order)
            y_pred = np.polyval(coeffs, x_valid)
            
            # 避免负值预测
            y_pred = np.maximum(y_pred, 0)
            
            r2 = r2_score(y_valid, y_pred)
            
            # 选择R²最高的有效拟合
            if r2 > best_r2 and not np.isnan(r2):
                best_r2 = r2
                best_order = order
                best_coeffs = coeffs
        except:
            continue
    
    # 没有有效拟合时使用线性插值
    if best_coeffs is None:
        best_coeffs = np.polyfit(x_valid, y_valid, deg=1)
        best_order = 1
        best_r2 = r2_score(y_valid, np.polyval(best_coeffs, x_valid))
    
    return best_coeffs, best_order, best_r2

# ====== 主处理脚本 ======
def process_sims_data():
    cfg = Config()
    
    # 构建完整文件路径
    full_path = os.path.join(cfg.file_path, cfg.file_name)
    base_name = os.path.splitext(cfg.file_name)[0]
    
    # 创建输出目录
    os.makedirs(cfg.plot_path, exist_ok=True)
    
    # 输出文件路径
    smooth_output = os.path.join(cfg.file_path, f"03_{base_name}_pinghua.xlsx")
    segment_output = os.path.join(cfg.file_path, f"04_{base_name}_fengenihe.xlsx")
    
    # 读取Excel文件
    xls = pd.ExcelFile(full_path)
    elements = xls.sheet_names
    
    # ===== 第一步：高级平滑处理 =====
    print("开始高级平滑处理...")
    smooth_results = {}
    
    with pd.ExcelWriter(smooth_output) as writer:
        for elem in elements:
            # 读取原始数据
            df = pd.read_excel(xls, sheet_name=elem)
            depth = df['DEPTH (nm)'].values
            conc = df[elem].values
            
            # 过滤无效深度
            valid_idx = (depth >= 0) & (depth <= cfg.max_depth)
            depth = depth[valid_idx]
            conc = conc[valid_idx]
            
            # 创建0-max_depth的均匀网格
            grid_depth = np.arange(0, cfg.max_depth + 1, 1)
            
            # 使用Akima插值（避免过冲）
            try:
                akima = Akima1DInterpolator(depth, conc)
                grid_conc = akima(grid_depth)
            except:
                # Akima失败时使用线性插值
                grid_conc = np.interp(grid_depth, depth, conc, left=0, right=0)
            
            # 将检测限以下的值设为检测限
            grid_conc[grid_conc < cfg.detection_limit] = cfg.detection_limit
            
            # 自适应平滑（只处理有效深度范围）
            valid_smooth_idx = (grid_depth >= cfg.min_valid_depth) & (grid_depth <= (depth.max() if len(depth) > 0 else 0))
            if np.sum(valid_smooth_idx) > cfg.min_smooth_window:
                conc_to_smooth = grid_conc.copy()
                # 对有效范围进行平滑
                smoothed_part, win_size = adaptive_smoothing(
                    grid_depth[valid_smooth_idx],
                    grid_conc[valid_smooth_idx],
                    min_window=cfg.min_smooth_window,
                    max_window=cfg.max_smooth_window,
                    order=cfg.smooth_order
                )
                conc_to_smooth[valid_smooth_idx] = smoothed_part
                smoothed = conc_to_smooth
                print(f"元素 {elem}: 使用自适应窗口大小 {win_size} 完成平滑")
            else:
                smoothed = grid_conc
                print(f"元素 {elem}: 数据不足，跳过平滑")
            
            # 确保非负并处理非有限值
            smoothed = np.maximum(smoothed, cfg.detection_limit)
            smoothed = np.nan_to_num(smoothed, nan=cfg.detection_limit, posinf=np.max(smoothed), neginf=cfg.detection_limit)
            
            # 存储平滑结果
            smooth_df = pd.DataFrame({
                'DEPTH (nm)': grid_depth,
                elem: smoothed
            })
            smooth_df.to_excel(writer, sheet_name=elem, index=False)
            
            # 保存插值器
            smooth_results[elem] = {
                'depth': grid_depth,
                'conc': smoothed,
                'interpolator': lambda x: np.interp(x, grid_depth, smoothed, left=cfg.detection_limit, right=cfg.detection_limit)
            }
    
    print(f"平滑数据已保存至: {smooth_output}")
    
    # ===== 第二步：高级分段拟合 =====
    print("开始高级分段拟合...")
    
    # 生成分段边界
    segments = []
    
    # 第一段 [0, D1]
    start = 0.0
    while start < cfg.D1:
        end = min(start + cfg.step1, cfg.D1)
        if (end - start) >= cfg.min_segment_length:
            segments.append((start, end))
        start = end
    
    # 第二段 [D1, D2]
    start = cfg.D1
    while start < cfg.D2:
        end = min(start + cfg.step2, cfg.D2)
        if (end - start) >= cfg.min_segment_length:
            segments.append((start, end))
        start = end
    
    # 第三段 [D2, D3]
    start = cfg.D2
    while start < cfg.D3:
        end = min(start + cfg.step3, cfg.D3)
        if (end - start) >= cfg.min_segment_length:
            segments.append((start, end))
        start = end
    
    # 准备分段结果存储
    results = []
    fit_quality = {elem: [] for elem in elements}  # 记录拟合质量
    
    # 对每个元素处理
    for elem in elements:
        interp_func = smooth_results[elem]['interpolator']
        
        for seg_idx, (abs_min, abs_max) in enumerate(segments, 1):
            seg_length = abs_max - abs_min
            
            # 跳过无效分段
            if seg_length <= 0:
                continue
                
            # 在小分割内采样
            rel_depth = np.linspace(0, seg_length, num=cfg.fit_samples)
            abs_depth_points = abs_min + rel_depth
            conc_points = interp_func(abs_depth_points)
            
            # 高级分段拟合
            coeffs, order, r2 = piecewise_fit(
                rel_depth, conc_points,
                max_order=cfg.max_poly_order
            )
            
            # 构建方程字符串
            equation = f"y = "
            for i, c in enumerate(coeffs):
                power = len(coeffs) - i - 1
                if power > 1:
                    equation += f"{c:.6e}·x^{power} + "
                elif power == 1:
                    equation += f"{c:.6e}·x + "
                else:
                    equation += f"{c:.6e}"
            
            # 添加到结果
            results.append({
                '元素名': elem,
                '小分割区n': seg_idx,
                '绝对深度min': abs_min,
                '绝对深度max': abs_max,
                '小分割深度min': 0.0,
                '小分割深度max': seg_length,
                '拟合方程f(X)': equation,
                '多项式阶数': order,
                'R²拟合优度': r2
            })
            
            # 记录拟合质量
            fit_quality[elem].append(r2)
            
            # 可视化拟合效果
            if cfg.save_plots and (seg_idx <= 5 or seg_idx % 20 == 0):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 绘制原始数据点
                ax.scatter(rel_depth, conc_points, color='blue', label='采样点')
                
                # 绘制拟合曲线
                fit_depth = np.linspace(0, seg_length, 100)
                fit_conc = np.polyval(coeffs, fit_depth)
                ax.plot(fit_depth, fit_conc, 'r-', linewidth=2, label='拟合曲线')
                
                # 设置图表属性
                ax.set_title(f"{elem} - 分段 {seg_idx}\n深度范围: {abs_min:.1f}-{abs_max:.1f} nm")
                ax.set_xlabel("相对深度 (nm)")
                ax.set_ylabel("浓度 (atoms/cm³)")
                ax.legend()
                ax.grid(True)
                ax.set_yscale('log')
                
                # 保存图表
                plot_file = os.path.join(cfg.plot_path, f"{elem}_segment_{seg_idx}_fit.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
    
    # 保存分段结果
    result_df = pd.DataFrame(results)
    result_df.to_excel(segment_output, index=False)
    
    # 打印拟合质量报告
    print("\n拟合质量报告:")
    for elem in elements:
        r2_values = [x for x in fit_quality[elem] if not np.isnan(x)]
        if r2_values:
            avg_r2 = np.mean(r2_values)
            min_r2 = np.min(r2_values)
            print(f"{elem}: 平均R² = {avg_r2:.4f}, 最小R² = {min_r2:.4f} (基于{len(r2_values)}个分段)")
        else:
            print(f"{elem}: 无有效拟合质量数据")
    
    print(f"分段拟合结果已保存至: {segment_output}")
    print("处理完成！")

if __name__ == "__main__":
    process_sims_data()
