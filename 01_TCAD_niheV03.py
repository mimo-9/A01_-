import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
from scipy.optimize import least_squares, curve_fit
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

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
    min_smooth_window = 21  # 最小平滑窗口
    max_smooth_window = 101 # 最大平滑窗口
    smooth_order = 3        # 平滑多项式阶数
    smoothing_factor = 0.5  # 平滑因子（0-1，越高越平滑）
    
    # 拟合参数
    fit_samples = 30        # 每段采样点数
    min_fit_points = 8      # 最小拟合点数
    max_poly_order = 4      # 最大多项式阶数
    min_segment_length = 5.0  # 最小分段长度(nm)
    continuity_weight = 10.0 # 连续性约束权重
    ridge_alpha = 0.1       # 岭回归正则化强度
    
    # 输出控制
    save_plots = True       # 是否保存拟合效果图
    plot_path = os.path.join(file_path, "拟合效果图")  # 效果图保存路径

# ====== 高级平滑与拟合函数 ======
def adaptive_smoothing(depth, conc, min_window=21, max_window=101, order=3, s_factor=0.5):
    """自适应平滑函数，使用Spline平滑"""
    try:
        # 使用UnivariateSpline进行自适应平滑
        spline = UnivariateSpline(depth, conc, s=len(depth)*s_factor)
        smoothed = spline(depth)
        return np.maximum(smoothed, 0), len(depth)
    except:
        # 失败时使用Savitzky-Golay
        window_size = min(max(min_window, int(len(depth)*0.3)), max_window)
        window_size = window_size + 1 if window_size % 2 == 0 else window_size
        try:
            smoothed = savgol_filter(conc, window_length=window_size, polyorder=order)
            return np.maximum(smoothed, 0), window_size
        except:
            return conc, 0

def constrained_poly_fit(x, y, start_val, end_val, max_order=4, weight=10.0, ridge_alpha=0.1):
    """
    带边界约束的岭回归多项式拟合
    强制满足：f(0) = start_val, f(x_max) = end_val
    """
    # 确保有足够点
    valid_idx = ~np.isnan(y) & ~np.isinf(y) & (y > 0)
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    
    if len(y_valid) < 2:
        # 点不足时使用线性插值
        return np.array([start_val, (end_val - start_val)/x[-1]]), 1, 0
    
    # 确定最佳多项式阶数（基于数据点数量）
    max_order = min(max_order, len(y_valid) - 1)
    if max_order < 1:
        max_order = 1
    
    # 创建多项式特征
    poly = PolynomialFeatures(degree=max_order)
    X_poly = poly.fit_transform(x_valid.reshape(-1, 1))
    
    # 添加边界约束作为额外样本
    boundary_points = np.array([[0], [x[-1]]])
    X_boundary = poly.transform(boundary_points)
    y_boundary = np.array([start_val, end_val])
    
    # 合并数据和约束
    X_combined = np.vstack([X_poly, weight * X_boundary])
    y_combined = np.hstack([y_valid, weight * y_boundary])
    
    # 岭回归拟合（防止过拟合）
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(X_combined, y_combined)
    coeffs = model.coef_
    
    # 计算R²
    y_pred = model.predict(X_poly)
    r2 = r2_score(y_valid, y_pred) if len(y_valid) > 1 else 1.0
    
    # 返回系数（降序排列）
    return coeffs[::-1], max_order, r2

def generate_equation(coeffs):
    """生成多项式方程字符串"""
    equation = "y = "
    for i, c in enumerate(coeffs):
        power = len(coeffs) - i - 1
        if power > 1:
            equation += f"{c:.6e}·x^{power} + "
        elif power == 1:
            equation += f"{c:.6e}·x + "
        else:
            equation += f"{c:.6e}"
    return equation

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
    global_interpolators = {}
    
    with pd.ExcelWriter(smooth_output) as writer:
        for elem in elements:
            # 读取原始数据
            df = pd.read_excel(xls, sheet_name=elem)
            depth = df['DEPTH (nm)'].values
            conc = df[elem].values
            
            # 过滤无效深度
            valid_idx = (depth >= 0) & (depth <= cfg.max_depth) & (~np.isnan(conc)) & (conc > 0)
            depth = depth[valid_idx]
            conc = conc[valid_idx]
            
            # 创建0-max_depth的均匀网格
            grid_depth = np.arange(0, cfg.max_depth + 1, 1)
            
            # 使用Akima插值
            try:
                akima = Akima1DInterpolator(depth, conc)
                grid_conc = akima(grid_depth)
            except:
                # Akima失败时使用线性插值
                grid_conc = np.interp(grid_depth, depth, conc, left=0, right=0)
            
            # 将检测限以下的值设为检测限
            grid_conc[grid_conc < cfg.detection_limit] = cfg.detection_limit
            
            # 自适应平滑
            smoothed, win_size = adaptive_smoothing(
                grid_depth, 
                grid_conc,
                min_window=cfg.min_smooth_window,
                max_window=cfg.max_smooth_window,
                order=cfg.smooth_order,
                s_factor=cfg.smoothing_factor
            )
            
            # 确保非负并处理非有限值
            smoothed = np.maximum(smoothed, cfg.detection_limit)
            smoothed = np.nan_to_num(smoothed, nan=cfg.detection_limit, 
                                    posinf=np.max(smoothed), neginf=cfg.detection_limit)
            
            # 存储平滑结果
            smooth_df = pd.DataFrame({
                'DEPTH (nm)': grid_depth,
                elem: smoothed
            })
            smooth_df.to_excel(writer, sheet_name=elem, index=False)
            
            # 保存全局插值器
            global_interpolators[elem] = Akima1DInterpolator(grid_depth, smoothed)
            
            print(f"元素 {elem}: 平滑完成")
    
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
    continuity_errors = {elem: [] for elem in elements}
    
    # 对每个元素处理
    for elem in elements:
        interp_func = global_interpolators[elem]
        
        # 存储前一段的结束点浓度（用于连续性检查）
        prev_end_value = None
        
        for seg_idx, (abs_min, abs_max) in enumerate(segments, 1):
            seg_length = abs_max - abs_min
            
            # 跳过无效分段
            if seg_length <= 0:
                continue
                
            # 获取边界点浓度值
            start_value = interp_func(abs_min)
            end_value = interp_func(abs_max)
            
            # 在小分割内密集采样
            rel_depth = np.linspace(0, seg_length, num=cfg.fit_samples)
            abs_depth_points = abs_min + rel_depth
            conc_points = interp_func(abs_depth_points)
            
            # 带约束的高级分段拟合
            coeffs, order, r2 = constrained_poly_fit(
                rel_depth, 
                conc_points,
                start_val=start_value,
                end_val=end_value,
                max_order=cfg.max_poly_order,
                weight=cfg.continuity_weight,
                ridge_alpha=cfg.ridge_alpha
            )
            
            # 检查连续性（与前一段的连接）
            if prev_end_value is not None:
                continuity_error = abs(start_value - prev_end_value)
                continuity_errors[elem].append(continuity_error)
            prev_end_value = end_value
            
            # 生成方程字符串
            equation = generate_equation(coeffs)
            
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
                'R²拟合优度': r2,
                '边界起始浓度': start_value,
                '边界结束浓度': end_value
            })
            
            # 可视化拟合效果
            if cfg.save_plots and (seg_idx <= 10 or seg_idx % 10 == 0 or r2 < 0.9):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
                
                # 主图：拟合曲线
                fit_depth = np.linspace(0, seg_length, 100)
                fit_conc = np.polyval(coeffs, fit_depth)
                
                ax1.plot(rel_depth, conc_points, 'bo', label='采样点')
                ax1.plot(fit_depth, fit_conc, 'r-', linewidth=2, label=f'拟合曲线 (R²={r2:.4f})')
                ax1.axhline(start_value, color='g', linestyle='--', alpha=0.5)
                ax1.axhline(end_value, color='m', linestyle='--', alpha=0.5)
                
                # 标记边界点
                ax1.scatter(0, start_value, color='green', s=100, zorder=5, label='起始点')
                ax1.scatter(seg_length, end_value, color='purple', s=100, zorder=5, label='结束点')
                
                # 设置图表属性
                ax1.set_title(f"{elem} - 分段 {seg_idx}\n深度范围: {abs_min:.1f}-{abs_max:.1f} nm")
                ax1.set_ylabel("浓度 (atoms/cm³)")
                ax1.legend()
                ax1.grid(True)
                ax1.set_yscale('log')
                
                # 残差图
                pred_points = np.polyval(coeffs, rel_depth)
                residuals = pred_points - conc_points
                ax2.plot(rel_depth, residuals, 'g-')
                ax2.fill_between(rel_depth, 0, residuals, alpha=0.3, color='green')
                ax2.axhline(0, color='r', linestyle='--')
                ax2.set_xlabel("相对深度 (nm)")
                ax2.set_ylabel("拟合残差")
                ax2.grid(True)
                
                # 保存图表
                plot_file = os.path.join(cfg.plot_path, f"{elem}_segment_{seg_idx}_fit.png")
                plt.tight_layout()
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
    
    # 保存分段结果到单个Excel文件的不同工作表
    with pd.ExcelWriter(segment_output) as writer:
        for elem in elements:
            # 提取当前元素的所有分段
            elem_results = [r for r in results if r['元素名'] == elem]
            
            # 转换为DataFrame
            if elem_results:
                elem_df = pd.DataFrame(elem_results)
                # 选择需要的列
                elem_df = elem_df[[
                    '小分割区n', '绝对深度min', '绝对深度max', 
                    '小分割深度min', '小分割深度max', 
                    '拟合方程f(X)', '多项式阶数', 'R²拟合优度',
                    '边界起始浓度', '边界结束浓度'
                ]]
                elem_df.to_excel(writer, sheet_name=elem, index=False)
    
    # 打印连续性报告
    print("\n边界连续性报告:")
    for elem in elements:
        if elem in continuity_errors and continuity_errors[elem]:
            max_error = np.max(continuity_errors[elem])
            avg_error = np.mean(continuity_errors[elem])
            print(f"{elem}: 最大边界误差 = {max_error:.2e}, 平均边界误差 = {avg_error:.2e}")
        else:
            print(f"{elem}: 无边界误差数据")
    
    print(f"分段拟合结果已保存至: {segment_output}")
    print("处理完成！")

if __name__ == "__main__":
    process_sims_data()
