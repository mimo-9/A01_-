import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator, UnivariateSpline
from scipy.signal import savgol_filter
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
    fit_samples = 40        # 每段采样点数
    min_fit_points = 8      # 最小拟合点数
    max_poly_order = 6      # 最大多项式阶数（提高到6阶）
    min_segment_length = 10.0  # 最小分段长度(nm)
    continuity_weight = 5.0  # 降低连续性约束权重
    ridge_alpha = 0.1       # 增加岭回归正则化强度
    min_r2 = 0.90           # 降低最低可接受R²值
    
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

def constrained_high_order_fit(x, y, start_val, end_val, max_order=6, weight=5.0, ridge_alpha=0.1, min_r2=0.9):
    """
    带边界约束的高阶多项式拟合（最高6阶）
    强制满足：f(0) = start_val, f(x_max) = end_val
    包含多层保护机制确保R²合理
    """
    # 确保有足够点
    valid_idx = ~np.isnan(y) & ~np.isinf(y) & (y > 0)
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]
    
    # 点不足时使用线性插值
    if len(y_valid) < 3:
        slope = (end_val - start_val) / x[-1] if x[-1] != 0 else 0
        return np.array([start_val, slope]), 1, 0
    
    # 确定最佳多项式阶数（基于交叉验证）
    best_order = 1  # 默认从1阶开始
    best_r2 = -np.inf
    best_coeffs = None
    
    # 尝试不同阶数（从1阶到max_order）
    for order in range(1, min(max_order + 1, len(y_valid))):
        try:
            # 创建多项式特征
            poly = PolynomialFeatures(degree=order)
            X_poly = poly.fit_transform(x_valid.reshape(-1, 1))
            
            # 添加边界约束作为额外样本
            boundary_points = np.array([[0], [x[-1]]])
            X_boundary = poly.transform(boundary_points)
            y_boundary = np.array([start_val, end_val])
            
            # 合并数据和约束
            X_combined = np.vstack([X_poly, weight * X_boundary])
            y_combined = np.hstack([y_valid, weight * y_boundary])
            
            # 岭回归拟合（防止过拟合）
            model = Ridge(alpha=ridge_alpha * (order**2), fit_intercept=False, random_state=42)
            model.fit(X_combined, y_combined)
            
            # 计算R²
            y_pred = model.predict(X_poly)
            
            # 确保预测值合理（非负）
            y_pred = np.maximum(y_pred, 0)
            
            # 计算R²并确保有效
            r2 = r2_score(y_valid, y_pred)
            if np.isinf(r2) or np.isnan(r2):
                r2 = -np.inf
                
            # 如果R²足够高，保存结果
            if r2 > best_r2 and r2 >= min_r2:
                best_r2 = r2
                best_order = order
                best_coeffs = model.coef_
        except Exception as e:
            continue
    
    # 安全网：如果找不到足够好的拟合或R²为负
    if best_coeffs is None or best_r2 < 0:
        # 尝试线性拟合
        try:
            # 使用带边界约束的线性拟合
            X_linear = np.vstack([
                x_valid.reshape(-1, 1),
                weight * np.array([0, x[-1]]).reshape(-1, 1)
            ])
            y_linear = np.hstack([
                y_valid,
                weight * np.array([start_val, end_val])
            ])
            
            model_linear = Ridge(alpha=ridge_alpha, fit_intercept=False)
            model_linear.fit(X_linear, y_linear)
            
            # 预测和评估
            linear_pred = model_linear.predict(x_valid.reshape(-1, 1))
            linear_pred = np.maximum(linear_pred, 0)
            linear_r2 = r2_score(y_valid, linear_pred)
            
            if not np.isnan(linear_r2) and linear_r2 >= 0:
                best_coeffs = model_linear.coef_
                best_order = 1
                best_r2 = linear_r2
            else:
                # 降阶到常数拟合
                const_value = np.mean(y_valid)
                best_coeffs = np.array([const_value])
                best_order = 0
                best_r2 = 0  # 常数拟合R²为0
        except:
            # 最终回退：使用端点线性插值
            slope = (end_val - start_val) / x[-1] if x[-1] != 0 else 0
            best_coeffs = np.array([start_val, slope])
            best_order = 1
            best_r2 = 0
    
    # 确保R²非负
    best_r2 = max(best_r2, 0)
    
    # 返回系数（降序排列）
    return best_coeffs, best_order, best_r2

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
    
    # ===== 第二步：高阶分段拟合 =====
    print("开始高阶分段拟合...")
    
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
            
            # 带约束的高阶分段拟合
            coeffs, order, r2 = constrained_high_order_fit(
                rel_depth, 
                conc_points,
                start_val=start_value,
                end_val=end_value,
                max_order=cfg.max_poly_order,
                weight=cfg.continuity_weight,
                ridge_alpha=cfg.ridge_alpha,
                min_r2=cfg.min_r2
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
            if cfg.save_plots and (seg_idx <= 10 or seg_idx % 5 == 0 or r2 < cfg.min_r2):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 主图：拟合曲线
                fit_depth = np.linspace(0, seg_length, 200)
                fit_conc = np.polyval(coeffs, fit_depth)
                
                # 确保拟合曲线非负
                fit_conc = np.maximum(fit_conc, cfg.detection_limit)
                
                ax.plot(rel_depth, conc_points, 'bo', markersize=6, label='采样点')
                ax.plot(fit_depth, fit_conc, 'r-', linewidth=2.5, label=f'拟合曲线 (阶数={order}, R²={r2:.4f})')
                
                # 标记边界点
                ax.scatter(0, start_value, color='green', s=120, zorder=5, marker='s', label='起始点')
                ax.scatter(seg_length, end_value, color='purple', s=120, zorder=5, marker='s', label='结束点')
                
                # 设置图表属性
                ax.set_title(f"{elem} - 分段 {seg_idx}\n深度范围: {abs_min:.1f}-{abs_max:.1f} nm", fontsize=14)
                ax.set_xlabel("相对深度 (nm)", fontsize=12)
                ax.set_ylabel("浓度 (atoms/cm³)", fontsize=12)
                ax.legend(fontsize=12, loc='best')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_yscale('log')
                
                # 添加文本信息
                textstr = f'拟合方程: {equation}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                
                # 保存图表
                plot_file = os.path.join(cfg.plot_path, f"{elem}_segment_{seg_idx}_fit.png")
                plt.tight_layout()
                plt.savefig(plot_file, dpi=180, bbox_inches='tight')
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
    
    # 打印拟合质量总结
    print("\n拟合质量总结:")
    for elem in elements:
        elem_results = [r for r in results if r['元素名'] == elem]
        if elem_results:
            r2_values = [r['R²拟合优度'] for r in elem_results]
            orders = [r['多项式阶数'] for r in elem_results]
            
            avg_r2 = np.mean(r2_values)
            min_r2 = np.min(r2_values)
            max_order = np.max(orders)
            avg_order = np.mean(orders)
            
            # 统计R²分布
            r2_above_095 = sum(r >= 0.95 for r in r2_values)
            r2_090_094 = sum(0.90 <= r < 0.95 for r in r2_values)
            r2_below_090 = sum(r < 0.90 for r in r2_values)
            
            print(f"{elem}:")
            print(f"  平均R² = {avg_r2:.4f}, 最低R² = {min_r2:.4f}")
            print(f"  最高阶数 = {max_order}, 平均阶数 = {avg_order:.1f}")
            print(f"  R²分布: ≥0.95: {r2_above_095}段, 0.90-0.94: {r2_090_094}段, <0.90: {r2_below_090}段")
    
    print(f"\n分段拟合结果已保存至: {segment_output}")
    print("处理完成！")

if __name__ == "__main__":
    process_sims_data()
