import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ====== 用户参数配置区 ======
# 文件路径和名称
file_path = r"C:\Your\Data\Path"  # 替换为您的Windows路径
file_name = "SIMS_data.xlsx"      # 替换为您的文件名

# 分段参数配置
D1 = 500.0   # 第一段结束深度 (nm)
D2 = 1500.0  # 第二段结束深度 (nm)
D3 = 3000.0  # 第三段结束深度 (nm)
step1 = 50.0  # 第一段步长 (nm)
step2 = 100.0 # 第二段步长 (nm)
step3 = 200.0 # 第三段步长 (nm)

# 平滑参数
smooth_window = 51  # 平滑窗口大小（奇数）
smooth_order = 3    # 平滑多项式阶数

# ====== 主处理脚本 ======
def process_sims_data():
    # 构建完整文件路径
    full_path = os.path.join(file_path, file_name)
    base_name = os.path.splitext(file_name)[0]
    
    # 输出文件路径
    smooth_output = os.path.join(file_path, f"03_{base_name}_pinghua.xlsx")
    segment_output = os.path.join(file_path, f"04_{base_name}_fengenihe.xlsx")
    
    # 读取Excel文件
    xls = pd.ExcelFile(full_path)
    elements = xls.sheet_names
    
    # ===== 第一步：平滑处理 =====
    print("开始平滑处理数据...")
    with pd.ExcelWriter(smooth_output) as writer:
        for elem in elements:
            # 读取原始数据
            df = pd.read_excel(xls, sheet_name=elem)
            depth = df['DEPTH (nm)'].values
            conc = df[elem].values
            
            # 创建0-3000nm的均匀网格
            grid_depth = np.arange(0, 3001, 1)
            
            # 三次样条插值
            cs = CubicSpline(depth, conc, extrapolate=False)
            grid_conc = cs(grid_depth)
            
            # Savitzky-Golay平滑
            smoothed = savgol_filter(
                grid_conc, 
                window_length=smooth_window, 
                polyorder=smooth_order
            )
            smoothed = np.maximum(smoothed, 0)  # 确保非负
            
            # 保存平滑结果
            pd.DataFrame({
                'DEPTH (nm)': grid_depth,
                elem: smoothed
            }).to_excel(writer, sheet_name=elem, index=False)
    
    print(f"平滑数据已保存至: {smooth_output}")
    
    # ===== 第二步：分段拟合处理 =====
    print("开始分段拟合处理...")
    
    # 创建全局插值器
    interp_dict = {}
    with pd.ExcelFile(smooth_output) as smooth_xls:
        for elem in elements:
            df_smooth = pd.read_excel(smooth_xls, sheet_name=elem)
            # 线性插值器
            interp_dict[elem] = interp1d(
                df_smooth['DEPTH (nm)'], 
                df_smooth[elem], 
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
    
    # 生成分段边界
    segments = []
    # 第一段 [0, D1]
    start = 0.0
    while start < D1:
        end = min(start + step1, D1)
        segments.append((start, end))
        start = end
    
    # 第二段 [D1, D2]
    start = D1
    while start < D2:
        end = min(start + step2, D2)
        segments.append((start, end))
        start = end
    
    # 第三段 [D2, D3]
    start = D2
    while start < D3:
        end = min(start + step3, D3)
        segments.append((start, end))
        start = end
    
    # 准备分段结果存储
    results = []
    
    # 对每个元素处理
    for elem in elements:
        interp_func = interp_dict[elem]
        
        for seg_idx, (abs_min, abs_max) in enumerate(segments, 1):
            seg_length = abs_max - abs_min
            
            # 在小分割内采样10个点
            rel_depth = np.linspace(0, seg_length, num=10)
            abs_depth_points = np.linspace(abs_min, abs_max, num=10)
            conc_points = interp_func(abs_depth_points)
            
            # 多项式拟合 (1阶或2阶)
            if len(conc_points) < 3:
                # 点太少时使用线性拟合
                coeffs = np.polyfit(rel_depth, conc_points, deg=1)
            else:
                # 使用二次拟合
                coeffs = np.polyfit(rel_depth, conc_points, deg=2)
            
            # 构建方程字符串
            equation = "y = "
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
                '拟合方程f(X)': equation
            })
    
    # 保存分段结果
    result_df = pd.DataFrame(results)
    result_df.to_excel(segment_output, index=False)
    
    print(f"分段拟合结果已保存至: {segment_output}")
    print("处理完成！")

if __name__ == "__main__":
    process_sims_data()
