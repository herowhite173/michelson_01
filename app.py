import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import qrcode
from PIL import Image
import io
import sys

# ======================== 全局配置 ========================
# 设置 Matplotlib 支持中文（多系统兼容）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
# 设置Matplotlib渲染后端（解决部分环境显示问题）
plt.switch_backend('Agg')


# ======================== 核心计算函数 ========================
def calculate_interference(k, h, wavelength_option, is_mobile=False):
    """
    计算迈克尔逊干涉图样（最终优化版）

    参数:
        k: 干涉级次 (1-200)
        h: 间距 (nm, 10-2000)
        wavelength_option: 波长选项字符串
        is_mobile: 是否为移动端（优化性能）
    """
    # 波长映射（增加容错）
    wavelength_map = {
        "红光 (650 nm)": 650e-9,
        "绿光 (532 nm)": 532e-9,
        "蓝光 (473 nm)": 473e-9,
        "黄光 (589.3 nm)": 589.3e-9
    }
    lamd = wavelength_map.get(wavelength_option, 650e-9)

    # 动态调整网格点数（平衡性能与精度）
    N = 512 if is_mobile else 1024

    # 固定物理参数
    hi = 400e-3  # 观察屏距离 (m)
    ym = 250e-3  # 坐标范围 (m)

    # 单位转换：nm -> m
    h1 = h * 1e-9

    # 生成坐标网格
    x = np.linspace(-ym, ym, N)
    y = np.linspace(-ym, ym, N)
    X, Y = np.meshgrid(x, y)

    # 迈克尔逊干涉核心物理计算（修正公式）
    r2 = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan(r2 / hi)  # 入射角

    # 光程差（加入干涉级次关联，让K值真正影响条纹密度）
    delta = 2 * h1 * np.cos(theta) - k * lamd

    # 相位差与光强分布（物理公式+归一化）
    phi = 2 * np.pi * delta / lamd
    I = 4 * 10 * np.cos(phi / 2) ** 2  # 保留实验系数
    I = I / np.max(I) if np.max(I) != 0 else I  # 防止除零

    # 波长对应的颜色映射（更精准的配色）
    cmap_dict = {
        650e-9: 'Reds_r',  # 反向色板，中心更亮
        532e-9: 'Greens_r',
        473e-9: 'Blues_r',
        589.3e-9: 'YlOrBr_r'
    }
    cmap = cmap_dict.get(lamd, 'viridis')

    # 创建图形（优化尺寸适配）
    fig_size = (10, 5) if is_mobile else (14, 6)
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3])

    # ---- 左图：优化版原理图 ----
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal')

    # 原理图间距缩放（更自然的显示效果）
    h_display = min(h, 800) * 0.012
    h_display = max(h_display, 0.2)

    # 绘制光学元件（优化位置和样式）
    # M2镜
    ax1.plot([-6, 6], [18, 18], linestyle='-', color='#2e8b57', linewidth=3, label='M2')
    # M1镜
    ax1.plot([20, 20], [-8, 4], linestyle='-', color='#2e8b57', linewidth=3, label='M1')
    # M2'虚像
    ax1.plot([-6, 6], [18 + h_display, 18 + h_display],
             linestyle='--', color='#ff6347', linewidth=2.5, label="M2'")

    # 文字标注（更清晰的位置）
    ax1.text(0, 14, "M2", fontsize=11, color='#2e8b57', fontweight='bold')
    ax1.text(0, 18 + h_display + 1.5, "M2'", fontsize=11, color='#ff6347', fontweight='bold')
    ax1.text(21, -1, 'M1', fontsize=11, color='#2e8b57', fontweight='bold')
    ax1.text(-1, 9, f'h = {h} nm', fontsize=10, color='black', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # 绘制分束镜和光路（带箭头，更直观）
    # 分束镜
    ax1.plot([-4, 4], [0, 4], linestyle='-', color='black', linewidth=2.5, label='分束镜')
    # 入射光（带箭头）
    ax1.arrow(0, -20, 0, 35, head_width=0.6, head_length=1.2,
              fc='#dc143c', ec='#dc143c', alpha=0.8, linewidth=1.5)
    # 反射光（带箭头）
    ax1.arrow(-18, -2, 30, 0, head_width=0.6, head_length=1.2,
              fc='#dc143c', ec='#dc143c', alpha=0.8, linewidth=1)

    # 原理图样式优化
    ax1.set_ylim(-28, 38)
    ax1.set_xlim(-28, 28)
    ax1.set_facecolor('#f0f0f0')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('迈克尔逊干涉原理图', fontsize=13 if is_mobile else 14,
                  fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=7 if is_mobile else 8, framealpha=0.9)

    # ---- 右图：干涉图样（优化显示）----
    ax2 = fig.add_subplot(gs[1])
    # 显示范围转换为mm（物理尺寸）
    extent_mm = [-ym * 1000, ym * 1000, -ym * 1000, ym * 1000]
    im = ax2.imshow(I, cmap=cmap, extent=extent_mm, origin='lower', aspect='auto')

    # 坐标轴优化（更易读）
    tick_step = 80 if is_mobile else 50
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax2.set_title(f"迈克尔逊等倾干涉条纹 ({wavelength_option})",
                  fontsize=12 if is_mobile else 14, fontweight='bold', pad=10)
    ax2.set_xlabel("x (mm)", fontsize=11 if is_mobile else 12)
    ax2.set_ylabel("y (mm)", fontsize=11 if is_mobile else 12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 颜色条优化
    cbar = plt.colorbar(im, ax=ax2, label='相对光强', shrink=0.85)
    cbar.ax.tick_params(labelsize=9 if is_mobile else 10)

    plt.tight_layout(pad=2.0)
    return fig


# ======================== 二维码生成函数（最终版）========================
def generate_qr_code(url, is_mobile=False):
    """生成适配不同设备的二维码"""
    try:
        # 动态调整二维码尺寸
        box_size = 4 if is_mobile else 5
        border = 1 if is_mobile else 2

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,  # 中等容错（更稳定）
            box_size=box_size,
            border=border
        )
        qr.add_data(url)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color="#2c3e50", back_color="white")

        # 转换为字节流
        img_bytes = io.BytesIO()
        qr_img.save(img_bytes, format='PNG', dpi=(96, 96))
        img_bytes.seek(0)

        return img_bytes
    except Exception as e:
        st.warning(f"二维码生成失败: {str(e)[:50]}")
        return None


# ======================== Streamlit 主界面（最终版）========================
def main():
    # ========== 兼容 Streamlit 1.28.0 的 query_params 获取方式 ==========
    # 获取移动端参数（兼容 1.28.0）
    mobile_param = st.query_params.get("mobile", "")
    is_mobile_by_param = (mobile_param == "true")

    # 页面配置
    st.set_page_config(
        page_title="迈克尔逊干涉实验仿真",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed" if is_mobile_by_param else "expanded"
    )

    # 移动端检测（再次获取，用于后续逻辑）
    mobile_param = st.query_params.get("mobile", "")
    is_mobile = (mobile_param == "true")

    # 自定义CSS（响应式设计+美化）
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: #f8f9fa;
    }}
    .main-header {{
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .stSlider label {{
        font-size: {'0.9rem' if is_mobile else '1rem'} !important;
        font-weight: 600 !important;
    }}
    .stSelectbox label {{
        font-size: {'0.9rem' if is_mobile else '1rem'} !important;
        font-weight: 600 !important;
    }}
    .qr-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }}
    .stAlert {{
        padding: 0.8rem;
        border-radius: 8px;
    }}
    @media (max-width: 768px) {{
        .stColumns {{
            flex-direction: column !important;
        }}
        .main-header h1 {{
            font-size: 1.5rem !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # 标题（响应式）
    st.markdown("""
    <div class="main-header">
        <h1>🔬 迈克尔逊干涉实验仿真</h1>
        <p>实时交互 | 等倾干涉 | 多端适配</p>
    </div>
    """, unsafe_allow_html=True)

    # 布局（移动端单列，桌面端双列）
    if is_mobile:
        col1 = st.container()
        col2 = st.container()
    else:
        col1, col2 = st.columns([1, 2])

    # ---------- col1 内容 ----------
    with col1:
        # 二维码显示
        st.markdown("### 📱 手机扫码访问")

        # 适配本地/部署环境的URL
        if 'STREAMLIT_SERVER_BASEURL_PATH' in st.secrets or not st.get_option("server.runOnSave"):
            app_url = "https://michelson01-ep7jkx2fbgnuttjjhzespm.streamlit.app"
        else:
            app_url = "http://localhost:8501"

        # 生成并显示二维码
        qr_bytes = generate_qr_code(app_url, is_mobile)
        if qr_bytes:
            qr_width = 120 if is_mobile else 150
            st.image(qr_bytes, caption="微信/浏览器扫码", width=qr_width)

        st.markdown(f"""
        <p style='text-align: center; font-size: {'0.7rem' if is_mobile else '0.8rem'}; color: gray;'>
            手机扫码，随时随地实验
        </p>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # 参数调节
        st.markdown("### ⚙️ 参数调节")

        # 经典参数快捷按钮
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📌 经典红光参数", use_container_width=True):
                st.session_state['k'] = 50
                st.session_state['h'] = 100
                st.session_state['wavelength'] = "红光 (650 nm)"
        with col_btn2:
            if st.button("🔄 重置参数", use_container_width=True):
                st.session_state.pop('k', None)
                st.session_state.pop('h', None)
                st.session_state.pop('wavelength', None)

        # 干涉级次滑块
        k = st.slider(
            "**干涉级次 K**",
            min_value=1,
            max_value=200,
            value=st.session_state.get('k', 50),
            step=1,
            help="K值越大，干涉条纹越密集 | 物理意义：明纹级次",
            key='k_slider'
        )
        st.session_state['k'] = k

        # 间距滑块
        h = st.slider(
            "**间距 h (nm)**",
            min_value=10,
            max_value=2000,
            value=st.session_state.get('h', 100),
            step=10,
            help="M1与M2'的空气膜间距 | 范围：10-2000nm",
            key='h_slider'
        )
        st.session_state['h'] = h

        # 波长选择框
        wavelength = st.selectbox(
            "**入射光波长**",
            options=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"],
            index=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"].index(
                st.session_state.get('wavelength', "红光 (650 nm)")
            ),
            help="不同波长对应不同颜色的干涉条纹",
            key='wavelength_select'
        )
        st.session_state['wavelength'] = wavelength

        # 当前参数显示
        st.markdown("---")
        st.markdown("### 📊 当前参数")
        st.info(f"""
        <div style='font-size: {'0.9rem' if is_mobile else '1rem'};'>
        • 干涉级次 K = <strong>{k}</strong><br>
        • 间距 h = <strong>{h}</strong> nm<br>
        • 波长 = <strong>{wavelength}</strong>
        </div>
        """, unsafe_allow_html=True)

        # 实验原理（修正：避免f-string反斜杠错误）
        with st.expander("📚 实验原理", expanded=False):
            原理_html = f"""
            <div style='font-size: {"0.85rem" if is_mobile else "0.95rem"}; line-height: 1.6;'>
            **迈克尔逊干涉**是典型的等倾干涉，相同入射角的光线形成同心圆环条纹：<br>
            - 明纹条件： \(2h \cos \theta = K\lambda\)<br>
            - 暗纹条件： \(2h \cos \theta = (2K+1)\frac{{\lambda}}{{2}}\)<br><br>
            🔍 公式说明：<br>
            • \(h\)：M1与M2'的间距（nm）<br>
            • \(\theta\)：入射光与法线的夹角<br>
            • \(K\)：干涉级次（整数）<br>
            • \(\lambda\)：入射光波长（nm）
            </div>
            """
            st.markdown(原理_html, unsafe_allow_html=True)


# ---------- col2 内容 ----------
with col2:
    # 干涉图样计算与显示
    try:
        with st.spinner('🔄 正在计算干涉图样...'):
            # 调用核心计算函数
            fig = calculate_interference(k, h, wavelength, is_mobile)
            # 显示图像（适配容器）
            st.pyplot(fig, use_container_width=True)
            # 清理Matplotlib资源
            plt.close(fig)

        # 成功提示（美化）
        st.success(f"""
            ✅ 计算完成 | 
            K=<strong>{k}</strong> | 
            h=<strong>{h}</strong>nm | 
            {wavelength}
            """, unsafe_allow_html=True)

    except Exception as e:
        # 友好的错误提示
        st.error(f"""
            ❌ 计算出错：{str(e)[:80]}...<br>
            💡 建议：刷新页面或调整参数重试
            """, unsafe_allow_html=True)
        # 调试模式下显示完整错误（可选）
        if st.get_option("server.runOnSave"):
            st.code(f"详细错误：{str(e)}", language='python')

# 页脚（响应式）
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #6c757d; padding: 1rem; font-size: {'0.75rem' if is_mobile else '0.85rem'};'>
        <p>基于 Python + Streamlit 构建 | 波动光学实验仿真</p>
        <p style='font-size: {'0.7rem' if is_mobile else '0.8rem'};'>
            © 2026 全国大学生物理竞赛参赛作品 | 多端适配 · 实时交互
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================== 程序入口 ========================
if __name__ == "__main__":
    # 初始化session_state（防止首次运行报错）
    if 'k' not in st.session_state:
        st.session_state['k'] = 50
    if 'h' not in st.session_state:
        st.session_state['h'] = 100
    if 'wavelength' not in st.session_state:
        st.session_state['wavelength'] = "红光 (650 nm)"

    # 运行主程序
    main()