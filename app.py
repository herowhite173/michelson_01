import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec

# ======================== 设置 Matplotlib 支持中文 ========================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================== 核心计算函数 ========================
def calculate_interference(k, h, wavelength_option):
    """
    计算迈克尔逊干涉图样

    参数:
        k: 干涉级次 (1-200)
        h: 间距 (nm, 10-2000)
        wavelength_option: 波长选项字符串
    """
    # 波长映射
    wavelength_map = {
        "红光 (650 nm)": 650e-9,
        "绿光 (532 nm)": 532e-9,
        "蓝光 (473 nm)": 473e-9,
        "黄光 (589.3 nm)": 589.3e-9
    }
    lamd = wavelength_map[wavelength_option]

    # 固定参数
    hi = 400e-3  # 观察屏距离 (m)
    N = 1024  # 网格点数

    # 单位转换
    h1 = h * 1e-9  # nm -> m

    # 生成坐标网格
    ym = 250e-3
    x = np.linspace(-ym, ym, N)
    y = np.linspace(-ym, ym, N)
    X, Y = np.meshgrid(x, y)

    # 物理计算
    r2 = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan(r2 / hi)
    di = lamd + k * lamd / 2 + h1
    delta = 2 * di * np.cos(theta)
    phi = 2 * np.pi * delta / lamd
    I = 4 * 10 * np.cos(phi / 2) ** 2
    I = I / np.max(I)

    # 根据波长选择颜色映射
    if lamd == 650e-9:
        cmap = 'Reds'
    elif lamd == 532e-9:
        cmap = 'Greens'
    elif lamd == 473e-9:
        cmap = 'Blues'
    elif lamd == 589.3e-9:
        cmap = 'YlOrBr'
    else:
        cmap = 'viridis'

    # 创建图形
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3])

    # ---- 左图：原理图 ----
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal')

    # 计算原理图显示用的h值
    h2_display = h * 1e-2 if h <= 300 else 300e-2

    # 绘制光学元件
    ax1.plot([-6, 6], [18, 18], linestyle='-', color='g', linewidth=2, label='M2')
    ax1.plot([20, 20], [-8, 4], linestyle='-', color='g', linewidth=2, label='M1')
    ax1.plot([-6, 6], [22 + h2_display, 22 + h2_display],
             linestyle='--', color='g', linewidth=2, label="M2'")

    # 添加文字标注
    ax1.text(10, 15, "M1", fontsize=12, color='g', fontweight='bold')
    ax1.text(10, 21 + h2_display, "M2'", fontsize=12, color='g', fontweight='bold')
    ax1.text(18, 5, 'M2', fontsize=12, color='g', fontweight='bold')

    # 绘制分束镜和光线
    ax1.plot([-4, 4], [-6, 2], linestyle='-', color='k', linewidth=2)
    ax1.plot([4, 12], [-6, 2], linestyle='-', color='k', linewidth=2)
    ax1.plot([0, 0], [-22, 18], linestyle='-', color='r', linewidth=1, alpha=0.7)
    ax1.plot([-20, 20], [-2, -2], linestyle='-', color='r', linewidth=0.7, alpha=0.7)

    # 设置坐标轴范围
    ax1.set_ylim(-28, 28)
    ax1.set_xlim(-28, 28)
    ax1.set_facecolor('lightgray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('原理图', fontsize=14, fontweight='bold')

    # ---- 右图：干涉图样 ----
    ax2 = fig.add_subplot(gs[1])
    im = ax2.imshow(I, cmap=cmap, extent=[-10, 10, -10, 10], origin='lower')
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_title("迈克尔逊干涉 (等倾条纹)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("x (mm)", fontsize=12)
    ax2.set_ylabel("y (mm)", fontsize=12)

    # 添加颜色条
    plt.colorbar(im, ax=ax2, label='相对光强')

    plt.tight_layout()
    return fig


# ======================== Streamlit 应用界面 ========================
def main():
    # 页面配置
    st.set_page_config(
        page_title="迈克尔逊干涉实验仿真",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义CSS，优化手机显示
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stSlider label {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 标题
    st.markdown("""
    <div class="main-header">
        <h1>🔬 迈克尔逊干涉实验仿真</h1>
        <p>手机版 | 实时交互 | 等倾干涉</p>
    </div>
    """, unsafe_allow_html=True)

    # 创建两列布局，左侧放参数，右侧放图像（手机版会自适应）
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ 参数调节")

        # 干涉级次 K
        k = st.slider(
            "**干涉级次 K**",
            min_value=1,
            max_value=200,
            value=50,
            step=1,
            help="K值越大，条纹越密"
        )

        # 间距 h (nm)
        h = st.slider(
            "**间距 h (nm)**",
            min_value=10,
            max_value=2000,
            value=20,
            step=10,
            help="M1与M2'之间的间距"
        )

        # 波长选择
        wavelength = st.selectbox(
            "**选择波长**",
            options=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"],
            index=3,
            help="选择入射光波长"
        )

        # 显示当前参数值
        st.markdown("---")
        st.markdown("### 📊 当前参数")
        st.info(f"""
        - 干涉级次 K = {k}
        - 间距 h = {h} nm
        - 波长 = {wavelength}
        """)

        # 实验原理（可折叠）
        with st.expander("📚 实验原理", expanded=False):
            st.markdown("""
            **迈克逊干涉**属于等倾干涉，相同入射角度的光线形成同心圆环。

            - **明纹条件**:  \(2d \cos \theta = K\lambda\)
            - **暗纹条件**:  \(2d \cos \theta = (2K+1)\frac{\lambda}{2}\)

            其中 \(d\) 为M1与M2'间距，\(\theta\) 为入射角。
            """)

    with col2:
        # 计算并显示干涉图样
        with st.spinner('🔄 正在计算干涉图样...'):
            fig = calculate_interference(k, h, wavelength)
            st.pyplot(fig, use_container_width=True)

        # 结果显示
        st.success(f"✅ 计算完成 | K={k} | h={h}nm | {wavelength}")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>基于 Python + Streamlit 构建 | 物理仿真实验 | 手机扫码即用</p>
        <p style='font-size: 0.8rem;'>© 2026 波动光学实验仿真 | 全国大学生物理竞赛参赛作品</p>
    </div>
    """, unsafe_allow_html=True)


# ======================== 程序入口 ========================
if __name__ == "__main__":
    main()