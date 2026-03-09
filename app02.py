import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import qrcode
from PIL import Image
import io

# ======================== 全局配置（云端兼容版） ========================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.switch_backend('Agg')


# ======================== 核心计算函数 ========================
def calculate_interference(k, h, wavelength_option, is_mobile=False):
    wavelength_map = {
        "红光 (650 nm)": 650e-9,
        "绿光 (532 nm)": 532e-9,
        "蓝光 (473 nm)": 473e-9,
        "黄光 (589.3 nm)": 589.3e-9
    }
    lamd = wavelength_map.get(wavelength_option, 650e-9)

    N = 512 if is_mobile else 800
    hi = 400e-3
    ym = 250e-3
    h1 = h * 1e-9

    x = np.linspace(-ym, ym, N)
    y = np.linspace(-ym, ym, N)
    X, Y = np.meshgrid(x, y)

    r2 = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan(r2 / hi)
    delta = 2 * h1 * np.cos(theta) - k * lamd
    phi = 2 * np.pi * delta / lamd
    I = 4 * 10 * np.cos(phi / 2) ** 2
    I = I / np.max(I) if np.max(I) != 0 else I

    cmap_dict = {
        650e-9: 'Reds_r',
        532e-9: 'Greens_r',
        473e-9: 'Blues_r',
        589.3e-9: 'YlOrBr_r'
    }
    cmap = cmap_dict.get(lamd, 'viridis')

    fig_size = (10, 5) if is_mobile else (12, 6)
    fig = plt.figure(figsize=fig_size, dpi=96)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # ---- 原理图 ----
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal')
    h_display = min(h, 800) * 0.012
    h_display = max(h_display, 0.2)

    ax1.plot([-6, 6], [18, 18], '-', color='#2e8b57', linewidth=3, label='M2')
    ax1.plot([20, 20], [-8, 4], '-', color='#2e8b57', linewidth=3, label='M1')
    ax1.plot([-6, 6], [18 + h_display, 18 + h_display], '--', color='#ff6347', linewidth=2.5, label="M2'")

    ax1.text(0, 14, "M2", fontsize=11, color='#2e8b57', fontweight='bold')
    ax1.text(0, 18 + h_display + 1.5, "M2'", fontsize=11, color='#ff6347', fontweight='bold')
    ax1.text(21, -1, 'M1', fontsize=11, color='#2e8b57', fontweight='bold')
    ax1.text(-1, 9, f'h = {h} nm', fontsize=10, color='black', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    ax1.plot([-4, 4], [0, 4], '-', color='black', linewidth=2.5, label='分束镜')
    ax1.arrow(0, -20, 0, 35, head_width=0.6, head_length=1.2, fc='#dc143c', ec='#dc143c', alpha=0.8, linewidth=1.5)
    ax1.arrow(-18, -2, 30, 0, head_width=0.6, head_length=1.2, fc='#dc143c', ec='#dc143c', alpha=0.8, linewidth=1)

    ax1.set_ylim(-28, 38)
    ax1.set_xlim(-28, 28)
    ax1.set_facecolor('#f0f0f0')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('迈克尔逊干涉原理图', fontsize=13 if is_mobile else 14, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=7 if is_mobile else 8, framealpha=0.9)

    # ---- 干涉图样 ----
    ax2 = fig.add_subplot(gs[1])
    extent_mm = [-ym * 1000, ym * 1000, -ym * 1000, ym * 1000]
    im = ax2.imshow(I, cmap=cmap, extent=extent_mm, origin='lower', aspect='auto')

    tick_step = 80 if is_mobile else 50
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax2.set_title(f"迈克尔逊等倾干涉条纹 ({wavelength_option})",
                  fontsize=12 if is_mobile else 14, fontweight='bold', pad=10)
    ax2.set_xlabel("x (mm)", fontsize=11 if is_mobile else 12)
    ax2.set_ylabel("y (mm)", fontsize=11 if is_mobile else 12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    cbar = plt.colorbar(im, ax=ax2, label='相对光强', shrink=0.85)
    cbar.ax.tick_params(labelsize=9 if is_mobile else 10)

    plt.tight_layout(pad=2.0)
    return fig


# ======================== 二维码生成函数 ========================
def generate_qr_code(url):
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=5,
            border=2
        )
        qr.add_data(url)
        qr.make(fit=True)

        qr_img = qr.make_image(fill_color="#2c3e50", back_color="white")
        img_bytes = io.BytesIO()
        qr_img.save(img_bytes, format='PNG', dpi=(96, 96))
        img_bytes.seek(0)
        return img_bytes
    except Exception as e:
        st.warning(f"二维码生成失败: {str(e)[:50]}")
        return None


# ======================== Streamlit 主界面（云端稳定版） ========================
def main():
    # 1. 页面配置
    st.set_page_config(
        page_title="迈克尔逊干涉实验仿真",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 2. 移动端检测
    try:
        is_mobile = st.query_params.get("mobile", "") == "true"
    except:
        is_mobile = False

    # 3. 自定义CSS
    st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSlider label, .stSelectbox label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    .param-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5rem !important; }
        .stColumns { flex-direction: column !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # 4. 标题
    st.markdown("""
    <div class="main-header">
        <h1>🔬 迈克尔逊干涉实验仿真</h1>
        <p>云端版 | 实时交互 | 多端适配</p>
    </div>
    """, unsafe_allow_html=True)

    # 5. 布局
    if is_mobile:
        col1, col2 = st.container(), st.container()
    else:
        col1, col2 = st.columns([1, 2])

    # ========== 左侧栏（参数+二维码） ==========
    with col1:
        # 二维码和访问地址（关键：改用占位符，避免内部API）
        st.markdown("### 📱 扫码访问")
        st.info("""
        💡 请使用浏览器地址栏中的 URL 生成二维码，或直接分享该链接。
        手机扫码或点击链接即可访问。
        """)

        # 参数调节区
        st.markdown("### ⚙️ 参数调节")
        with st.container():
            # 快捷按钮
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📌 经典红光参数", use_container_width=True):
                    st.session_state.update({'k': 50, 'h': 100, 'wavelength': "红光 (650 nm)"})
            with col_btn2:
                if st.button("🔄 重置参数", use_container_width=True):
                    st.session_state.update({'k': 50, 'h': 100, 'wavelength': "红光 (650 nm)"})

            # 初始化Session State
            if 'k' not in st.session_state:
                st.session_state['k'] = 50
                st.session_state['h'] = 100
                st.session_state['wavelength'] = "红光 (650 nm)"

            # 滑块
            k = st.slider(
                "**干涉级次 K**",
                min_value=1, max_value=200,
                value=st.session_state['k'], step=1,
                help="K值越大，干涉条纹越密集"
            )
            h = st.slider(
                "**间距 h (nm)**",
                min_value=10, max_value=2000,
                value=st.session_state['h'], step=10,
                help="M1与M2'的空气膜间距 (10-2000nm)"
            )
            wavelength = st.selectbox(
                "**入射光波长**",
                options=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"],
                index=["红光 (650 nm)", "绿光 (532 nm)", "蓝光 (473 nm)", "黄光 (589.3 nm)"].index(
                    st.session_state['wavelength'])
            )

            # 更新Session State
            st.session_state.update({'k': k, 'h': h, 'wavelength': wavelength})

        # 当前参数展示
        st.markdown("### 📊 当前参数")
        st.markdown(f"""
        <div class="param-card">
            • 干涉级次 K = <strong>{k}</strong><br>
            • 间距 h = <strong>{h}</strong> nm<br>
            • 波长 = <strong>{wavelength}</strong>
        </div>
        """, unsafe_allow_html=True)

        # 实验原理
        with st.expander("📚 实验原理", expanded=False):
            st.markdown("""
            **迈克尔逊干涉**是典型的等倾干涉，相同入射角的光线形成同心圆环条纹：
            - 明纹条件： $2h \\cos \\theta = K\\lambda$
            - 暗纹条件： $2h \\cos \\theta = (2K+1)\\frac{\\lambda}{2}$

            **参数说明**：
            - $h$：M1与M2'的间距（nm）
            - $\\theta$：入射光与法线的夹角
            - $K$：干涉级次（整数）
            - $\\lambda$：入射光波长（nm）
            """)

    # ========== 右侧栏（仿真结果） ==========
    with col2:
        try:
            with st.spinner('🔄 正在计算干涉图样...'):
                fig = calculate_interference(k, h, wavelength, is_mobile)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.success(f"✅ 计算完成 | K={k} | h={h}nm | {wavelength}")
        except Exception as e:
            st.error(f"❌ 计算出错：{str(e)[:80]}...")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 1rem; font-size: 0.85rem;'>
        <p>基于 Streamlit Cloud 部署 | 波动光学实验仿真</p>
        <p style='font-size: 0.75rem;'>© 2026 全国大学生物理竞赛参赛作品</p>
    </div>
    """, unsafe_allow_html=True)


# ======================== 程序入口 ========================
if __name__ == "__main__":
    main()