import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
import io
import warnings
import random
from datetime import datetime, timedelta
from PIL import ImageDraw, ImageFont, Image
import pandas as pd
import time

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.switch_backend("Agg")

DEFAULT_H = 0.0
OPTICAL_OFFSET_MM = 2.50000
OPTICAL_OFFSET_NM = OPTICAL_OFFSET_MM * 1e6
DEFAULT_WL = "红光 (650 nm)"
wave_dict = {
    "红光 (650 nm)": 650e-9,
    "绿光 (532 nm)": 532e-9,
    "蓝光 (473 nm)": 473e-9,
    "黄光 (589.3 nm)": 589.3e-9,
}
TRUE_WAVELENGTH = {
    "红光 (650 nm)": 650.0,
    "绿光 (532 nm)": 532.0,
    "蓝光 (473 nm)": 473.0,
    "黄光 (589.3 nm)": 589.3,
}

gif_map = {
    "红光 (650 nm)": "michelson_red.gif",
    "绿光 (532 nm)": "michelson_green.gif",
    "蓝光 (473 nm)": "michelson_blue.gif",
    "黄光 (589.3 nm)": "michelson_yellow.gif",
}

K_CONST = 20
RECORD_N = [0, 50, 100, 150, 200, 250, 300, 350]
MAX_USES_PER_HOUR = 3
MAX_H_VALUE = 20000.0
MAX_FRINGE_COUNT = 360
ERROR_RANGE = 0.0015


def get_step_per_fringe(wl_name):
    lambda_m = wave_dict[wl_name]
    return (lambda_m * 1e9) / 2


def get_thickness_increment_per_10_fringes(wl_name):
    return 10 * get_step_per_fringe(wl_name)


def calc_fringe_count_from_h(h, wl_name):
    step = get_step_per_fringe(wl_name)
    return int(round((h - DEFAULT_H) / step))


def calc_h_from_fringe_count(fringe_count, wl_name):
    step = get_step_per_fringe(wl_name)
    return DEFAULT_H + fringe_count * step


def add_measurement_error(value):
    error = random.uniform(-ERROR_RANGE, ERROR_RANGE)
    return round(value + error, 6)


def check_usage_limit(mode):
    now = datetime.now()
    if f"usage_records_{mode}" not in st.session_state:
        st.session_state[f"usage_records_{mode}"] = []
    valid_records = [
        t for t in st.session_state[f"usage_records_{mode}"]
        if now - t < timedelta(hours=1)
    ]
    st.session_state[f"usage_records_{mode}"] = valid_records
    return len(valid_records) < MAX_USES_PER_HOUR


def add_usage_record(mode):
    if f"usage_records_{mode}" not in st.session_state:
        st.session_state[f"usage_records_{mode}"] = []
    st.session_state[f"usage_records_{mode}"].append(datetime.now())


def get_remaining_uses(mode):
    now = datetime.now()
    if f"usage_records_{mode}" not in st.session_state:
        return MAX_USES_PER_HOUR
    valid_records = [
        t for t in st.session_state[f"usage_records_{mode}"]
        if now - t < timedelta(hours=1)
    ]
    return max(0, MAX_USES_PER_HOUR - len(valid_records))


def check_demo_export_quota():
    return check_usage_limit("demo_export")


def consume_demo_export_quota():
    add_usage_record("demo_export")


def get_demo_export_remaining():
    return get_remaining_uses("demo_export")


def check_sim_complete_quota():
    return check_usage_limit("sim_complete")


def consume_sim_complete_quota():
    add_usage_record("sim_complete")


def get_sim_complete_remaining():
    return get_remaining_uses("sim_complete")


def reset_all_to_default():
    st.session_state.h = DEFAULT_H
    st.session_state.wl = DEFAULT_WL
    st.session_state.fringe_count = 0
    st.session_state.pos_data = {n: None for n in RECORD_N}
    initial_actual_mm = add_measurement_error(OPTICAL_OFFSET_MM)
    st.session_state.pos_data[0] = initial_actual_mm
    st.session_state.experiment_completed = False
    st.session_state.need_reset = True
    st.session_state.gif_click_count = 0
    st.session_state.lock_wavelength = False  # 重置解锁光源
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    st.session_state.reset_counter += 1


def update_pos_data():
    step = get_step_per_fringe(st.session_state.wl)
    for n in RECORD_N:
        if st.session_state.fringe_count >= n and st.session_state.pos_data[n] is None:
            theoretical_h_mm = OPTICAL_OFFSET_MM + (DEFAULT_H + n * step) / 1e6
            if n == 0:
                st.session_state.pos_data[n] = theoretical_h_mm
            else:
                st.session_state.pos_data[n] = add_measurement_error(theoretical_h_mm)


def force_update_pos_data():
    step = get_step_per_fringe(st.session_state.wl)
    for n in RECORD_N:
        if st.session_state.fringe_count >= n:
            theoretical_h_mm = OPTICAL_OFFSET_MM + (DEFAULT_H + n * step) / 1e6
            if n == 0:
                st.session_state.pos_data[n] = theoretical_h_mm
            else:
                st.session_state.pos_data[n] = add_measurement_error(theoretical_h_mm)
        else:
            st.session_state.pos_data[n] = None


def calculate_uncertainty():
    try:
        data = [st.session_state.pos_data[n] for n in RECORD_N]
        if None in data:
            return None
        data_nm = [d * 1e6 for d in data]
        deltas = []
        for i in range(4):
            deltas.append(data_nm[i + 4] - data_nm[i])
        delta_h_mean = np.mean(deltas)
        lambda_measured = 2 * delta_h_mean / 200
        lambda_true = TRUE_WAVELENGTH[st.session_state.wl]
        rel_error = abs(lambda_measured - lambda_true) / lambda_true * 100
        std = np.std(deltas, ddof=1)
        u_A = std / np.sqrt(len(deltas))
        U = 2 * u_A
        return {
            "lambda_measured": round(lambda_measured, 3),
            "lambda_true": lambda_true,
            "rel_error": round(rel_error, 3),
            "u_A": round(u_A, 3),
            "U": round(U, 3),
        }
    except Exception:
        return None


def calculate_data(k_const, h, lamd):
    n_grid = 150
    hi = 400e-3
    view_range = 10e-3
    x = np.linspace(-view_range, view_range, n_grid)
    y = np.linspace(-view_range, view_range, n_grid)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan(r / hi)
    h_total_nm = OPTICAL_OFFSET_NM + h
    h_meters = h_total_nm * 1e-9
    d_eff = h_meters + (1 + k_const / 2) * lamd
    delta = 2 * d_eff * np.cos(theta)
    phi = 2 * np.pi * delta / lamd
    i_intensity = 4 * 10 * np.cos(phi / 2) ** 2
    if np.max(i_intensity) > 0:
        i_intensity = i_intensity / np.max(i_intensity)
    cmap_map = {650e-9: "Reds", 532e-9: "Greens", 473e-9: "Blues", 589.3e-9: "YlOrBr"}
    cmap = cmap_map.get(lamd, "Reds")
    return i_intensity, cmap


def plot_interference(i_intensity, cmap, h):
    try:
        fig = plt.figure(figsize=(13, 6), dpi=100)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.2])
        ax1 = fig.add_subplot(gs[0])
        ax1.set_aspect("equal")
        display_h = h / 1000
        h2 = np.clip(display_h, 0.0, 30.0)

        ax1.plot((-6.0, 6.0), (18.0, 18.0), "-g", linewidth=2)
        ax1.plot((20.0, 20.0), (-8.0, 4.0), "-g", linewidth=2)

        if st.session_state.mode == "sim":
            ax1.plot((-6.0, 6.0), (22.0, 22.0), "--g", linewidth=2)
            ax1.text(10, 21, "M2'", fontsize=12, color="g")
        else:
            ax1.plot((-6.0, 6.0), (22.0 + h2, 22.0 + h2), "--g", linewidth=2)
            ax1.text(10, 21 + h2, "M2'", fontsize=12, color="g")

        ax1.text(10, 15, "M1", fontsize=12, color="g")
        ax1.text(18, 7, "M2", fontsize=12, color="g")
        ax1.text(-10, -10, "分光镜", fontsize=12, color="black")
        ax1.text(3, -10, "补偿镜", fontsize=12, color="black")
        ax1.plot((-4.0, 4.0), (-6.0, 2.0), "-k", linewidth=2)
        ax1.plot((4.0, 12.0), (-6.0, 2.0), "-k", linewidth=2)
        ax1.plot((0.0, 0.0), (-22.0, 18.0), "-r", linewidth=1)
        ax1.plot((-20.0, 20.0), (-2.0, -2.0), "-r", linewidth=0.7)

        ax1.set_ylim(-28.0, 28.0)
        ax1.set_xlim(-28.0, 28.0)
        ax1.set_facecolor("lightgray")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f"原理图 (动镜移动={h:.1f} nm)")

        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(i_intensity, cmap=cmap, extent=[-10, 10, -10, 10], origin="lower")
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
        actual_mm = OPTICAL_OFFSET_MM + h / 1e6
        ax2.set_title(f"迈克尔逊干涉 (镜片间距={actual_mm:.6f} mm)")
        ax2.grid(alpha=0.3)
        plt.tight_layout(pad=2)
        return fig
    except Exception as e:
        st.error(f"绘图错误: {str(e)}")
        return plt.figure(figsize=(13, 6), dpi=100)


def add_watermark(img, k_const, h, wl):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    actual_mm = OPTICAL_OFFSET_MM + h / 1e6
    line1 = f"实验日期：{now}"
    line2 = f"K={k_const}  间距={actual_mm:.6f}mm  移动={h:.1f}nm  {wl}"
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 26)
    except Exception:
        font = ImageFont.load_default(size=26)
    draw.text((20, img.height - 65), line1, fill=(255, 0, 0), font=font)
    draw.text((20, img.height - 30), line2, fill=(255, 0, 0), font=font)
    return img


@st.dialog("📊 波长不确定度计算结果", width="large")
def show_uncertainty_dialog():
    res = calculate_uncertainty()
    if not res:
        st.error("❌ 数据不完整，请先完成8组实验数据采集！")
        return
    st.markdown(
        f"""
    <div style="font-size:18px; line-height:1.8;">
    <h3 style='text-align:center;'>✅ 实验数据处理结果（逐差法）</h3>
    <hr>
    <b>当前实验波长：</b> {st.session_state.wl}<br>
    <b>标准波长 λ₀：</b> {res['lambda_true']} nm<br>
    <b>测量波长平均值 λ：</b> {res['lambda_measured']} nm<br>
    <b>相对误差：</b> {res['rel_error']} %<br>
    <b>A类不确定度 u_A：</b> ±{res['u_A']} nm<br>
    <b>扩展不确定度 U(k=2)：</b> ±{res['U']} nm<br>
    <hr>
    <p style='color:green; text-align:center;'>数据有效，可用于实验报告</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


@st.dialog("📘 实验原理", width="small")
def show_principle_dialog():
    st.markdown(
        """ 
    <div style="font-size:17px;"> 
    1. 利用分振幅法实现光的干涉<br>
    2. 光程差公式：Δ = 2d·cosθ。d为等效空气膜厚度，θ为光线与光轴夹角<br>
    3. 动镜每移动 λ/2，中心条纹就会冒出或缩进1个<br>
    4. 条纹每变化 N 条，厚度变化 Δh = N·λ/2<br> 
    5. 红光相干长度：10 mm（典型热光源）<br> 
    6. 允许动镜最大移动：5 mm<br> 
    </div> 
    """,
        unsafe_allow_html=True,
    )


@st.dialog("📖 使用说明", width="small")
def show_guide_dialog():
    demo_export_remaining = get_demo_export_remaining()
    sim_complete_remaining = get_sim_complete_remaining()
    st.markdown(
        f"""
    <div style="font-size:17px; line-height:1.6;">
    <b>📋 使用步骤</b><br>
    1. 选择演示模式/实验模式，不可以同时使用<br>
    2. 实验模式下先进行光源设定，再点击「冒出50个条纹」<br>
    4.开始采数后禁止切换光源！<br>
    5. <b style='color:red;'>观察效果：条纹从中心向外冒出50个！</b><br>
    6. 需要连续点击7次，每次冒出50个条纹，共350个条纹<br>
    7. 数据采集完成后，点击【不确定度】查看计算结果<br><br>
    <b>⚠防作弊措施</b><br>
    1.演示模式：PNG/GIF导出合计 {MAX_USES_PER_HOUR} 次/小时<br>
    2.实验模式：完成完整实验 {MAX_USES_PER_HOUR} 次/小时<br>
    3.当前剩余：演示导出 {demo_export_remaining} 次 | 实验完成 {sim_complete_remaining} 次<br><br>
    </div>
    """,
        unsafe_allow_html=True,
    )


@st.dialog("冒出50个条纹", width="medium")
def play_fringe_animation(_current_count, _click_count, wl_name):
    st.markdown(
        """
    <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 750px !important;
            max-width: 90vw !important;
        }
        div[data-testid="stDialog"] > div:first-child {
            padding: 0px !important;
        }
        div[data-testid="stDialog"] h2 {
            margin-bottom: 5px !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    gif_path = gif_map.get(wl_name, "michelson_red.gif")
    st.image(gif_path)
    time.sleep(10)


def main():
    st.set_page_config(page_title="迈克尔逊干涉实验", page_icon="🔬", layout="wide")

    # 手机检测（只保留一次）
    mobile_view = False
    try:
        headers = st.context.headers
        user_agent = headers.get('User-Agent', '').lower()
        mobile_view = 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent
    except:
        mobile_view = False

    st.markdown(
        """ 
    <style> 
    /* 隐藏顶部标题栏 + 右上角三个点菜单 */
    header[data-testid="stHeader"] { display: none !important; }
    /* 隐藏右下角的两个图标（帮助+升级） */
    div[data-testid="stToolbar"] { display: none !important; }
    .stApp > div:nth-child(3) { display: none !important; }
    
    html, body, [class*="stText"] { font-size: 16px !important; } 
    .stButton>button { font-size: 15px !important; } 
    .stNumberInput, .stSelectbox { font-size: 15px !important; } 

    /* 手机竖屏样式 */
    @media only screen and (max-width: 768px) and (orientation: portrait) {
        .stMarkdown h3 {
            display: none !important;
        }
        .stMarkdown h4 {
            font-size: 15px !important;
            font-weight: normal !important;
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }
    }

    @media only screen and (max-width: 1024px) {
        .block-container { padding: 10px !important; max-width: 100% !important; }
        img, .stPyplot { max-height: 65vh !important; object-fit: contain !important; }
    }
    </style> 
    """,
        unsafe_allow_html=True,
    )

    # 手机顶部按钮（修复完成！）
    if mobile_view:
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
        if st.button("📘 实验原理", use_container_width=True):
            show_principle_dialog()
        if st.button("📖 使用说明", use_container_width=True):
            show_guide_dialog()
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1])

    if "initialized" not in st.session_state:
        st.session_state.mode = "demo"
        st.session_state.h = DEFAULT_H
        st.session_state.wl = DEFAULT_WL
        st.session_state.fringe_count = 0
        st.session_state.pos_data = {n: None for n in RECORD_N}
        st.session_state.pos_data[0] = add_measurement_error(OPTICAL_OFFSET_MM)
        st.session_state.usage_records_demo_export = []
        st.session_state.usage_records_sim_complete = []
        st.session_state.reset_counter = 0
        st.session_state.experiment_completed = False
        st.session_state.initialized = True
        st.session_state.gif_click_count = 0
        st.session_state.lock_wavelength = False
    if "need_reset" not in st.session_state:
        st.session_state.need_reset = False

    with col_right:
        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

        st.markdown("### 演示模式")
        col_demo, col_reset_demo = st.columns(2)
        with col_demo:
            if st.button(
                    "开始演示",
                    width="stretch",
                    type="primary" if st.session_state.mode == "demo" else "secondary",
            ):
                st.session_state.mode = "demo"
                reset_all_to_default()
                st.rerun()
        with col_reset_demo:
            demo_disabled = st.session_state.mode != "demo"
            if st.button("演示重置", width="stretch", disabled=demo_disabled):
                st.session_state.mode = "demo"
                reset_all_to_default()
                st.rerun()

        demo_enabled = st.session_state.mode == "demo"
        if demo_enabled:
            col_lbl, col_ctrl = st.columns([1, 3])
            with col_lbl:
                st.markdown("#### 镜片间距")
            with col_ctrl:
                actual_mm = OPTICAL_OFFSET_MM + st.session_state.h / 1e6
                h_mm_input = st.number_input(
                    "镜片间距(mm)",
                    min_value=2.50000,
                    max_value=2.50500,
                    value=actual_mm,
                    step=0.00010,
                    format="%.6f",
                    label_visibility="collapsed",
                    key=f"h_mm_demo_{st.session_state.reset_counter}",
                )
                new_h_nm = (h_mm_input - OPTICAL_OFFSET_MM) * 1e6
                if abs(new_h_nm - st.session_state.h) > 1:
                    st.session_state.h = new_h_nm
                    st.session_state.fringe_count = calc_fringe_count_from_h(
                        new_h_nm, st.session_state.wl
                    )
                    st.rerun()
            col_lbl, col_ctrl = st.columns([1, 3])
            with col_lbl:
                st.markdown("#### 光源设定")
            with col_ctrl:
                wl_options = list(wave_dict.keys())
                current_index = wl_options.index(st.session_state.wl)
                wl = st.selectbox(
                    "光源选择",
                    wl_options,
                    index=current_index,
                    label_visibility="collapsed",
                    key=f"wl_demo_{st.session_state.reset_counter}",
                )
                if wl != st.session_state.wl:
                    st.session_state.wl = wl
                    st.session_state.fringe_count = calc_fringe_count_from_h(
                        st.session_state.h, wl
                    )
                    st.rerun()

        st.markdown("#### 演示结果导出")
        col_png, col_gif = st.columns(2)
        demo_export_remaining = get_demo_export_remaining()
        export_disabled = not demo_enabled or demo_export_remaining <= 0
        with col_png:
            btn_png_text = f"导出PNG" + (
                f" ({demo_export_remaining}次)" if demo_export_remaining > 0 else " (已用完)"
            )
            if st.button(btn_png_text, width="stretch", disabled=export_disabled):
                if check_demo_export_quota():
                    consume_demo_export_quota()
                    i_intensity, cmap = calculate_data(
                        K_CONST, st.session_state.h, wave_dict[st.session_state.wl]
                    )
                    fig_temp = plot_interference(i_intensity, cmap, st.session_state.h)
                    buf = io.BytesIO()
                    fig_temp.savefig(buf, format="png", bbox_inches="tight", dpi=130)
                    buf.seek(0)
                    img = Image.open(buf).convert("RGB")
                    img = add_watermark(img, K_CONST, st.session_state.h, st.session_state.wl)
                    o = io.BytesIO()
                    img.save(o, format="PNG")
                    plt.close(fig_temp)
                    st.session_state.png_data = o.getvalue()
                    st.success(f"✅ PNG已生成！剩余次数：{get_demo_export_remaining()}")
                    st.rerun()
                else:
                    st.error(f"❌ 导出次数已用完！")
            if "png_data" in st.session_state and demo_enabled:
                st.download_button(
                    "⬇️ 下载PNG",
                    st.session_state.png_data,
                    f"干涉_{st.session_state.h:.0f}nm.png",
                    "image/png",
                    width="stretch",
                )

        with col_gif:
            btn_gif_text = f"生成GIF" + (
                f" ({demo_export_remaining}次)" if demo_export_remaining > 0 else " (已用完)"
            )
            if st.button(btn_gif_text, width="stretch", disabled=export_disabled):
                if check_demo_export_quota():
                    consume_demo_export_quota()
                    try:
                        import imageio
                        lamd_val = wave_dict[st.session_state.wl]
                        with st.spinner("生成中..."):
                            frames = []
                            for hi in np.linspace(20, 200, 10):
                                fi, cmapi = calculate_data(K_CONST, hi, lamd_val)
                                fg = plot_interference(fi, cmapi, hi)
                                b = io.BytesIO()
                                fg.savefig(b, format="png", bbox_inches="tight", dpi=100)
                                b.seek(0)
                                im = Image.open(b).convert("RGB")
                                im = add_watermark(im, K_CONST, hi, st.session_state.wl)
                                frames.append(im)
                                plt.close(fg)
                            bg = io.BytesIO()
                            imageio.mimsave(bg, frames, format="GIF", duration=0.4)
                            st.session_state.gif_data = bg.getvalue()
                        st.success(f"✅ GIF已生成！剩余次数：{get_demo_export_remaining()}")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"GIF生成失败：{e}")
                else:
                    st.error(f"❌ 导出次数已用完！")
            if "gif_data" in st.session_state and demo_enabled:
                st.download_button(
                    "⬇️ 下载GIF",
                    st.session_state.gif_data,
                    "干涉动画.gif",
                    "image/gif",
                    width="stretch",
                )

        st.markdown("### 实验模式")
        col_start, col_reset_sim = st.columns(2)
        with col_start:
            if st.button(
                    "开始实验",
                    width="stretch",
                    type="primary" if st.session_state.mode == "sim" else "secondary",
            ):
                st.session_state.mode = "sim"
                reset_all_to_default()
                st.rerun()
        with col_reset_sim:
            sim_disabled = st.session_state.mode != "sim"
            if st.button("实验重置", width="stretch", disabled=sim_disabled):
                st.session_state.mode = "sim"
                reset_all_to_default()
                st.rerun()

        sim_enabled = st.session_state.mode == "sim"
        if sim_enabled:
            col_lbl_sim, col_ctrl_sim = st.columns([1, 3])
            with col_lbl_sim:
                st.markdown("#### 光源设定")
            with col_ctrl_sim:
                wl_options = list(wave_dict.keys())
                current_index = wl_options.index(st.session_state.wl)
                # 锁定光源：采数后禁止修改
                wl = st.selectbox(
                    "光源选择",
                    wl_options,
                    index=current_index,
                    label_visibility="collapsed",
                    key=f"wl_sim_{st.session_state.reset_counter}",
                    disabled=st.session_state.lock_wavelength
                )
                if wl != st.session_state.wl and not st.session_state.lock_wavelength:
                    st.session_state.wl = wl
                    st.session_state.fringe_count = calc_fringe_count_from_h(
                        st.session_state.h, wl
                    )
                    force_update_pos_data()
                    st.session_state.experiment_completed = False
                    st.rerun()

        if sim_enabled:
            col_btn, col_val = st.columns([0.6, 0.4])
            is_fringe_limit_reached = st.session_state.fringe_count >= MAX_FRINGE_COUNT
            experiment_done = st.session_state.get("experiment_completed", False)
            with col_btn:
                btn_disabled = is_fringe_limit_reached or experiment_done
                btn_text = "冒出50个条纹"
                if is_fringe_limit_reached:
                    btn_text = "已达上限(350条)"
                elif experiment_done:
                    btn_text = "实验已完成"
                if st.button(btn_text, width="stretch", disabled=btn_disabled):
                    # 第一次点击就锁定光源
                    st.session_state.lock_wavelength = True
                    st.session_state.gif_click_count += 1
                    current_click = st.session_state.gif_click_count
                    play_fringe_animation(
                        st.session_state.fringe_count,
                        current_click,
                        st.session_state.wl,
                    )
                    st.session_state.fringe_count += 50
                    st.session_state.h = calc_h_from_fringe_count(
                        st.session_state.fringe_count, st.session_state.wl
                    )
                    update_pos_data()
                    if current_click >= 5:
                        st.session_state.gif_click_count = 0
                        data_complete = all(
                            st.session_state.pos_data[n] is not None for n in RECORD_N
                        )
                        if data_complete and not st.session_state.experiment_completed:
                            if check_sim_complete_quota():
                                consume_sim_complete_quota()
                                st.session_state.experiment_completed = True
                    st.rerun()
            with col_val:
                sim_remaining = get_sim_complete_remaining()
                quota_info = f"(剩余{sim_remaining}次)" if sim_remaining > 0 else ""
                st.markdown(
                    f"<div style='font-size:13px;'>移动量：{st.session_state.h:.1f} nm<br>条纹数：{st.session_state.fringe_count} / {MAX_FRINGE_COUNT} {quota_info}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("#### 实验数据预览")
        if sim_enabled:
            k_row = RECORD_N
            h_row = []
            for n in RECORD_N:
                if st.session_state.pos_data[n] is not None:
                    h_row.append(f"{st.session_state.pos_data[n]:.6f}")
                else:
                    h_row.append("")
            df_preview = pd.DataFrame([h_row], columns=[str(x) for x in k_row])
            df_preview.index = ["镜片间距 d (mm)"]
            st.dataframe(df_preview, width="stretch", hide_index=False)
        else:
            st.info("请切换至【开始实验】模式")

        st.markdown("<br>", unsafe_allow_html=True)
        data_complete = all(st.session_state.pos_data[n] is not None for n in RECORD_N)
        if sim_enabled and data_complete:
            if st.button("📊 不确定度", type="primary", width="stretch"):
                show_uncertainty_dialog()
        else:
            st.button("📊 不确定度", disabled=True, width="stretch")

    with col_left:
        current_wavelength = wave_dict[st.session_state.wl]
        i_final, cmap_final = calculate_data(
            K_CONST, st.session_state.h, current_wavelength
        )
        fig_final = plot_interference(i_final, cmap_final, st.session_state.h)
        st.pyplot(fig_final, use_container_width=True)
        plt.close(fig_final)

    if st.session_state.need_reset:
        st.session_state.need_reset = False


if __name__ == "__main__":
    main()