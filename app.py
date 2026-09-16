import datetime
import io
import os
import sqlite3
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import requests
import streamlit as st
import torch
from ultralytics import YOLO

# ---------------------------------------------------------
# 【頁面配置與自訂 CSS 工業風主題】
# ---------------------------------------------------------
st.set_page_config(
    page_title="PCB 智能缺陷檢測雲端系統",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .css-card {
        background-color: #1a1c23;
        border: 1px solid #2d313e;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    section[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #2d313e;
    }
    </style>
""",
    unsafe_allow_html=True,
)

DB_PATH = "pcb_system.db"
MODEL_PT_PATH = "best.pt"
MODEL_ONNX_PATH = "best.onnx"


# ---------------------------------------------------------
# 【LINE / Webhook / Telegram 告警模組】
# ---------------------------------------------------------
def send_batch_alert_notification(
    webhook_url, total_files, pass_count, fail_count, total_defects, batch_results
):
  """發送詳細的批次檢測異常通報至 Telegram"""
  if not webhook_url:
    return False

  yield_rate = (pass_count / total_files) * 100 if total_files > 0 else 0

  # 組裝逐張瑕疵說明
  defect_details_lines = []
  for item in batch_results:
    if item["defect_count"] > 0:
      item_details = ", ".join(
          [f"{k}: {v}個" for k, v in item["defect_counts"].items()]
      )
      defect_details_lines.append(f" 📄 {item['filename']}\n    └ {item_details}")

  details_text = "\n".join(defect_details_lines)

  alert_text = (
      f"🚨 [PCB 產線品質異常警報]\n"
      f"📦 批次良率：{yield_rate:.1f}% (合格 {pass_count} / 不良 {fail_count} / 總數"
      f" {total_files})\n"
      f"⚠️ 瑕疵總數：{total_defects} 處\n\n"
      f"🔍 各板瑕疵詳細清單：\n{details_text}\n\n"
      f"⏰ 時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
  )

  if "telegram.org" in webhook_url:
    try:
      res = requests.post(webhook_url, data={"text": alert_text}, timeout=5)
      return res.status_code == 200
    except Exception:
      return False
  else:
    try:
      res = requests.post(webhook_url, json={"content": alert_text}, timeout=5)
      return res.status_code in [200, 204]
    except Exception:
      return False


# ---------------------------------------------------------
# 【PDF 報告生成模組】
# ---------------------------------------------------------
def sanitize_text(text):
  return str(text).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(records_df):
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font("Helvetica", "B", 16)
  pdf.cell(
      0, 10, "PCB Automated Optical Inspection (AOI) Report", ln=True, align="C"
  )
  pdf.set_font("Helvetica", "", 10)
  pdf.cell(
      0,
      10,
      f"Report Generated Date:"
      f" {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
      ln=True,
      align="C",
  )
  pdf.ln(5)

  pdf.set_font("Helvetica", "B", 10)
  pdf.set_fill_color(30, 144, 255)
  pdf.set_text_color(255, 255, 255)

  col_widths = [15, 45, 40, 25, 35, 30]
  headers = ["ID", "Timestamp", "Filename", "Defects", "Primary Defect", "Engine"]

  for i, h in enumerate(headers):
    pdf.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
  pdf.ln()

  pdf.set_font("Helvetica", "", 9)
  pdf.set_text_color(0, 0, 0)

  defect_zh_to_en = {
      "漏孔": "missing_hole",
      "鼠咬": "mouse_bite",
      "突起": "spur",
      "短路": "short",
      "斷路": "open_circuit",
      "雜銅": "spurious_copper",
      "無": "None",
  }

  for _, row in records_df.iterrows():
    raw_defect = str(row["主要瑕疵"])
    primary_en = defect_zh_to_en.get(raw_defect, raw_defect)

    pdf.cell(col_widths[0], 7, sanitize_text(row["ID"]), border=1, align="C")
    pdf.cell(col_widths[1], 7, sanitize_text(row["時間"]), border=1, align="C")
    clean_filename = sanitize_text(row["圖片名稱"])[:18]
    pdf.cell(col_widths[2], 7, clean_filename, border=1)
    pdf.cell(
        col_widths[3], 7, sanitize_text(row["瑕疵數量"]), border=1, align="C"
    )
    pdf.cell(col_widths[4], 7, sanitize_text(primary_en), border=1, align="C")
    pdf.cell(
        col_widths[5], 7, sanitize_text(row["推論引擎"]), border=1, align="C"
    )
    pdf.ln()

  return pdf.output()


# ---------------------------------------------------------
# 【SQLite 資料庫模組】
# ---------------------------------------------------------
def init_db():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            defect_count INTEGER,
            primary_defect TEXT,
            engine_used TEXT,
            inference_time_ms REAL
        )
    """)
  conn.commit()
  conn.close()


def save_record(
    filename, defect_count, primary_defect, engine_used, infer_time
):
  now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO detection_history (timestamp, filename, defect_count, primary_defect, engine_used, inference_time_ms)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
      (now, filename, defect_count, primary_defect, engine_used, infer_time),
  )
  conn.commit()
  conn.close()


def fetch_all_records():
  conn = sqlite3.connect(DB_PATH)
  df = pd.read_sql_query(
      """
        SELECT 
            id AS "ID",
            timestamp AS "時間", 
            filename AS "圖片名稱", 
            defect_count AS "瑕疵數量", 
            primary_defect AS "主要瑕疵",
            engine_used AS "推論引擎",
            round(inference_time_ms, 2) AS "耗時 (ms)"
        FROM detection_history
        ORDER BY id DESC
    """,
      conn,
  )
  conn.close()
  return df


def clear_all_records():
  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  cursor.execute("DELETE FROM detection_history")
  conn.commit()
  conn.close()


init_db()


# ---------------------------------------------------------
# 【模型載入】
# ---------------------------------------------------------
@st.cache_resource
def load_pt_model():
  return YOLO(MODEL_PT_PATH) if os.path.exists(MODEL_PT_PATH) else None


@st.cache_resource
def load_onnx_model():
  return YOLO(MODEL_ONNX_PATH) if os.path.exists(MODEL_ONNX_PATH) else None


# ---------------------------------------------------------
# 【側邊欄控制台】
# ---------------------------------------------------------
st.sidebar.markdown("### 🧭 系統導覽控制台")
page = st.sidebar.selectbox(
    "請選擇功能頁面：",
    ["📸 即時瑕疵檢測中心", "📋 歷史檢測紀錄看板", "📊 瑕疵數據統計分析"],
)
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ 推論引擎配置")
engine_choice = st.sidebar.radio(
    "選擇推論引擎 (Inference Engine)：",
    ("PyTorch (.pt)", "ONNX Runtime (.onnx)"),
    help="ONNX 引擎通常具備更低的記憶體開銷與更高的邊緣端推論速度。",
)

conf_threshold = st.sidebar.slider(
    "判定信心度閾值 (Confidence)",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 比例尺換算校正")
mm_per_pixel = st.sidebar.number_input(
    "比例尺係數 (mm / Pixel)",
    value=0.025,
    step=0.005,
    format="%.4f",
    help="設定單一像素對應的實體長度，用於計算 PCB 瑕疵物理尺寸。",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚨 產線異常即時通報設定")
enable_notify = st.sidebar.checkbox("啟用 Telegram 異常通報", value=False)
webhook_url = ""
if enable_notify:
  webhook_url = st.sidebar.text_input(
      "Telegram API 網址",
      value="",
      type="password",
      help="貼上包含 Token 與 Chat ID 的完整 sendMessage API 網址",
  )

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background-color: #1e2330; padding: 10px; border-radius: 8px; border: 1px solid #00d2ff; text-align: center;">
        <span style="color: #00ff88; font-weight: bold;">🟢 系統狀態：即時線上</span><br>
        <small style="color: #888;">AI Model: YOLO11 Detection</small>
    </div>
""",
    unsafe_allow_html=True,
)

defects_zh = {
    "mouse_bite": "鼠咬 (mouse_bite)",
    "spur": "突起 (spur)",
    "missing_hole": "漏孔 (missing_hole)",
    "short": "短路 (short)",
    "open_circuit": "斷路 (open_circuit)",
    "spurious_copper": "雜銅 (spurious_copper)",
}

# ---------------------------------------------------------
# 【頁面 1：即時瑕疵檢測中心】
# ---------------------------------------------------------
if page == "📸 即時瑕疵檢測中心":
  st.markdown(
      '<div class="main-header">🔍 PCB 智能自動化瑕疵檢測控制台</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      "工業級 Vision AI 實時邊緣端檢測系統 ｜ 支援批次檢測與產線良率計算 ｜ YOLO11"
      " 引擎"
  )
  st.markdown("---")

  active_model = (
      load_pt_model() if engine_choice == "PyTorch (.pt)" else load_onnx_model()
  )
  engine_name = "PyTorch" if engine_choice == "PyTorch (.pt)" else "ONNX Runtime"

  if active_model is None:
    st.error(
        f"❌ 無法載入模型，請確認專案目錄下是否有"
        f" `{MODEL_PT_PATH if engine_choice=='PyTorch (.pt)' else MODEL_ONNX_PATH}`"
        " 檔案！"
    )
  else:
    uploaded_files = st.file_uploader(
        "📸 請上傳待檢測 PCB 影像檔 (支援單張或多張批次上傳)...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
      st.markdown("### 📊 系統即時指標與預覽說明")
      m1, m2, m3, m4 = st.columns(4)
      m1.metric("當前推論引擎", engine_name)
      m2.metric("信心度閾值", f"{conf_threshold:.2f}")
      m3.metric("實體比例尺", f"{mm_per_pixel:.4f} mm/px")
      m4.metric(
          "即時通報狀態",
          "已啟用" if (enable_notify and webhook_url) else "未啟用",
      )

      st.markdown(
          """
            <div class="css-card">
                <h4>💡 工業 AOI 檢測操作指南：</h4>
                <ol>
                    <li><b>批次檢測</b>：可一次選取多張 PCB 圖片上傳，系統將自動計算整批次之<b>良率 (Yield Rate)</b>。</li>
                    <li><b>手動觸發通報</b>：檢出瑕疵後點擊通報按鈕，會詳細彙整每一張 PCB 的圖片檔名與具體瑕疵發送到 Telegram。</li>
                    <li><b>尺寸量測</b>：系統會將像素距離轉換為實體物理長寬 ($mm$)。</li>
                </ol>
            </div>
            """,
          unsafe_allow_html=True,
      )

    else:
      total_files = len(uploaded_files)
      pass_count = 0
      fail_count = 0

      st.markdown(f"### 📦 本次批次檢測任務（共 {total_files} 張 PCB）")
      progress_bar = st.progress(0)

      batch_results = []
      total_defects_in_batch = 0

      for file_idx, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        start_time = time.time()
        results = active_model.predict(
            source=image, imgsz=640, conf=conf_threshold
        )
        infer_time = (time.time() - start_time) * 1000

        boxes = results[0].boxes
        defect_cnt = len(boxes)
        total_defects_in_batch += defect_cnt

        defect_counts = {}
        for box in boxes:
          cls_name = active_model.names[int(box.cls[0])]
          zh_name = defects_zh.get(cls_name, cls_name)
          defect_counts[zh_name] = defect_counts.get(zh_name, 0) + 1

        primary_defect = (
            max(defect_counts, key=defect_counts.get) if defect_counts else "無"
        )

        if defect_cnt == 0:
          pass_count += 1
        else:
          fail_count += 1

        batch_results.append({
            "filename": file.name,
            "image": image,
            "results": results,
            "defect_count": defect_cnt,
            "primary_defect": primary_defect,
            "infer_time": infer_time,
            "boxes": boxes,
            "defect_counts": defect_counts,
        })

        progress_bar.progress((file_idx + 1) / total_files)

      yield_rate = (pass_count / total_files) * 100
      y1, y2, y3, y4 = st.columns(4)
      y1.metric("總檢測數", f"{total_files} 張")
      y2.metric("合格品 (PASS)", f"{pass_count} 張", delta=f"{pass_count}")
      y3.metric(
          "不良品 (FAIL)",
          f"{fail_count} 張",
          delta=f"-{fail_count}",
          delta_color="inverse",
      )
      y4.metric("批次良率 (Yield Rate)", f"{yield_rate:.1f}%")

      st.markdown("---")
      st.markdown("### 🔍 個別 PCB 檢測詳情與 ROI 特寫")

      for item in batch_results:
        with st.expander(
            f"📄 圖片：{item['filename']} ｜ 狀態："
            f" {'🔴 FAIL (' + str(item['defect_count']) + ' 處瑕疵)' if item['defect_count'] > 0 else '🟢 PASS'}",
            expanded=(total_files == 1),
        ):
          col1, col2 = st.columns(2)
          with col1:
            st.image(
                item["image"],
                caption=f"原始影像: {item['filename']}",
                use_container_width=True,
            )
          with col2:
            res_plotted = item["results"][0].plot()
            st.image(
                res_plotted,
                channels="BGR",
                caption="YOLO11 AI 檢測標記",
                use_container_width=True,
            )

          boxes = item["boxes"]
          if len(boxes) > 0:
            st.markdown("**🔍 瑕疵 ROI 區域與實體尺寸換算：**")
            crop_cols = st.columns(min(len(boxes), 4))
            for idx, box in enumerate(boxes):
              cls_name = active_model.names[int(box.cls[0])]
              zh_name = defects_zh.get(cls_name, cls_name)
              conf = float(box.conf[0])

              xyxy = box.xyxy[0].cpu().numpy().astype(int)
              w_mm = (xyxy[2] - xyxy[0]) * mm_per_pixel
              h_mm = (xyxy[3] - xyxy[1]) * mm_per_pixel

              img_w, img_h = item["image"].size
              pad = 30
              cropped_img = item["image"].crop((
                  max(0, xyxy[0] - pad),
                  max(0, xyxy[1] - pad),
                  min(img_w, xyxy[2] + pad),
                  min(img_h, xyxy[3] + pad),
              ))

              with crop_cols[idx % 4]:
                st.image(
                    cropped_img,
                    caption=(
                        f"{zh_name}\n({conf:.1%})\n尺寸:"
                        f" {w_mm:.2f}×{h_mm:.2f} mm"
                    ),
                    width=160,
                )

      st.markdown("---")
      btn_col1, btn_col2 = st.columns(2)

      with btn_col1:
        if st.button("💾 將整批檢測數據寫入 SQLite 資料庫", type="primary"):
          for item in batch_results:
            save_record(
                item["filename"],
                item["defect_count"],
                item["primary_defect"],
                engine_name,
                item["infer_time"],
            )
          st.success(
              f"✅ 已成功將 {len(batch_results)} 筆批次檢測紀錄寫入 SQLite"
              " 資料庫！"
          )

      with btn_col2:
        if fail_count > 0 and enable_notify and webhook_url:
          if st.button("🚨 手動傳送 Telegram 異常通報", type="secondary"):
            success = send_batch_alert_notification(
                webhook_url,
                total_files,
                pass_count,
                fail_count,
                total_defects_in_batch,
                batch_results,
            )
            if success:
              st.success("✅ 已成功傳送詳細警報至 Telegram！")
            else:
              st.error(
                  "❌ 警報傳送失敗，請確認網址格式或 Bot 是否有按 Start。"
              )

# ---------------------------------------------------------
# 【頁面 2：歷史檢測紀錄看板】
# ---------------------------------------------------------
elif page == "📋 歷史檢測紀錄看板":
  st.markdown(
      '<div class="main-header">📋 歷史檢測紀錄管理看板</div>',
      unsafe_allow_html=True,
  )
  st.markdown("---")
  db_df = fetch_all_records()

  col_dl1, col_dl2, col_clr = st.columns([1.5, 1.5, 3])
  with col_dl1:
    if not db_df.empty:
      csv_data = db_df.to_csv(index=False, encoding="utf-8-sig")
      st.download_button(
          label="📥 匯出 CSV 數據集",
          data=csv_data,
          file_name=f"pcb_inspection_{datetime.date.today()}.csv",
          mime="text/csv",
      )

  with col_dl2:
    if not db_df.empty:
      try:
        pdf_bytes = generate_pdf_report(db_df)
        st.download_button(
            label="📄 下載 PDF 工業檢測報告",
            data=bytes(pdf_bytes),
            file_name=f"PCB_AOI_Report_{datetime.date.today()}.pdf",
            mime="application/pdf",
        )
      except Exception as e:
        st.error(f"⚠️ 生成 PDF 時發生錯誤: {e}")

  with col_clr:
    if not db_df.empty:
      if st.button("🗑️ 清空歷史資料庫紀錄"):
        clear_all_records()
        st.rerun()

  st.dataframe(db_df, use_container_width=True)

  if not db_df.empty:
    avg_ms = db_df["耗時 (ms)"].dropna().mean()
    st.metric(
        label="📊 系統累計檢測總板數",
        value=len(db_df),
        delta=f"平均辨識耗時 {avg_ms:.1f} ms" if pd.notnull(avg_ms) else None,
    )

# ---------------------------------------------------------
# 【頁面 3：瑕疵數據統計分析】
# ---------------------------------------------------------
elif page == "📊 瑕疵數據統計分析":
  st.markdown(
      '<div class="main-header">📊 廠務瑕疵品質產能與柏拉圖分析</div>',
      unsafe_allow_html=True,
  )
  st.markdown("---")
  db_df = fetch_all_records()

  if not db_df.empty:
    defect_df = db_df[db_df["主要瑕疵"] != "無"]

    c1, c2 = st.columns(2)
    with c1:
      st.subheader("📈 歷史瑕疵檢出數量趨勢")
      st.line_chart(data=db_df, x="圖片名稱", y="瑕疵數量")

    with c2:
      st.subheader("🍩 主要瑕疵類型次數統計")
      if not defect_df.empty:
        counts = defect_df["主要瑕疵"].value_counts()
        st.bar_chart(counts)
      else:
        st.info("目前資料庫中尚無不良品紀錄。")

    st.markdown("---")
    st.subheader("📉 品管柏拉圖分析 (Pareto Quality Analysis)")
    st.caption(
        "80/20 法則：優先改善前 20% 的主要瑕疵類型，即可解決 80% 的產線不良率。"
    )

    if not defect_df.empty:
      pareto_df = defect_df["主要瑕疵"].value_counts().reset_index()
      pareto_df.columns = ["瑕疵類型", "數量"]
      pareto_df["累計百分比"] = (
          pareto_df["數量"].cumsum() / pareto_df["數量"].sum()
      ) * 100

      fig, ax1 = plt.subplots(figsize=(10, 4))
      plt.style.use("dark_background")
      fig.patch.set_facecolor("#1a1c23")
      ax1.set_facecolor("#1a1c23")

      ax1.bar(
          pareto_df["瑕疵類型"],
          pareto_df["數量"],
          color="#00d2ff",
          alpha=0.8,
          width=0.4,
      )
      ax1.set_ylabel("出現次數", color="#00d2ff")
      ax1.tick_params(axis="y", labelcolor="#00d2ff")

      ax2 = ax1.twinx()
      ax2.plot(
          pareto_df["瑕疵類型"],
          pareto_df["累計百分比"],
          color="#ff4b4b",
          marker="o",
          linewidth=2,
      )
      ax2.axhline(80, color="#00ff88", linestyle="--", alpha=0.7)
      ax2.set_ylabel("累計影響百分比 (%)", color="#ff4b4b")
      ax2.tick_params(axis="y", labelcolor="#ff4b4b")
      ax2.set_ylim(0, 110)

      st.pyplot(fig)
    else:
      st.info("尚無足夠數據生成柏拉圖。")
  else:
    st.info("資料庫為空，請先前往「即時瑕疵檢測中心」進行上傳與檢測。")