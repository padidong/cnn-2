"""
Waste Detection App - ระบบตรวจจับและจำแนกขยะ
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd

# ตรวจสอบและ import dependencies อย่างปลอดภัย
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV ไม่พร้อมใช้งาน")

# นำเข้าไฟล์ที่แยกไว้
try:
    from config import (
        CLASSES, WASTE_CATEGORIES, CSS_STYLES, HIDE_STREAMLIT_STYLE, 
        DEFAULT_CONFIDENCE, DEFAULT_IOU, SUPPORTED_IMAGE_TYPES
    )
    from utils import (
        find_model_files, load_model, process_detections, 
        calculate_environmental_score, generate_recommendations, create_summary_table
    )
    from ui_components import (
        render_header, render_waste_info_section, render_sidebar_controls,
        render_statistics_cards, render_results_table, render_charts,
        render_recommendations_section, render_detailed_results, 
        render_help_section, render_footer
    )
except ImportError as e:
    st.error(f"❌ ไม่สามารถโหลดไฟล์ที่จำเป็นได้: {e}")
    st.stop()

def convert_image_for_processing(image):
    """แปลงรูปภาพสำหรับการประมวลผลโดยไม่ใช้ cv2"""
    # แปลง PIL Image เป็น numpy array
    image_array = np.array(image)
    
    # ถ้าเป็นรูป RGBA แปลงเป็น RGB
    if len(image_array.shape) == 3 and image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    return image_array

def process_yolo_results(results):
    """ประมวลผลผลลัพธ์ YOLO โดยไม่ใช้ cv2"""
    try:
        # ลองใช้ฟังก์ชัน plot แบบปกติก่อน
        if CV2_AVAILABLE:
            annotated_image = results[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        else:
            # ถ้าไม่มี cv2 ให้ใช้รูปต้นฉบับ
            annotated_image = np.array(results[0].orig_img)
            if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                # แปลง BGR เป็น RGB ถ้าจำเป็น
                annotated_image = annotated_image[:, :, ::-1]
        
        return annotated_image
    except Exception as e:
        st.warning(f"ไม่สามารถแสดงรูปที่มี annotation ได้: {e}")
        # ใช้รูปต้นฉบับแทน
        try:
            return np.array(results[0].orig_img)[:, :, ::-1]
        except:
            # หากทุกอย่างล้มเหลว ให้ส่งคืน None
            return None

def main():
    """ฟังก์ชันหลักของแอปพลิเคชัน"""
    
    # ตั้งค่าหน้าเว็บ
    st.set_page_config(
        page_title="Waste Detection App",
        page_icon="🗂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ใช้ CSS
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    # แสดงหัวเรื่อง
    render_header()
    
    # แถบควบคุมด้านข้าง
    controls = render_sidebar_controls()
    
    # ส่วนเลือกโมเดล
    with st.sidebar:
        st.markdown("### 🤖 เลือกโมเดล")
        model_files = find_model_files()
        
        if model_files:
            selected_model = st.selectbox(
                "โมเดลที่มี:",
                model_files,
                help="เลือกโมเดล .pt ที่ต้องการใช้"
            )
        else:
            st.warning("ไม่พบไฟล์โมเดล .pt")
            selected_model = st.text_input(
                "Path โมเดล:",
                value="yolov8n.pt",
                help="ระบุ path ของโมเดลที่ฝึกแล้ว"
            )
    
    # โหลดโมเดล
    if selected_model:
        model, model_status = load_model(selected_model)
    else:
        model, model_status = None, "ไม่ได้เลือกโมเดล"
    
    with st.sidebar:
        if model:
            st.success(model_status)
        else:
            st.error(model_status)
    
    # แสดงข้อมูลการจำแนกขยะ
    render_waste_info_section()
    
    # ส่วนอัปโหลดรูป
    st.markdown('<h2 class="sub-header">📤 อัปโหลดรูปภาพขยะ</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพขยะที่ต้องการจำแนกประเภท",
        type=SUPPORTED_IMAGE_TYPES,
        help=f"รองรับไฟล์รูปภาพ: {', '.join([ext.upper() for ext in SUPPORTED_IMAGE_TYPES])}"
    )
    
    if uploaded_file is not None:
        if model is None:
            st.error("❌ กรุณาเลือกโมเดลที่ถูกต้องก่อนใช้งาน")
            return
            
        # แสดงรูปต้นฉบับ
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 รูปต้นฉบับ")
            image = Image.open(uploaded_file)
            st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)
        
        # ปุ่มเริ่มประมวลผล
        if st.button("🔍 เริ่มตรวจจับและจำแนกขยะ", type="primary"):
            with st.spinner("กำลังประมวลผล..."):
                try:
                    # แปลงรูปสำหรับการประมวลผล
                    image_array = convert_image_for_processing(image)
                    
                    # ทำการตรวจจับด้วย YOLO
                    results = model(
                        image_array, 
                        conf=controls["confidence"], 
                        iou=controls["iou"]
                    )
                    
                    # ประมวลผลรูปผลลัพธ์
                    annotated_image = process_yolo_results(results)
                    
                    with col2:
                        st.markdown("### 🎯 ผลการตรวจจับ")
                        if annotated_image is not None:
                            st.image(annotated_image, caption="รูปที่ตรวจจับแล้ว", use_column_width=True)
                        else:
                            st.image(image, caption="รูปต้นฉบับ (ไม่สามารถแสดง annotation ได้)", use_column_width=True)
                    
                    # ประมวลผลการตรวจจับ
                    class_counts, category_counts, detailed_data = process_detections(
                        results, controls["confidence"], controls["iou"]
                    )
                    
                    if class_counts:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 ผลการจำแนกขยะ")
                        
                        total_detections = sum([data["count"] for data in class_counts.values()])
                        
                        # แสดงสถิติ
                        render_statistics_cards(category_counts, total_detections)
                        
                        # แสดงตารางผลลัพธ์
                        results_data = create_summary_table(class_counts)
                        render_results_table(results_data)
                        
                        # แสดงกราฟ
                        render_charts(class_counts, category_counts)
                        
                        # คำแนะนำและคะแนนสิ่งแวดล้อม
                        recommendations = generate_recommendations(category_counts)
                        eco_score, eco_message = calculate_environmental_score(
                            category_counts, total_detections
                        )
                        render_recommendations_section(recommendations, eco_score, eco_message)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # ข้อมูลรายละเอียด
                        render_detailed_results(detailed_data)
                        
                    else:
                        st.warning("⚠️ ไม่พบขยะในรูปภาพนี้")
                        st.info("💡 ลองปรับค่า Confidence Threshold ให้ต่ำลง หรือใช้รูปภาพที่ชัดเจนขึ้น")
                        
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")
                    st.info("💡 โปรดตรวจสอบรูปแบบของรูปภาพและโมเดลที่ใช้")
                    st.exception(e)  # แสดงรายละเอียด error เพิ่มเติม
    
    # ส่วนช่วยเหลือ
    render_help_section()
    
    # ส่วนท้าย
    render_footer()

if __name__ == "__main__":
    main()