import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import os

# ตั้งค่า environment สำหรับประหยัด memory
os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ตรวจสอบและ import dependencies อย่างปลอดภัย
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV ไม่พร้อมใช้งาน - จะใช้การแสดงผลแบบพื้นฐาน")

try:
    import torch
    torch.set_num_threads(1)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("⚠️ PyTorch ไม่พร้อมใช้งาน")

# นำเข้าไฟล์ที่แยกไว้
try:
    from config import (
        CLASSES, WASTE_CATEGORIES, CSS_STYLES, HIDE_STREAMLIT_STYLE, 
        DEFAULT_CONFIDENCE, DEFAULT_IOU, SUPPORTED_IMAGE_TYPES
    )
    from utils import (
        find_model_files, load_model, process_detections, 
        calculate_environmental_score, generate_recommendations, 
        create_summary_table, cleanup_memory
    )
    from ui_components import (
        render_header, render_waste_info_section, render_sidebar_controls,
        render_statistics_cards, render_results_table, render_charts,
        render_recommendations_section, render_detailed_results, 
        render_help_section, render_footer
    )
except ImportError as e:
    st.error(f"❌ ไม่สามารถโหลดไฟล์ที่จำเป็นได้: {e}")
    st.info("💡 กรุณาตรวจสอบให้แน่ใจว่าไฟล์ config.py, utils.py และ ui_components.py อยู่ในโฟลเดอร์เดียวกัน")
    st.stop()

def convert_image_for_processing(image):
    """แปลงรูปภาพสำหรับการประมวลผลโดยไม่ใช้ cv2"""
    try:
        # แปลง PIL Image เป็น numpy array
        image_array = np.array(image)
        
        # ถ้าเป็นรูป RGBA แปลงเป็น RGB
        if len(image_array.shape) == 3 and image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        # ตรวจสอบขนาดรูป หากใหญ่เกินไปให้ย่อขนาด
        height, width = image_array.shape[:2]
        max_size = 1024
        
        if height > max_size or width > max_size:
            from PIL import Image
            image_pil = Image.fromarray(image_array)
            
            # คำนวณขนาดใหม่โดยรักษาอัตราส่วน
            if height > width:
                new_height = max_size
                new_width = int(width * max_size / height)
            else:
                new_width = max_size
                new_height = int(height * max_size / width)
            
            image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            image_array = np.array(image_pil)
            
            st.info(f"📏 ปรับขนาดรูปภาพจาก {width}x{height} เป็น {new_width}x{new_height}")
        
        return image_array
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการแปลงรูปภาพ: {e}")
        return np.array(image)

def process_model_results(results, original_image):
    """ประมวลผลผลลัพธ์จากโมเดลทุกประเภท"""
    try:
        if not results or len(results) == 0:
            st.warning("ไม่ได้รับผลลัพธ์จากโมเดล")
            return None
        
        result = results[0]
        
        # พยายามสร้างรูปที่มี annotation
        annotated_image = None
        
        # วิธีที่ 1: ใช้ฟังก์ชัน plot ของโมเดล (สำหรับ YOLO)
        if hasattr(result, 'plot') and CV2_AVAILABLE:
            try:
                annotated_image = result.plot()
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    # แปลง BGR เป็น RGB
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.success("✅ สร้างรูป annotation ด้วย OpenCV")
            except Exception as e:
                st.warning(f"ไม่สามารถใช้ OpenCV plot: {e}")
                annotated_image = None
        
        # วิธีที่ 2: ใช้รูปต้นฉบับจากโมเดล
        if annotated_image is None and hasattr(result, 'orig_img') and result.orig_img is not None:
            try:
                annotated_image = np.array(result.orig_img)
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    # แปลง BGR เป็น RGB ถ้าจำเป็น
                    annotated_image = annotated_image[:, :, ::-1]
                st.info("📷 ใช้รูปต้นฉบับจากโมเดล")
            except Exception as e:
                st.warning(f"ไม่สามารถใช้รูปจากโมเดล: {e}")
                annotated_image = None
        
        # วิธีที่ 3: วาด bounding boxes ด้วย PIL (fallback)
        if annotated_image is None:
            try:
                annotated_image = draw_detections_pil(original_image, result)
                if annotated_image is not None:
                    st.info("🎨 วาด bounding boxes ด้วย PIL")
            except Exception as e:
                st.warning(f"ไม่สามารถวาดด้วย PIL: {e}")
        
        # วิธีที่ 4: ใช้รูปต้นฉบับ (last resort)
        if annotated_image is None:
            annotated_image = np.array(original_image)
            st.warning("⚠️ ใช้รูปต้นฉบับ (ไม่มี annotation)")
        
        return annotated_image
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลลัพธ์: {e}")
        return np.array(original_image)

def draw_detections_pil(image, result):
    """วาด bounding boxes ด้วย PIL สำหรับกรณีที่ OpenCV ไม่พร้อมใช้งาน"""
    try:
        from PIL import Image as PILImage, ImageDraw, ImageFont
        
        # แปลงเป็น PIL Image
        if isinstance(image, np.ndarray):
            pil_image = PILImage.fromarray(image)
        else:
            pil_image = image.copy()
        
        draw = ImageDraw.Draw(pil_image)
        
        # ตรวจสอบว่ามี detections หรือไม่
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            detections = result.boxes
            
            # เตรียม font (ใช้ default ถ้าไม่มี)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            colors = ["red", "blue", "green", "orange", "purple", "yellow", 
                     "pink", "brown", "gray", "cyan", "magenta", "lime"]
            
            for i, detection in enumerate(detections):
                try:
                    # ข้อมูล detection
                    if hasattr(detection, 'cls'):
                        class_id = int(detection.cls[0])
                        confidence = float(detection.conf[0])
                        
                        # bounding box
                        if hasattr(detection.xyxy[0], 'cpu'):
                            bbox = detection.xyxy[0].cpu().numpy()
                        else:
                            bbox = detection.xyxy[0].numpy()
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # เลือกสีและชื่อคลาส
                        color = colors[class_id % len(colors)]
                        class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Class_{class_id}"
                        
                        # วาดกรอบ
                        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                        
                        # วาดข้อความ
                        text = f"{class_name}: {confidence:.2f}"
                        text_bbox = draw.textbbox((x1, y1-20), text, font=font) if font else (x1, y1-20, x1+100, y1)
                        
                        # วาดพื้นหลังข้อความ
                        draw.rectangle(text_bbox, fill=color)
                        draw.text((x1, y1-20), text, fill="white", font=font)
                        
                except Exception as e:
                    st.warning(f"ไม่สามารถวาด detection {i}: {e}")
                    continue
        
        return np.array(pil_image)
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการวาดด้วย PIL: {e}")
        return None

def display_model_info(model, model_path):
    """แสดงข้อมูลโมเดล"""
    try:
        model_name = os.path.basename(model_path)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        with st.expander("📊 ข้อมูลโมเดล"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ชื่อไฟล์", model_name)
            with col2:
                st.metric("ขนาดไฟล์", f"{file_size:.1f} MB")
            with col3:
                # ตรวจสอบประเภทโมเดล
                if hasattr(model, 'model_type'):
                    model_type = model.model_type
                else:
                    model_type = "YOLO"
                st.metric("ประเภท", model_type)
            
            # ข้อมูลเพิ่มเติม
            st.markdown(f"""
            **รายละเอียด:**
            - 📁 Path: `{model_path}`
            - 🎯 คลาสที่รองรับ: {len(CLASSES)} ประเภท
            - 💾 Memory: CPU Only (เพื่อประหยัด resources)
            """)
            
    except Exception as e:
        st.warning(f"ไม่สามารถแสดงข้อมูลโมเดลได้: {e}")

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
        
        # ค้นหาไฟล์โมเดล
        model_files = find_model_files()
        
        if model_files:
            selected_model = st.selectbox(
                "โมเดลที่มี:",
                model_files,
                help="เลือกโมเดล .pt ที่ต้องการใช้"
            )
        else:
            st.warning("⚠️ ไม่พบไฟล์โมเดล .pt ในโฟลเดอร์")
            st.info("""
            💡 วางไฟล์โมเดลในตำแหน่งต่อไปนี้:
            - `./your_model.pt`
            - `./models/your_model.pt`  
            - `./weights/your_model.pt`
            """)
            
            selected_model = st.text_input(
                "หรือระบุ Path โมเดล:",
                value="yolov8n.pt",
                help="ระบุ path ของโมเดลที่ฝึกแล้ว"
            )
        
        # ปุ่มรีเฟรชรายการโมเดล
        if st.button("🔄 รีเฟรชรายการโมเดล"):
            st.cache_data.clear()
            st.rerun()
    
    # โหลดโมเดล
    model = None
    model_status = "ไม่ได้เลือกโมเดล"
    
    if selected_model:
        with st.spinner(f"กำลังโหลดโมเดล {os.path.basename(selected_model)}..."):
            model, model_status = load_model(selected_model)
            
            # เคลียร์ memory หลังโหลดโมเดล
            cleanup_memory()
    
    with st.sidebar:
        if model:
            st.success(model_status)
            
            # แสดงข้อมูลโมเดล
            display_model_info(model, selected_model)
            
            # ปุ่มเคลียร์โมเดลออกจากแคช
            if st.button("🗑️ เคลียร์โมเดลจากแคช"):
                st.cache_resource.clear()
                cleanup_memory()
                st.success("เคลียร์แคชเรียบร้อย")
                st.rerun()
        else:
            st.error(model_status)
            
            if "ไม่พบไฟล์" in model_status:
                st.info("💡 ตรวจสอบ path ของโมเดลให้ถูกต้อง")
            elif "Dependencies" in model_status:
                st.info("💡 รอให้ Streamlit Cloud ติดตั้ง libraries เสร็จ")
    
    # แสดงข้อมูลการจำแนกขยะ
    render_waste_info_section()
    
    # ส่วนอัปโหลดรูป
    st.markdown('<h2 class="sub-header">📤 อัปโหลดรูปภาพขยะ</h2>', unsafe_allow_html=True)
    
    # เพิ่มตัวอย่างรูปภาพ
    with st.expander("💡 ตัวอย่างรูปภาพที่แนะนำ"):
        st.markdown("""
        **รูปภาพที่ให้ผลลัพธ์ดี:**
        - 📸 แสงเพียงพอและชัดเจน
        - 🎯 ขยะอยู่ในกรอบ
        - 📏 ไม่เบลอหรือเอียง
        - 🔍 ขนาดไฟล์ไม่เกิน 10MB
        
        **หลีกเลี่ยง:**
        - 🌃 รูปมืดหรือแสงน้อย
        - 🔄 รูปเบลอหรือไม่ชัด
        - 📐 ขยะอยู่มุมรูป
        - 🖼️ รูปที่มีสิ่งบดบัง
        """)
    
    uploaded_file = st.file_uploader(
        "เลือกรูปภาพขยะที่ต้องการจำแนกประเภท",
        type=SUPPORTED_IMAGE_TYPES,
        help=f"รองรับไฟล์รูปภาพ: {', '.join([ext.upper() for ext in SUPPORTED_IMAGE_TYPES])}"
    )
    
    if uploaded_file is not None:
        if model is None:
            st.error("❌ กรุณาเลือกโมเดลที่ถูกต้องก่อนใช้งาน")
            st.info("💡 ตรวจสอบให้แน่ใจว่าโมเดลโหลดสำเร็จแล้ว")
            return
            
        # แสดงรูปต้นฉบับ
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 รูปต้นฉบับ")
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)
                
                # แสดงข้อมูลรูปภาพ
                with st.expander("📊 ข้อมูลรูปภาพ"):
                    st.write(f"📏 ขนาด: {image.size[0]} x {image.size[1]} pixels")
                    st.write(f"🎨 โหมด: {image.mode}")
                    st.write(f"📁 ชนิดไฟล์: {image.format}")
                    
                    if hasattr(uploaded_file, 'size'):
                        file_size = uploaded_file.size / 1024  # KB
                        st.write(f"💾 ขนาดไฟล์: {file_size:.1f} KB")
                        
            except Exception as e:
                st.error(f"❌ ไม่สามารถเปิดรูปภาพได้: {e}")
                return
        
        # ปุ่มเริ่มประมวลผล
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            process_button = st.button(
                "🔍 เริ่มตรวจจับและจำแนกขยะ", 
                type="primary", 
                use_container_width=True
            )
        
        if process_button:
            with st.spinner("กำลังประมวลผล... โปรดรอสักครู่"):
                try:
                    # แปลงรูปสำหรับการประมวลผล
                    with st.status("เตรียมรูปภาพ...") as status:
                        image_array = convert_image_for_processing(image)
                        status.update(label="เตรียมรูปภาพเสร็จ", state="complete")
                    
                    # ทำการตรวจจับด้วยโมเดล
                    with st.status("กำลังวิเคราะห์ด้วย AI...") as status:
                        results = model(
                            image_array, 
                            conf=controls["confidence"], 
                            iou=controls["iou"]
                        )
                        status.update(label="วิเคราะห์เสร็จ", state="complete")
                    
                    # ประมวลผลรูปผลลัพธ์
                    with st.status("สร้างรูปผลลัพธ์...") as status:
                        annotated_image = process_model_results(results, image)
                        status.update(label="สร้างรูปเสร็จ", state="complete")
                    
                    with col2:
                        st.markdown("### 🎯 ผลการตรวจจับ")
                        if annotated_image is not None:
                            st.image(annotated_image, caption="รูปที่ตรวจจับแล้ว", use_column_width=True)
                        else:
                            st.image(image, caption="รูปต้นฉบับ (ไม่สามารถแสดง annotation ได้)", use_column_width=True)
                    
                    # ประมวลผลการตรวจจับ
                    with st.status("วิเคราะห์ผลลัพธ์...") as status:
                        class_counts, category_counts, detailed_data = process_detections(
                            results, controls["confidence"], controls["iou"]
                        )
                        status.update(label="วิเคราะห์ผลลัพธ์เสร็จ", state="complete")
                    
                    if class_counts:
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("### 📊 ผลการจำแนกขยะ")
                        
                        total_detections = sum([data["count"] for data in class_counts.values()])
                        
                        # แสดงข้อความสรุป
                        st.success(f"🎉 พบขยะทั้งหมด **{total_detections}** ชิ้น จาก **{len(class_counts)}** ประเภท")
                        
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
                        
                        # เคลียร์ memory หลังประมวลผล
                        cleanup_memory()
                        
                    else:
                        st.warning("⚠️ ไม่พบขยะในรูปภาพนี้")
                        
                        # แสดงคำแนะนำ
                        st.markdown("""
                        ### 💡 คำแนะนำ:
                        - **ลดค่า Confidence Threshold** ในแถบข้าง (ลองเริ่มที่ 0.3)
                        - **ใช้รูปภาพที่ชัดเจนขึ้น** มีแสงเพียงพอ
                        - **ตรวจสอบว่าขยะอยู่ในกรอบรูป** 
                        - **ลองใช้รูปภาพอื่น** ที่มีขยะชัดเจนกว่า
                        """)
                        
                        # แสดงรายละเอียด technical
                        with st.expander("🔧 รายละเอียดทางเทคนิค"):
                            st.json({
                                "Model": os.path.basename(selected_model),
                                "Confidence Threshold": controls["confidence"],
                                "IoU Threshold": controls["iou"],
                                "Image Size": f"{image.size[0]}x{image.size[1]}",
                                "Processed Size": f"{image_array.shape[1]}x{image_array.shape[0]}",
                                "Results": len(results) if results else 0
                            })
                        
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {e}")
                    
                    # แสดงข้อมูล debug
                    with st.expander("🔍 ข้อมูล Debug"):
                        st.exception(e)
                        st.json({
                            "Model Type": type(model).__name__,
                            "Model Path": selected_model,
                            "Image Mode": image.mode if 'image' in locals() else 'Unknown',
                            "Image Size": image.size if 'image' in locals() else 'Unknown'
                        })
                    
                    st.info("""
                    ### 🛠️ วิธีแก้ไขปัญหา:
                    1. **ลอง Refresh หน้าเว็บ** และโหลดโมเดลใหม่
                    2. **ตรวจสอบไฟล์โมเดล** ให้แน่ใจว่าไม่เสียหาย  
                    3. **ใช้รูปภาพรูปแบบอื่น** (JPG, PNG)
                    4. **ลดขนาดรูปภาพ** หากมีขนาดใหญ่มาก
                    """)
    
    # ส่วนช่วยเหลือ
    render_help_section()
    
    # ส่วนท้าย
    render_footer()

if __name__ == "__main__":
    # ตรวจสอบ dependencies ก่อนเริ่มแอป
    try:
        main()
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเริ่มแอป: {e}")
        st.info("💡 ลอง refresh หน้าเว็บ หรือตรวจสอบว่าไฟล์ทั้งหมดอยู่ครบ")
        st.exception(e)