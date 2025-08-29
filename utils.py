"""
ฟังก์ชันช่วยเหลือสำหรับ Waste Detection App
"""

import glob
import os
import numpy as np
import streamlit as st
from config import MODEL_EXTENSIONS, WASTE_CATEGORIES, CLASSES

# Import dependencies อย่างปลอดภัย
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    st.error(f"⚠️ ไม่สามารถโหลด Ultralytics ได้: {e}")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.error("⚠️ PyTorch ไม่พร้อมใช้งาน")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    st.warning("⚠️ TIMM ไม่พร้อมใช้งาน")

def check_dependencies():
    """ตรวจสอบ dependencies ที่จำเป็น"""
    missing_deps = []
    
    if not ULTRALYTICS_AVAILABLE:
        missing_deps.append("ultralytics")
    if not TORCH_AVAILABLE:
        missing_deps.append("torch")
    if not TIMM_AVAILABLE:
        missing_deps.append("timm")
    
    if missing_deps:
        st.error(f"❌ Dependencies ที่ขาดหายไป: {', '.join(missing_deps)}")
        st.info("💡 กรุณารอให้ Streamlit Cloud ติดตั้ง libraries ทั้งหมด หรือลอง refresh หน้าเว็บ")
        return False
    
    return True

def find_model_files():
    """ค้นหาไฟล์โมเดล .pt ในโฟลเดอร์ต่างๆ"""
    model_files = []
    for pattern in MODEL_EXTENSIONS:
        try:
            model_files.extend(glob.glob(pattern))
        except Exception as e:
            st.warning(f"ไม่สามารถค้นหาโมเดลในรูปแบบ {pattern}: {e}")
    
    return sorted(model_files)

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """โหลดโมเดล YOLO"""
    if not check_dependencies():
        return None, "❌ Dependencies ไม่ครบถ้วน"
    
    try:
        # ตั้งค่า torch เพื่อประหยัด memory
        if TORCH_AVAILABLE:
            torch.set_num_threads(2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if os.path.exists(model_path):
            with st.spinner(f"กำลังโหลดโมเดล {os.path.basename(model_path)}..."):
                # โหลดโมเดลด้วย CPU
                model = YOLO(model_path)
                
                # บังคับใช้ CPU เพื่อประหยัด memory
                if hasattr(model.model, 'to'):
                    model.model.to('cpu')
                
                return model, f"✅ โหลดโมเดลสำเร็จ: {os.path.basename(model_path)}"
        else:
            return None, f"❌ ไม่พบไฟล์โมเดล: {model_path}"
            
    except Exception as e:
        error_msg = str(e)
        if "timm" in error_msg.lower():
            return None, "❌ ขาด timm library - กรุณารอให้ติดตั้งเสร็จ"
        elif "lightning" in error_msg.lower():
            return None, "❌ ขาด pytorch-lightning - กรุณารอให้ติดตั้งเสร็จ"
        else:
            return None, f"❌ ไม่สามารถโหลดโมเดลได้: {error_msg}"

def process_detections(results, confidence_threshold, iou_threshold):
    """ประมวลผลผลลัพธ์การตรวจจับ"""
    try:
        if not results or len(results) == 0:
            return None, None, None
            
        detections = results[0].boxes
        if detections is None or len(detections) == 0:
            return None, None, None
        
        class_counts = {}
        category_counts = {"รีไซเคิลได้": 0, "ย่อยสลายได้": 0, "อันตราย": 0, "ทั่วไป": 0}
        detailed_data = []
        
        for i, detection in enumerate(detections):
            try:
                class_id = int(detection.cls[0])
                if class_id < len(CLASSES):
                    class_name = CLASSES[class_id]
                    confidence = float(detection.conf[0])
                    
                    # ตรวจสอบ confidence threshold
                    if confidence < confidence_threshold:
                        continue
                    
                    # ใช้ CPU tensor
                    bbox = detection.xyxy[0].cpu().numpy() if hasattr(detection.xyxy[0], 'cpu') else detection.xyxy[0].numpy()
                    
                    # ข้อมูลประเภทขยะ
                    if class_name in WASTE_CATEGORIES:
                        info = WASTE_CATEGORIES[class_name]
                        category = info["category"]
                        
                        # นับจำนวนแต่ละประเภท
                        if class_name not in class_counts:
                            class_counts[class_name] = {"count": 0, "avg_confidence": 0, "category": category}
                        class_counts[class_name]["count"] += 1
                        class_counts[class_name]["avg_confidence"] += confidence
                        category_counts[category] += 1
                        
                        # ข้อมูลรายละเอียด
                        detailed_data.append({
                            "ID": i+1,
                            "ประเภท": f"{info['icon']} {class_name}",
                            "หมวดหมู่": category,
                            "วิธีจัดการ": info["recycle"],
                            "ถังขยะ": f"ถัง{info['bin_color']}",
                            "Confidence": f"{confidence:.3f}",
                            "X1": int(bbox[0]),
                            "Y1": int(bbox[1]), 
                            "X2": int(bbox[2]),
                            "Y2": int(bbox[3]),
                            "ขนาด": f"{int(bbox[2]-bbox[0])} x {int(bbox[3]-bbox[1])}"
                        })
                        
            except Exception as e:
                st.warning(f"ข้ามการประมวลผล detection {i}: {e}")
                continue
        
        # คำนวณค่าเฉลี่ย confidence
        for class_name in class_counts:
            if class_counts[class_name]["count"] > 0:
                class_counts[class_name]["avg_confidence"] /= class_counts[class_name]["count"]
        
        return class_counts, category_counts, detailed_data
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
        return None, None, None

def calculate_environmental_score(category_counts, total_detections):
    """คำนวณคะแนนสิ่งแวดล้อม"""
    if total_detections == 0:
        return 0, "ไม่พบขยะ"
    
    recyclable = category_counts.get("รีไซเคิลได้", 0)
    biodegradable = category_counts.get("ย่อยสลายได้", 0)
    
    eco_score = ((recyclable + biodegradable) / total_detections) * 100
    
    if eco_score >= 80:
        return eco_score, "🌟 ยอดเยี่ยม! คุณแยกขยะได้ดีมาก"
    elif eco_score >= 60:
        return eco_score, "👍 ดี! พยายามแยกขยะรีไซเคิลเพิ่มขึ้น"
    elif eco_score >= 40:
        return eco_score, "⚡ พอใช้ แยกขยะรีไซเคิลให้มากขึ้น"
    else:
        return eco_score, "🚨 ต้องปรับปรุง! แยกขยะเพื่อสิ่งแวดล้อม"

def generate_recommendations(category_counts):
    """สร้างคำแนะนำการจัดการขยะ"""
    recommendations = []
    
    if category_counts.get("รีไซเคิลได้", 0) > 0:
        recommendations.append("♻️ **รีไซเคิล**: แยกใส่ถังรีไซเคิลสีเหลือง")
    if category_counts.get("ย่อยสลายได้", 0) > 0:
        recommendations.append("🌱 **ย่อยสลายได้**: ใส่ถังขยะเปียกสีเขียวหรือทำปุ่ยหมัก")
    if category_counts.get("อันตราย", 0) > 0:
        recommendations.append("⚠️ **อันตราย**: นำไปทิ้งที่จุดรับขยะอันตรายพิเศษสีแดง")
    if category_counts.get("ทั่วไป", 0) > 0:
        recommendations.append("🗑️ **ทั่วไป**: ใส่ถังขยะแห้งสีน้ำเงิน")
    
    return recommendations

def create_summary_table(class_counts):
    """สร้างตารางสรุปผลการตรวจจับ"""
    results_data = []
    for class_name, data in class_counts.items():
        if class_name in WASTE_CATEGORIES:
            info = WASTE_CATEGORIES[class_name]
            results_data.append({
                "ประเภทขยะ": f"{info['icon']} {class_name}",
                "หมวดหมู่": data["category"],
                "จำนวน": data["count"],
                "ถังขยะ": f"ถัง{info['bin_color']}",
                "วิธีจัดการ": info["recycle"],
                "Confidence": f"{data['avg_confidence']:.3f}"
            })
    return results_data

def cleanup_memory():
    """เคลียร์ memory"""
    try:
        import gc
        gc.collect()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass