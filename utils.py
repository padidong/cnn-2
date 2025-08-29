"""
ฟังก์ชันช่วยเหลือสำหรับ Waste Detection App
"""

import glob
import os
import numpy as np
import streamlit as st
from config import MODEL_EXTENSIONS, WASTE_CATEGORIES, CLASSES

# Import YOLO อย่างปลอดภัย
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    st.error(f"⚠️ ไม่สามารถโหลด Ultralytics ได้: {e}")

def find_model_files():
    """ค้นหาไฟล์โมเดล .pt ในโฟลเดอร์ต่างๆ"""
    model_files = []
    for pattern in MODEL_EXTENSIONS:
        model_files.extend(glob.glob(pattern))
    return sorted(model_files)

@st.cache_resource
def load_model(model_path):
    """โหลดโมเดล YOLO"""
    if not ULTRALYTICS_AVAILABLE:
        return None, "❌ Ultralytics ไม่พร้อมใช้งาน"
    
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
            return model, f"✅ โหลดโมเดลสำเร็จ: {os.path.basename(model_path)}"
        else:
            return None, f"❌ ไม่พบไฟล์โมเดล: {model_path}"
    except Exception as e:
        return None, f"❌ ไม่สามารถโหลดโมเดลได้: {e}"

def process_detections(results, confidence_threshold, iou_threshold):
    """ประมวลผลผลลัพธ์การตรวจจับ"""
    try:
        detections = results[0].boxes
        if detections is None or len(detections) == 0:
            return None, None, None
        
        class_counts = {}
        category_counts = {"รีไซเคิลได้": 0, "ย่อยสลายได้": 0, "อันตราย": 0, "ทั่วไป": 0}
        detailed_data = []
        
        for i, detection in enumerate(detections):
            class_id = int(detection.cls[0])
            if class_id < len(CLASSES):
                class_name = CLASSES[class_id]
                confidence = float(detection.conf[0])
                bbox = detection.xyxy[0].cpu().numpy()
                
                # ข้อมูลประเภทขยะ
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
        
        # คำนวณค่าเฉลี่ย confidence
        for class_name in class_counts:
            class_counts[class_name]["avg_confidence"] /= class_counts[class_name]["count"]
        
        return class_counts, category_counts, detailed_data
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
        return None, None, None

def calculate_environmental_score(category_counts, total_detections):
    """คำนวณคะแนนสิ่งแวดล้อม"""
    recyclable = category_counts.get("รีไซเคิลได้", 0)
    biodegradable = category_counts.get("ย่อยสลายได้", 0)
    
    if total_detections == 0:
        return 0, "ไม่พบขยะ"
    
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
    
    if category_counts["รีไซเคิลได้"] > 0:
        recommendations.append("♻️ **รีไซเคิล**: แยกใส่ถังรีไซเคิลสีเหลือง")
    if category_counts["ย่อยสลายได้"] > 0:
        recommendations.append("🌱 **ย่อยสลายได้**: ใส่ถังขยะเปียกสีเขียวหรือทำปุ่ยหมัก")
    if category_counts["อันตราย"] > 0:
        recommendations.append("⚠️ **อันตราย**: นำไปทิ้งที่จุดรับขยะอันตรายพิเศษสีแดง")
    if category_counts["ทั่วไป"] > 0:
        recommendations.append("🗑️ **ทั่วไป**: ใส่ถังขยะแห้งสีน้ำเงิน")
    
    return recommendations

def create_summary_table(class_counts):
    """สร้างตารางสรุปผลการตรวจจับ"""
    results_data = []
    for class_name, data in class_counts.items():
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