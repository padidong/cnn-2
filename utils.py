"""
ฟังก์ชันช่วยเหลือสำหรับ Waste Detection App
รองรับ YOLO, EfficientNet และโมเดลอื่นๆ
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
    import torch.nn as nn
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

def detect_model_type(model_path):
    """ตรวจสอบประเภทของโมเดล"""
    try:
        # โหลดเพื่อตรวจสอบประเภท
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # ตรวจสอบ metadata
        if 'model' in checkpoint:
            model_info = checkpoint.get('model', {})
            if hasattr(model_info, 'yaml') or 'yaml' in str(checkpoint):
                return 'yolo'
            elif 'efficientnet' in str(model_info).lower():
                return 'efficientnet'
            elif 'mobilenet' in str(model_info).lower():
                return 'mobilenet'
        
        # ตรวจสอบจากชื่อไฟล์
        filename = os.path.basename(model_path).lower()
        if 'yolo' in filename:
            return 'yolo'
        elif 'efficientnet' in filename or 'efficient' in filename:
            return 'efficientnet'
        elif 'mobilenet' in filename or 'mobile' in filename:
            return 'mobilenet'
        
        # Default เป็น yolo
        return 'yolo'
        
    except Exception as e:
        st.warning(f"ไม่สามารถตรวจสอบประเภทโมเดลได้: {e}")
        return 'unknown'

class ModelWrapper:
    """Wrapper class สำหรับโมเดลประเภทต่างๆ"""
    
    def __init__(self, model, model_type='yolo'):
        self.model = model
        self.model_type = model_type
        
    def predict(self, image, conf=0.5, iou=0.45):
        """ทำนายผลจากรูปภาพ"""
        try:
            if self.model_type == 'yolo':
                return self.model(image, conf=conf, iou=iou)
            elif self.model_type in ['efficientnet', 'mobilenet']:
                return self._predict_classification(image, conf)
            else:
                st.error(f"ไม่รองรับประเภทโมเดล: {self.model_type}")
                return None
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
            return None
    
    def _predict_classification(self, image, conf=0.5):
        """สำหรับโมเดล classification"""
        try:
            import torch.nn.functional as F
            from PIL import Image
            import torchvision.transforms as transforms
            
            # เตรียมรูปภาพ
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Transform สำหรับ EfficientNet
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            # Prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                # สร้าง mock result เพื่อให้เข้ากันได้กับ YOLO format
                mock_result = MockDetectionResult(
                    class_id=predicted.item(),
                    confidence=confidence.item(),
                    image_shape=image.size
                )
                
                return [mock_result] if confidence.item() >= conf else []
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการ classify: {e}")
            return None
    
    def __call__(self, *args, **kwargs):
        """เพื่อให้ใช้งานได้เหมือน YOLO model"""
        return self.predict(*args, **kwargs)

class MockDetectionResult:
    """Mock class สำหรับผลลัพธ์ที่ไม่ใช่ YOLO"""
    
    def __init__(self, class_id, confidence, image_shape):
        self.class_id = class_id
        self.confidence = confidence
        self.image_shape = image_shape
        self.boxes = MockBoxes(class_id, confidence, image_shape)
        self.orig_img = None
    
    def plot(self):
        """Mock plot function"""
        # ส่งคืน array เปล่าเนื่องจากไม่มี bounding box
        return np.zeros((self.image_shape[1], self.image_shape[0], 3), dtype=np.uint8)

class MockBoxes:
    """Mock boxes สำหรับ classification results"""
    
    def __init__(self, class_id, confidence, image_shape):
        self.data = [{
            'cls': torch.tensor([class_id]),
            'conf': torch.tensor([confidence]),
            'xyxy': torch.tensor([[0, 0, image_shape[0], image_shape[1]]])  # Full image
        }]
    
    def __len__(self):
        return 1 if self.data else 0
    
    def __getitem__(self, index):
        return MockBox(self.data[index])

class MockBox:
    """Mock individual box"""
    
    def __init__(self, data):
        self.cls = data['cls']
        self.conf = data['conf']
        self.xyxy = data['xyxy']

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """โหลดโมเดลประเภทต่างๆ"""
    if not check_dependencies():
        return None, "❌ Dependencies ไม่ครบถ้วน"
    
    try:
        # ตรวจสอบประเภทโมเดล
        model_type = detect_model_type(model_path)
        st.info(f"🔍 ตรวจพบโมเดลประเภท: {model_type}")
        
        # ตั้งค่า torch เพื่อประหยัด memory
        if TORCH_AVAILABLE:
            torch.set_num_threads(2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not os.path.exists(model_path):
            return None, f"❌ ไม่พบไฟล์โมเดล: {model_path}"
        
        with st.spinner(f"กำลังโหลดโมเดล {os.path.basename(model_path)}..."):
            
            if model_type == 'yolo':
                # โหลด YOLO model
                model = YOLO(model_path)
                
                # บังคับใช้ CPU
                if hasattr(model.model, 'to'):
                    model.model.to('cpu')
                
                return ModelWrapper(model, 'yolo'), f"✅ โหลดโมเดล YOLO สำเร็จ: {os.path.basename(model_path)}"
                
            elif model_type in ['efficientnet', 'mobilenet']:
                # โหลด EfficientNet/MobileNet model
                checkpoint = torch.load(model_path, map_location='cpu')
                
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # สร้างโมเดลใหม่และโหลด state_dict
                    model = create_efficientnet_model()
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model = checkpoint
                
                # ตั้งค่าโมเดล
                model.eval()
                model.to('cpu')
                
                return ModelWrapper(model, model_type), f"✅ โหลดโมเดล {model_type} สำเร็จ: {os.path.basename(model_path)}"
            
            else:
                # ลองโหลดเป็น YOLO ก่อน
                try:
                    model = YOLO(model_path)
                    if hasattr(model.model, 'to'):
                        model.model.to('cpu')
                    return ModelWrapper(model, 'yolo'), f"✅ โหลดโมเดลสำเร็จ (YOLO): {os.path.basename(model_path)}"
                except:
                    return None, f"❌ ไม่สามารถโหลดโมเดลได้: ไม่รู้จักประเภทโมเดล"
            
    except Exception as e:
        error_msg = str(e)
        if "timm" in error_msg.lower():
            return None, "❌ ขาด timm library - กรุณารอให้ติดตั้งเสร็จ"
        elif "lightning" in error_msg.lower():
            return None, "❌ ขาด pytorch-lightning - กรุณารอให้ติดตั้งเสร็จ"
        elif "efficientnet" in error_msg.lower():
            return None, f"❌ ปัญหากับ EfficientNet model: {error_msg}"
        else:
            return None, f"❌ ไม่สามารถโหลดโมเดลได้: {error_msg}"

def create_efficientnet_model():
    """สร้างโมเดล EfficientNet"""
    try:
        import timm
        # ใช้ EfficientNet จาก timm
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(CLASSES))
        return model
    except Exception as e:
        st.error(f"ไม่สามารถสร้าง EfficientNet model: {e}")
        return None

def process_detections(results, confidence_threshold, iou_threshold):
    """ประมวลผลผลลัพธ์การตรวจจับ - รองรับทั้ง YOLO และ Classification"""
    try:
        if not results or len(results) == 0:
            return None, None, None
            
        # ตรวจสอบประเภทผลลัพธ์
        result = results[0]
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            detections = result.boxes
        else:
            return None, None, None
        
        if len(detections) == 0:
            return None, None, None
        
        class_counts = {}
        category_counts = {"รีไซเคิลได้": 0, "ย่อยสลายได้": 0, "อันตราย": 0, "ทั่วไป": 0}
        detailed_data = []
        
        for i, detection in enumerate(detections):
            try:
                # รองรับทั้ง YOLO format และ Mock format
                if hasattr(detection, 'cls'):
                    class_id = int(detection.cls[0])
                    confidence = float(detection.conf[0])
                    
                    if hasattr(detection.xyxy[0], 'cpu'):
                        bbox = detection.xyxy[0].cpu().numpy()
                    else:
                        bbox = detection.xyxy[0].numpy()
                        
                elif isinstance(detection, MockBox):
                    class_id = int(detection.cls[0])
                    confidence = float(detection.conf[0])
                    bbox = detection.xyxy[0].numpy()
                else:
                    continue
                
                # ตรวจสอบ confidence threshold
                if confidence < confidence_threshold:
                    continue
                
                if class_id < len(CLASSES):
                    class_name = CLASSES[class_id]
                    
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
                            "X1": int(bbox[0]) if len(bbox) > 0 else 0,
                            "Y1": int(bbox[1]) if len(bbox) > 1 else 0,
                            "X2": int(bbox[2]) if len(bbox) > 2 else 100,
                            "Y2": int(bbox[3]) if len(bbox) > 3 else 100,
                            "ขนาด": f"{int(bbox[2]-bbox[0])} x {int(bbox[3]-bbox[1])}" if len(bbox) >= 4 else "N/A"
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