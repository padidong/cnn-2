import glob
import os
import numpy as np
import streamlit as st
from config import MODEL_EXTENSIONS, WASTE_CATEGORIES, CLASSES

# ============ เพิ่มส่วนนี้เพื่อปิด warnings ============
import warnings
# ปิด warning ที่เกี่ยวกับ weights_only
warnings.filterwarnings("ignore", message=".*weights_only.*")
# ปิด FutureWarning ทั้งหมด
warnings.filterwarnings("ignore", category=FutureWarning)
# ปิด UserWarning ที่เกี่ยวกับ torch.load
warnings.filterwarnings("ignore", message=".*torch.load.*")
# ========================================================

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
    
    # เพิ่ม safe globals สำหรับ model components ต่างๆ
    safe_globals_list = []
    
    # Lightning components
    try:
        import lightning.fabric.wrappers
        safe_globals_list.append(lightning.fabric.wrappers._FabricModule)
    except ImportError:
        pass
    
    try:
        import lightning.pytorch.core.module
        safe_globals_list.append(lightning.pytorch.core.module.LightningModule)
    except ImportError:
        pass
    
    # TIMM components
    try:
        import timm.models.mobilenetv3
        safe_globals_list.append(timm.models.mobilenetv3.MobileNetV3)
        import timm.models.efficientnet
        safe_globals_list.append(timm.models.efficientnet.EfficientNet)
        import timm.models.resnet
        safe_globals_list.append(timm.models.resnet.ResNet)
    except ImportError:
        pass
    
    # PyTorch components
    safe_globals_list.extend([
        torch.nn.Conv2d,
        torch.nn.BatchNorm2d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.Linear,
        torch.nn.Dropout,
        torch.nn.Sequential,
        torch.nn.ModuleList,
        torch.nn.ModuleDict,
        torch.nn.Identity,
        torch.nn.Hardswish,
        torch.nn.Hardsigmoid
    ])
    
    # เพิ่ม safe globals ถ้า PyTorch รองรับ
    if hasattr(torch.serialization, 'add_safe_globals'):
        try:
            torch.serialization.add_safe_globals(safe_globals_list)
            # ลบข้อความ success ออก เพื่อไม่ให้รก
            # st.success(f"✅ เพิ่ม {len(safe_globals_list)} safe globals สำเร็จ")
        except Exception as e:
            # ลบข้อความ warning ออก เพื่อไม่ให้รก
            # st.warning(f"⚠️ ไม่สามารถเพิ่ม safe globals: {e}")
            pass

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

def safe_torch_load(model_path):
    """โหลดโมเดลด้วย torch.load อย่างปลอดภัย"""
    if not TORCH_AVAILABLE:
        raise Exception("PyTorch ไม่พร้อมใช้งาน")
    
    try:
        # วิธีที่ 1: ลองโหลดแบบ weights_only=True ก่อน
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                return checkpoint, 'weights_only_true'
        except Exception as weights_only_error:
            # ไม่แสดง warning message
            pass
            
        # วิธีที่ 2: ใช้ safe_globals context
        try:
            # สร้างรายการ safe globals ที่ครบถ้วน
            additional_safe_globals = []
            
            # เพิ่ม TIMM models ที่อาจต้องใช้
            timm_models = [
                'timm.models.mobilenetv3.MobileNetV3',
                'timm.models.efficientnet.EfficientNet', 
                'timm.models.resnet.ResNet',
                'timm.models._registry.model_entrypoint'
            ]
            
            for model_name in timm_models:
                try:
                    parts = model_name.split('.')
                    module = __import__('.'.join(parts[:-1]), fromlist=[parts[-1]])
                    cls = getattr(module, parts[-1])
                    additional_safe_globals.append(cls)
                except (ImportError, AttributeError):
                    pass
            
            # รวม safe globals ทั้งหมด
            all_safe_globals = safe_globals_list + additional_safe_globals
            
            if all_safe_globals and hasattr(torch.serialization, 'safe_globals'):
                with torch.serialization.safe_globals(all_safe_globals):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                        return checkpoint, 'safe_globals'
            else:
                raise Exception("ไม่สามารถใช้ safe_globals ได้")
                
        except Exception as safe_globals_error:
            # ไม่แสดง warning message
            pass
            
        # วิธีที่ 3: โหลดแบบ weights_only=False (unsafe) - ปิด warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            return checkpoint, 'unsafe'
            
    except Exception as e:
        raise Exception(f"ไม่สามารถโหลดโมเดลได้ในทุกวิธี: {e}")

def detect_model_type(model_path):
    """ตรวจสอบประเภทของโมเดล - ปรับปรุงให้ปลอดภัยขึ้น"""
    try:
        # ตรวจสอบจากชื่อไฟล์ก่อน (เร็วและปลอดภัย)
        filename = os.path.basename(model_path).lower()
        
        # ตรวจสอบคำสำคัญในชื่อไฟล์
        if 'yolo' in filename:
            return 'yolo'
        elif 'efficientnet' in filename or 'efficient' in filename:
            return 'efficientnet'
        elif 'mobilenet' in filename or 'mobile' in filename:
            return 'mobilenet'
        elif 'resnet' in filename:
            return 'resnet'
        
        # ถ้าจากชื่อไฟล์ไม่ได้ ลองเปิดไฟล์ดู (ระมัดระวัง)
        try:
            checkpoint, load_method = safe_torch_load(model_path)
            # ลบข้อความ info ออก
            # st.info(f"🔍 ตรวจสอบโมเดลด้วยวิธี: {load_method}")
            
            # ตรวจสอบ structure ของโมเดล
            if isinstance(checkpoint, dict):
                # ตรวจสอบ keys
                keys = list(checkpoint.keys())
                keys_str = ' '.join(str(k).lower() for k in keys)
                
                # ตรวจสอบจาก model object
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    model_str = str(type(model)).lower()
                    
                    if 'efficientnet' in model_str:
                        return 'efficientnet'
                    elif 'mobilenet' in model_str:
                        return 'mobilenet'
                    elif 'yolo' in model_str:
                        return 'yolo'
                    elif hasattr(model, '__class__'):
                        class_name = model.__class__.__name__.lower()
                        if 'efficientnet' in class_name:
                            return 'efficientnet'
                        elif 'mobilenet' in class_name:
                            return 'mobilenet'
                
                # ตรวจสอบจาก metadata
                if 'yaml' in keys_str or 'anchors' in keys_str or 'stride' in keys_str:
                    return 'yolo'
                elif 'efficientnet' in keys_str:
                    return 'efficientnet'
                elif 'mobilenet' in keys_str or 'mobile' in keys_str:
                    return 'mobilenet'
                
                # ตรวจสอบจาก state_dict structure
                if 'state_dict' in checkpoint:
                    state_keys = list(checkpoint['state_dict'].keys())
                    state_str = ' '.join(state_keys).lower()
                    
                    if any(keyword in state_str for keyword in ['backbone', 'neck', 'head', 'detect']):
                        return 'yolo'
                    elif 'classifier' in state_str or 'features' in state_str:
                        if 'efficientnet' in state_str:
                            return 'efficientnet'
                        elif 'mobilenet' in state_str:
                            return 'mobilenet'
                        else:
                            return 'efficientnet'  # default classification
            
            # ถ้าเป็น model object โดยตรง
            elif hasattr(checkpoint, '__class__'):
                class_name = checkpoint.__class__.__name__.lower()
                if 'efficientnet' in class_name:
                    return 'efficientnet'
                elif 'mobilenet' in class_name:
                    return 'mobilenet'
                elif 'yolo' in class_name:
                    return 'yolo'
            
            # Default fallback
            # ลบข้อความ info ออก
            # st.info("ไม่สามารถระบุประเภทโมเดลจาก structure ได้ จะลองใช้เป็น EfficientNet")
            return 'efficientnet'
            
        except Exception as load_error:
            # ลบข้อความ warning ออก
            # st.warning(f"ไม่สามารถตรวจสอบโมเดลจากเนื้อหา: {load_error}")
            # ถ้าโหลดไม่ได้ ใช้ EfficientNet เป็น default (เพราะส่วนใหญ่เป็น classification)
            return 'efficientnet'
        
    except Exception as e:
        # ลบข้อความ warning ออก
        # st.warning(f"เกิดข้อผิดพลาดในการตรวจสอบประเภท: {e}")
        return 'efficientnet'  # fallback เป็น EfficientNet

# ============ ส่วนที่เหลือของโค้ดเหมือนเดิมทุกประการ ============
# คลาสและฟังก์ชันที่เหลือทั้งหมดยังคงเหมือนเดิม

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
            elif self.model_type in ['efficientnet', 'mobilenet', 'resnet']:
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
            
            # Transform สำหรับ classification models
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
                if confidence.item() >= conf:
                    mock_result = MockDetectionResult(
                        class_id=predicted.item(),
                        confidence=confidence.item(),
                        image_shape=image.size
                    )
                    return [mock_result]
                else:
                    return []
                
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

def extract_model_from_checkpoint(checkpoint):
    """แยกโมเดลจาก checkpoint ที่ซับซ้อน"""
    try:
        model = None
        
        # วิธีที่ 1: โมเดลอยู่ใน key 'model'
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            
            # ถ้าเป็น _FabricModule ให้แยก module ออกมา
            if hasattr(model, '_forward_module'):
                model = model._forward_module
            elif hasattr(model, 'module'):
                model = model.module
            elif hasattr(model, '_orig_mod'):
                model = model._orig_mod
                
        # วิธีที่ 2: มี state_dict ให้สร้างโมเดลใหม่
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # ลองสร้างโมเดลใหม่และโหลด state_dict
            model = create_model_from_state_dict(checkpoint['state_dict'])
            
        # วิธีที่ 3: checkpoint เป็นโมเดลโดยตรง
        elif hasattr(checkpoint, '__class__') and hasattr(checkpoint, 'forward'):
            model = checkpoint
            
            # ถ้าเป็น wrapped module
            if hasattr(model, '_forward_module'):
                model = model._forward_module
            elif hasattr(model, 'module'):
                model = model.module
        
        return model
        
    except Exception as e:
        # ลบข้อความ warning ออก
        # st.warning(f"ไม่สามารถแยกโมเดลจาก checkpoint: {e}")
        return None

def create_model_from_state_dict(state_dict):
    """สร้างโมเดลจาก state_dict"""
    try:
        if not TIMM_AVAILABLE:
            return None
        
        # วิเคราะห์ state_dict เพื่อหาประเภทโมเดล
        keys = list(state_dict.keys())
        keys_str = ' '.join(keys).lower()
        
        # ตรวจสอบจำนวนคลาส
        num_classes = len(CLASSES)
        for key in keys:
            if 'classifier' in key.lower() or 'head' in key.lower():
                shape = state_dict[key].shape
                if len(shape) >= 2:
                    num_classes = shape[0]
                    break
        
        # สร้างโมเดลตามประเภท
        if 'efficientnet' in keys_str:
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        elif 'mobilenet' in keys_str:
            model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=num_classes)
        else:
            # Default เป็น EfficientNet
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
        
        # โหลด state_dict
        model.load_state_dict(state_dict, strict=False)
        return model
        
    except Exception as e:
        # ลบข้อความ warning ออก
        # st.warning(f"ไม่สามารถสร้างโมเดลจาก state_dict: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """โหลดโมเดลประเภทต่างๆ - แก้ไขปัญหาทั้งหมด"""
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
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = YOLO(model_path)
                    
                    # บังคับใช้ CPU
                    if hasattr(model.model, 'to'):
                        model.model.to('cpu')
                    
                    return ModelWrapper(model, 'yolo'), f"✅ โหลดโมเดล YOLO สำเร็จ: {os.path.basename(model_path)}"
                
                except Exception as yolo_error:
                    # ลบข้อความ warning ออก
                    # st.warning(f"ไม่สามารถโหลดเป็น YOLO: {yolo_error}")
                    # ถ้า YOLO ไม่ได้ ลองเป็น classification
                    model_type = 'efficientnet'
            
            if model_type in ['efficientnet', 'mobilenet', 'resnet']:
                # โหลด Classification model
                try:
                    checkpoint, load_method = safe_torch_load(model_path)
                    st.info(f"📥 โหลดโมเดลด้วยวิธี: {load_method}")
                    
                    # แยกโมเดลจาก checkpoint
                    model = extract_model_from_checkpoint(checkpoint)
                    
                    if model is None:
                        raise Exception("ไม่สามารถแยกโมเดลจาก checkpoint ได้")
                    
                    # ตั้งค่าโมเดล
                    model.eval()
                    model.to('cpu')
                    
                    return ModelWrapper(model, model_type), f"✅ โหลดโมเดล {model_type} สำเร็จ (วิธี: {load_method}): {os.path.basename(model_path)}"
                
                except Exception as classification_error:
                    error_msg = str(classification_error)
                    st.error(f"ไม่สามารถโหลดเป็น classification model: {error_msg}")
                    
                    # Last resort: ลองเป็น YOLO อีกครั้ง
                    if "not iterable" in error_msg:
                        st.info("🔄 ลองโหลดเป็น YOLO แทน...")
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model = YOLO(model_path)
                            return ModelWrapper(model, 'yolo'), f"✅ โหลดโมเดลเป็น YOLO (fallback): {os.path.basename(model_path)}"
                        except Exception as final_error:
                            return None, f"❌ ไม่สามารถโหลดในทุกรูปแบบ: {final_error}"
                    
                    return None, f"❌ ไม่สามารถโหลด classification model: {error_msg}"
            
    except Exception as e:
        error_msg = str(e)
        if "weights_only" in error_msg.lower():
            return None, f"❌ ปัญหา PyTorch weights_only: ลองใช้โมเดล YOLO แทน"
        elif "not iterable" in error_msg:
            return None, f"❌ ปัญหา _FabricModule: โมเดลนี้ใช้ Lightning Fabric ที่ซับซ้อน ลองแปลงโมเดลก่อน"
        elif "lightning" in error_msg.lower():
            return None, f"❌ ปัญหา Lightning Framework: ลองใช้โมเดลที่ไม่ใช้ Lightning"
        elif "timm" in error_msg.lower():
            return None, "❌ ขาด timm library - กรุณารอให้ติดตั้งเสร็จ"
        else:
            return None, f"❌ ไม่สามารถโหลดโมเดลได้: {error_msg}"

# ฟังก์ชันที่เหลือเหมือนเดิม (process_detections, calculate_environmental_score, etc.)

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
                            "ขนาด": f"{int(bbox[2]-bbox[0])} x {int(bbox[3]-bbox[1])}" if len(bbox) >= 4 else "Full Image"
                        })
                        
            except Exception as e:
                # ลบข้อความ warning ออก (ปิด warning)
                # st.warning(f"ข้ามการประมวลผล detection {i}: {e}")
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