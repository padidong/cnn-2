def process_yolo_results(results):
    """ประมวลผลผลลัพธ์ YOLO/EfficientNet โดยไม่ใช้ cv2"""
    try:
        if not results or len(results) == 0:
            return None
            
        result = results[0]
        
        # ลองใช้ฟังก์ชัน plot ถ้ามี
        if hasattr(result, 'plot') and CV2_AVAILABLE:
            try:
                annotated_image = result.plot()
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                return annotated_image
            except:
                pass
        
        # ถ้าไม่มี cv2 หรือ plot ไม่ทำงาน ให้ใช้รูปต้นฉบับ
        if hasattr(result, 'orig_img') and result.orig_img is not None:
            annotated_image = np.array(result.orig_img)
            if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                # แปลง BGR เป็น RGB ถ้าจำเป็น
                annotated_image = annotated_image[:, :, ::-1]
            return annotated_image
        
        # หากไม่มีรูปต้นฉบับ ส่งคืน None
        return None
        
    except Exception as e:
        st.warning(f"ไม่สามารถแสดงรูปที่มี annotation ได้: {e}")
        return None