# คลาสขยะที่ระบบสามารถตรวจจับได้
CLASSES = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

# การจำแนกประเภทขยะ
WASTE_CATEGORIES = {
    "battery": {
        "category": "อันตราย", 
        "color": "#E74C3C", 
        "recycle": "ต้องทิ้งที่จุดรับพิเศษ", 
        "icon": "🔋",
        "bin_color": "แดง",
        "description": "ขยะอันตรายต่อสิ่งแวดล้อม"
    },
    "biological": {
        "category": "ย่อยสลายได้", 
        "color": "#8E44AD", 
        "recycle": "ทำปุ่ยหมัก", 
        "icon": "🧬",
        "bin_color": "เขียว",
        "description": "ขยะเปียกที่สามารถย่อยสลายได้"
    },
    "brown-glass": {
        "category": "รีไซเคิลได้", 
        "color": "#A0522D", 
        "recycle": "ขวดแก้ว", 
        "icon": "🟤",
        "bin_color": "เหลือง",
        "description": "แก้วสีน้ำตาลสำหรับรีไซเคิล"
    },
    "cardboard": {
        "category": "รีไซเคิลได้", 
        "color": "#D2691E", 
        "recycle": "กระดาษรีไซเคิล", 
        "icon": "📦",
        "bin_color": "เหลือง",
        "description": "กระดาษแข็งและกล่องลูกฟูก"
    },
    "clothes": {
        "category": "รีไซเคิลได้", 
        "color": "#FF69B4", 
        "recycle": "บริจาคหรือรีไซเคิล", 
        "icon": "👕",
        "bin_color": "เหลือง",
        "description": "เสื้อผ้าที่ยังใช้งานได้"
    },
    "green-glass": {
        "category": "รีไซเคิลได้", 
        "color": "#228B22", 
        "recycle": "ขวดแก้ว", 
        "icon": "🟢",
        "bin_color": "เหลือง",
        "description": "แก้วสีเขียวสำหรับรีไซเคิล"
    },
    "metal": {
        "category": "รีไซเคิลได้", 
        "color": "#708090", 
        "recycle": "โลหะรีไซเคิล", 
        "icon": "🔩",
        "bin_color": "เหลือง",
        "description": "โลหะต่างๆ เช่น อลูมิเนียม เหล็ก"
    },
    "paper": {
        "category": "รีไซเคิลได้", 
        "color": "#F5F5DC", 
        "recycle": "กระดาษรีไซเคิล", 
        "icon": "📄",
        "bin_color": "เหลือง",
        "description": "กระดาษธรรมดาและหนังสือพิมพ์"
    },
    "plastic": {
        "category": "รีไซเคิลได้", 
        "color": "#4169E1", 
        "recycle": "พลาสติกรีไซเคิล", 
        "icon": "♻️",
        "bin_color": "เหลือง",
        "description": "ขวดพลาสติกและบรรจุภัณฑ์"
    },
    "shoes": {
        "category": "รีไซเคิลได้", 
        "color": "#8B4513", 
        "recycle": "บริจาคหรือรีไซเคิล", 
        "icon": "👟",
        "bin_color": "เหลือง",
        "description": "รองเท้าที่ยังใช้งานได้"
    },
    "trash": {
        "category": "ทั่วไป", 
        "color": "#696969", 
        "recycle": "ขยะทั่วไป", 
        "icon": "🗑️",
        "bin_color": "น้ำเงิน",
        "description": "ขยะแห้งทั่วไป"
    },
    "white-glass": {
        "category": "รีไซเคิลได้", 
        "color": "#F5F5F5", 
        "recycle": "ขวดแก้ว", 
        "icon": "⚪",
        "bin_color": "เหลือง",
        "description": "แก้วใสสำหรับรีไซเคิล"
    }
}

# CSS สำหรับปรับแต่งหน้าเว็บ
CSS_STYLES = """
<style>
.main-header {
    font-size: 3rem;
    color: #27AE60;
    text-align: center;
    font-weight: bold;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #D35400;
    font-weight: bold;
}
.info-box {
    background-color: #E8F8F5;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #27AE60;
    margin: 1rem 0;
}
.result-box {
    background-color: #FEF9E7;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #F39C12;
    margin: 1rem 0;
}
.recyclable { color: #27AE60; font-weight: bold; }
.non-recyclable { color: #E74C3C; font-weight: bold; }
.organic { color: #8E44AD; font-weight: bold; }
.general { color: #696969; font-weight: bold; }
</style>
"""

# การซ่อน Streamlit style
HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""

# การตั้งค่าเริ่มต้น
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.45
MODEL_EXTENSIONS = ["*.pt", "models/*.pt", "weights/*.pt"]
SUPPORTED_IMAGE_TYPES = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']