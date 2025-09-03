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
        "icon": "🍃",
        "bin_color": "เขียว",
        "description": "ขยะเปียกที่สามารถย่อยสลายได้"
    },
    "brown-glass": {
        "category": "รีไซเคิลได้", 
        "color": "#8B4513", 
        "recycle": "ขวดแก้ว", 
        "icon": "🍾",
        "bin_color": "เหลือง",
        "description": "แก้วสีน้ำตาลสำหรับรีไซเคิล"
    },
    "cardboard": {
        "category": "รีไซเคิลได้", 
        "color": "#D4A574", 
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
        "icon": "🍸",
        "bin_color": "เหลือง",
        "description": "แก้วสีเขียวสำหรับรีไซเคิล"
    },
    "metal": {
        "category": "รีไซเคิลได้", 
        "color": "#708090", 
        "recycle": "โลหะรีไซเคิล", 
        "icon": "🔧",
        "bin_color": "เหลือง",
        "description": "โลหะต่างๆ เช่น อลูมิเนียม เหล็ก"
    },
    "paper": {
        "category": "รีไซเคิลได้", 
        "color": "#87CEEB", 
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
        "color": "#E8E8E8", 
        "recycle": "ขวดแก้ว", 
        "icon": "🥛",
        "bin_color": "เหลือง",
        "description": "แก้วใสสำหรับรีไซเคิล"
    }
}

# ส่วน CSS ที่แก้ไขแล้ว - ใส่แทนที่ CSS_STYLES เดิมในไฟล์ config.py

CSS_STYLES = """
<style>
/* Global Styles - ลบ font override ที่ทำให้เกิดปัญหา */
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* Main Header */
.main-header {
    font-size: 3rem;
    color: #27AE60;
    text-align: center;
    font-weight: bold;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

/* Sub Headers */
.sub-header {
    font-size: 1.5rem;
    color: #D35400;
    font-weight: bold;
    margin: 1rem 0;
}

/* Info Box with better styling */
.info-box {
    background: linear-gradient(135deg, #E8F8F5, #D4EFDF);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #27AE60;
    margin: 1.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Result Box */
.result-box {
    background: linear-gradient(135deg, #FEF9E7, #FDEBD0);
    padding: 1.5rem;
    border-radius: 15px;
    border-left: 5px solid #F39C12;
    margin: 1.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Category Colors */
.recyclable { 
    color: #27AE60;
    font-weight: bold;
}

.non-recyclable { 
    color: #E74C3C;
    font-weight: bold;
}

.organic { 
    color: #8E44AD;
    font-weight: bold;
}

.general { 
    color: #696969;
    font-weight: bold;
}

/* Fix for Streamlit specific elements */
.stTextInput > div > div > input {
    font-family: inherit !important;
}

.stSelectbox > div > div > select {
    font-family: inherit !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #27AE60, #229954);
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(39, 174, 96, 0.05);
    border: 2px dashed #27AE60;
    border-radius: 10px;
    padding: 1rem;
}

/* Metric containers */
div[data-testid="metric-container"] {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Tables */
.dataframe {
    border-radius: 10px !important;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.dataframe th {
    background: #27AE60 !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px !important;
}

.dataframe td {
    padding: 8px !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(39, 174, 96, 0.1);
    border-radius: 10px;
    font-weight: 500;
}

/* Alert boxes */
.stAlert {
    border-radius: 10px;
    border-left-width: 5px;
}

/* Fix text overlap in input fields */
.stTextInput label {
    display: block !important;
    margin-bottom: 5px !important;
    font-weight: 500 !important;
}

.stSelectbox label {
    display: block !important;
    margin-bottom: 5px !important;
    font-weight: 500 !important;
}

/* Ensure proper spacing */
.element-container {
    margin-bottom: 1rem;
}

/* Fix for Streamlit's default overlapping */
.row-widget.stTextInput {
    margin-bottom: 1rem !important;
}

.row-widget.stSelectbox {
    margin-bottom: 1rem !important;
}

/* Simple animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.main-header, .info-box, .result-box {
    animation: fadeIn 0.5s ease;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    
    .sub-header {
        font-size: 1.25rem;
    }
}
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