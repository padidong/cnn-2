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

# CSS สำหรับปรับแต่งหน้าเว็บ (ปรับปรุงใหม่ให้สวยงามขึ้น)
CSS_STYLES = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&family=Sarabun:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Kanit', 'Sarabun', sans-serif !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Main Header with gradient animation */
.main-header {
    font-size: 3.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #27AE60 50%, #667eea 75%, #764ba2 100%);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-weight: 700;
    margin-bottom: 2.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    animation: fadeIn 0.8s ease;
    letter-spacing: 0.5px;
}

/* Sub Headers */
.sub-header {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #FF6B6B, #D35400);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    margin: 1.5rem 0;
    display: inline-block;
    position: relative;
    padding-bottom: 10px;
}

.sub-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #FF6B6B, #D35400);
    border-radius: 2px;
    animation: fadeIn 1s ease;
}

/* Info Box with glassmorphism effect */
.info-box {
    background: linear-gradient(135deg, rgba(232, 248, 245, 0.95), rgba(200, 230, 220, 0.95));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 20px;
    border: 1px solid rgba(39, 174, 96, 0.2);
    box-shadow: 
        0 8px 32px rgba(39, 174, 96, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.6);
    margin: 1.5rem 0;
    animation: fadeIn 0.5s ease;
    position: relative;
    overflow: hidden;
}

.info-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Result Box with modern card design */
.result-box {
    background: linear-gradient(135deg, #FFF9E6 0%, #FFE4B5 100%);
    padding: 2rem;
    border-radius: 25px;
    box-shadow: 
        0 20px 40px rgba(243, 156, 18, 0.15),
        0 10px 20px rgba(243, 156, 18, 0.1),
        inset 0 2px 5px rgba(255, 255, 255, 0.9);
    margin: 2rem 0;
    border: 1px solid rgba(243, 156, 18, 0.2);
    position: relative;
    animation: fadeIn 0.6s ease;
    transition: all 0.3s ease;
}

.result-box:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 25px 50px rgba(243, 156, 18, 0.2),
        0 15px 30px rgba(243, 156, 18, 0.15);
}

/* Waste Category Cards */
.waste-card {
    background: linear-gradient(145deg, #ffffff, #f0f0f0);
    border-radius: 15px;
    padding: 15px;
    margin: 10px;
    box-shadow: 
        5px 5px 15px rgba(0, 0, 0, 0.1),
        -5px -5px 15px rgba(255, 255, 255, 0.7);
    transition: all 0.3s ease;
    border: 2px solid transparent;
    position: relative;
    overflow: hidden;
}

.waste-card:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 
        8px 8px 20px rgba(0, 0, 0, 0.15),
        -8px -8px 20px rgba(255, 255, 255, 0.8);
}

.waste-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--color, #27AE60), var(--color-light, #52E888));
}

/* Emoji Containers with animation */
.emoji-container {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 50px;
    height: 50px;
    font-size: 2em;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 50%;
    box-shadow: 
        0 4px 15px rgba(0, 0, 0, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.8);
    margin: 0 10px;
    transition: all 0.3s ease;
}

.emoji-container:hover {
    transform: rotate(10deg) scale(1.1);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
}

/* Category Colors with modern gradients */
.recyclable { 
    background: linear-gradient(135deg, #27AE60, #52E888);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    font-size: 1.1em;
}

.non-recyclable { 
    background: linear-gradient(135deg, #E74C3C, #FF6B6B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    font-size: 1.1em;
}

.organic { 
    background: linear-gradient(135deg, #8E44AD, #B86FD4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    font-size: 1.1em;
}

.general { 
    background: linear-gradient(135deg, #696969, #909090);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    font-size: 1.1em;
}

/* Metric Cards with neumorphism */
div[data-testid="metric-container"] {
    background: linear-gradient(145deg, #f5f5f5, #e0e0e0);
    box-shadow: 
        8px 8px 20px rgba(0, 0, 0, 0.1),
        -8px -8px 20px rgba(255, 255, 255, 0.7),
        inset 1px 1px 2px rgba(255, 255, 255, 0.3);
    border-radius: 15px;
    padding: 1rem;
    transition: all 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 
        10px 10px 25px rgba(0, 0, 0, 0.12),
        -10px -10px 25px rgba(255, 255, 255, 0.8);
}

/* Buttons with modern design */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 30px;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
}

/* Tables with modern styling */
.dataframe {
    border-radius: 15px !important;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
}

.dataframe th {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px !important;
    text-align: center !important;
    font-size: 0.95rem !important;
}

.dataframe td {
    padding: 10px !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
    transition: background-color 0.3s ease;
}

.dataframe tr:hover td {
    background-color: rgba(102, 126, 234, 0.05) !important;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    background-size: 200% 100%;
    animation: gradient 2s ease infinite;
    border-radius: 10px;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Expander with modern look */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(240, 240, 240, 0.9));
    border-radius: 15px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    font-weight: 500;
    transition: all 0.3s ease;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    transform: translateX(5px);
}

/* Success/Error/Warning/Info boxes */
.stAlert {
    border-radius: 15px;
    border-left-width: 5px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    animation: fadeIn 0.5s ease;
}

/* File uploader with modern design */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
    border: 2px dashed rgba(102, 126, 234, 0.5);
    border-radius: 20px;
    transition: all 0.3s ease;
}

[data-testid="stFileUploadDropzone"]:hover {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
    border-color: rgba(102, 126, 234, 0.8);
    transform: scale(1.02);
}

/* Tooltips */
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    background: linear-gradient(135deg, #333, #555);
    color: white;
    text-align: center;
    border-radius: 10px;
    padding: 8px 12px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -60px;
    opacity: 0;
    transition: opacity 0.3s;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main-header {
        font-size: 2.5rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
    }
    
    .info-box, .result-box {
        padding: 1rem;
        margin: 1rem 0;
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