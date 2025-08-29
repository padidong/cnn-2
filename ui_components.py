import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import WASTE_CATEGORIES

def render_header():
    """แสดงหัวเรื่องของแอป"""
    st.markdown('<h1 class="main-header">🗂️ ระบบตรวจจับและจำแนกขยะ</h1>', unsafe_allow_html=True)

def render_waste_info_section():
    """แสดงส่วนข้อมูลเกี่ยวกับการจำแนกขยะ"""
    with st.expander("📚 ข้อมูลเกี่ยวกับการจำแนกขยะ"):
        st.markdown("""
        <div class="info-box">
        <h4>ประเภทขยะที่ระบบสามารถตรวจจับได้:</h4>
        """, unsafe_allow_html=True)
        
        # สร้างตารางแสดงประเภทขยะ
        cols = st.columns(4)
        for i, (waste_type, info) in enumerate(WASTE_CATEGORIES.items()):
            col_idx = i % 4
            with cols[col_idx]:
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; margin: 5px; border: 2px solid {info['color']}; border-radius: 10px;">
                    <div style="font-size: 2em;">{info['icon']}</div>
                    <strong>{waste_type}</strong><br>
                    <span style="color: {info['color']};">{info['category']}</span><br>
                    <small>{info['recycle']}</small><br>
                    <small>ถัง{info['bin_color']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <h4>วิธีการทิ้งขยะที่ถูกต้อง:</h4>
        <ul>
        <li><span class="recyclable">♻️ รีไซเคิลได้:</span> แยกใส่ถังรีไซเคิลสีเหลือง</li>
        <li><span class="organic">🌱 ย่อยสลายได้:</span> ทำปุ่ยหมักหรือขยะเปียกสีเขียว</li>
        <li><span class="non-recyclable">⚠️ อันตราย:</span> ทิ้งที่จุดรับพิเศษสีแดง</li>
        <li><span class="general">🗑️ ทั่วไป:</span> ขยะแห้งทั่วไปสีน้ำเงิน</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

def render_sidebar_controls():
    """แสดงแถบควบคุมด้านข้าง"""
    with st.sidebar:
        st.markdown("## ⚙️ การตั้งค่า")
        return {
            "confidence": st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05),
            "iou": st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05),
            "filters": {
                "recyclable": st.checkbox("รีไซเคิลได้", value=True),
                "organic": st.checkbox("ย่อยสลายได้", value=True),
                "hazardous": st.checkbox("อันตราย", value=True),
                "general": st.checkbox("ทั่วไป", value=True)
            }
        }

def render_statistics_cards(category_counts, total_detections):
    """แสดงบัตรสถิติ"""
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("🗑️ ขยะทั้งหมด", total_detections)
    with col_stat2:
        st.metric("♻️ รีไซเคิลได้", category_counts.get("รีไซเคิลได้", 0))
    with col_stat3:
        st.metric("🌱 ย่อยสลายได้", category_counts.get("ย่อยสลายได้", 0))
    with col_stat4:
        st.metric("⚠️ อันตราย", category_counts.get("อันตราย", 0))

def render_results_table(results_data):
    """แสดงตารางผลลัพธ์"""
    st.markdown("#### 📋 รายละเอียดการตรวจจับ")
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

def render_charts(class_counts, category_counts):
    """แสดงกราฟ"""
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        if len(class_counts) > 1:
            fig1 = px.pie(
                values=[data["count"] for data in class_counts.values()],
                names=[f"{WASTE_CATEGORIES[name]['icon']} {name}" for name in class_counts.keys()],
                title="สัดส่วนประเภทขยะที่พบ",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
    
    with col_chart2:
        if sum(category_counts.values()) > 0:
            colors = {
                "รีไซเคิลได้": "#27AE60",
                "ย่อยสลายได้": "#8E44AD", 
                "อันตราย": "#E74C3C",
                "ทั่วไป": "#696969"
            }
            
            fig2 = px.bar(
                x=list(category_counts.keys()),
                y=list(category_counts.values()),
                title="จำนวนขยะตามหมวดหมู่",
                color=list(category_counts.keys()),
                color_discrete_map=colors
            )
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)

def render_recommendations_section(recommendations, eco_score, eco_message):
    """แสดงส่วนคำแนะนำ"""
    st.markdown("#### 💡 คำแนะนำการจัดการขยะ")
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # คะแนนสิ่งแวดล้อม
    st.markdown(f"""
    #### 🌍 คะแนนเพื่อสิ่งแวดล้อม
    **{eco_score:.1f}%** ของขยะสามารถจัดการอย่างเป็นมิตรกับสิ่งแวดล้อมได้
    """)
    
    if eco_score >= 80:
        st.success(eco_message)
    elif eco_score >= 60:
        st.info(eco_message)
    elif eco_score >= 40:
        st.warning(eco_message)
    else:
        st.error(eco_message)

def render_detailed_results(detailed_data):
    """แสดงผลลัพธ์รายละเอียด"""
    with st.expander("🔍 รายละเอียดการตรวจจับแต่ละชิ้น"):
        detailed_df = pd.DataFrame(detailed_data)
        st.dataframe(detailed_df, use_container_width=True)

def render_help_section():
    """แสดงส่วนช่วยเหลือ"""
    st.markdown("---")
    with st.expander("ℹ️ วิธีใช้งานและข้อมูลเพิ่มเติม"):
        st.markdown("""
        ### วิธีใช้งาน:
        1. เลือกโมเดล .pt ที่ต้องการใช้จากแถบข้าง
        2. อัปโหลดรูปภาพขยะในรูปแบบ PNG, JPG, JPEG, BMP หรือ TIFF
        3. ปรับค่า Confidence และ IoU Threshold (ถ้าต้องการ)
        4. คลิกปุ่ม "เริ่มตรวจจับและจำแนกขยะ"
        5. ดูผลการจำแนกและคำแนะนำการจัดการขยะ
        
        ### เกี่ยวกับโมเดล:
        - ระบบรองรับโมเดล YOLOv8 ที่ฝึกสำหรับการจำแนกขยะ 12 ประเภท
        - โมเดลควรมีการฝึกด้วยคลาส: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass
        
        ### คำแนะนำการใช้งาน:
        - **Confidence Threshold**: ค่าความมั่นใจขั้นต่ำในการตรวจจับ (แนะนำ 0.3-0.7)
        - **IoU Threshold**: ค่าสำหรับกรองการตรวจจับที่ซ้ำกัน (แนะนำ 0.4-0.5)
        - ใช้รูปภาพที่มีแสงเพียงพอและขยะชัดเจน
        
        ### การจัดการโฟลเดอร์โมเดล:
        - วางไฟล์ .pt ในโฟลเดอร์เดียวกับสคริปต์
        - หรือสร้างโฟลเดอร์ `models/` หรือ `weights/` แล้ววางไฟล์โมเดลไว้
        """)

def render_footer():
    """แสดงส่วนท้าย"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D;">
        <p>🗂️ ระบบตรวจจับและจำแนกขยะ | Powered by YOLO & Streamlit</p>
        <p>🌍 ช่วยกันดูแลสิ่งแวดล้อม แยกขยะให้ถูกวิธี</p>
    </div>
    """, unsafe_allow_html=True)