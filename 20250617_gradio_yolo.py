#!pip install gradio torch torchvision ultralytics opencv-python pillow matplotlib seaborn pandas

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# YOLOv5 모델 로드
def load_yolo_model(model_path=None):
    """YOLOv5 모델을 로드합니다."""
    if model_path:
        # 커스텀 모델 로드
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    else:
        # 기본 YOLOv5s 모델 로드 (체스 기물용 커스텀 모델이 없는 경우)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    return model

# 체스 기물 클래스 정의
CHESS_PIECES = {
    'white_king': '♔', 'white_queen': '♕', 'white_rook': '♖', 
    'white_bishop': '♗', 'white_knight': '♘', 'white_pawn': '♙',
    'black_king': '♚', 'black_queen': '♛', 'black_rook': '♜', 
    'black_bishop': '♝', 'black_knight': '♞', 'black_pawn': '♟'
}

PIECE_COLORS = {
    'white': '#F8F8FF', 'black': '#2F2F2F'
}

def detect_chess_pieces(image, model, confidence_threshold=0.5):
    """이미지에서 체스 기물을 탐지합니다."""
    results = model(image)
    detections = results.pandas().xyxy[0]
    
    # 신뢰도 필터링
    detections = detections[detections['confidence'] >= confidence_threshold]
    
    return detections, results

def draw_detections(image, detections):
    """탐지된 기물에 바운딩 박스를 그립니다."""
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # 폰트 설정 (기본 폰트 사용)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for idx, (_, detection) in enumerate(detections.iterrows()):
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        label = detection['name']
        
        # 바운딩 박스 색상
        color = colors[idx % len(colors)]
        
        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 라벨 배경
        text = f"{label}: {confidence:.2f}"
        bbox = draw.textbbox((x1, y1-25), text, font=font)
        draw.rectangle(bbox, fill=color, outline=color)
        draw.text((x1, y1-25), text, fill='white', font=font)
    
    return img_with_boxes

def create_piece_count_visualization(detections):
    """기물 개수를 시각화합니다."""
    if detections.empty:
        return None, None
    
    # 기물 개수 계산
    piece_counts = Counter(detections['name'])
    
    # 데이터프레임 생성
    df = pd.DataFrame(list(piece_counts.items()), columns=['Piece', 'Count'])
    
    # matplotlib 스타일 설정
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 막대 차트
    colors = sns.color_palette("husl", len(df))
    bars = ax1.bar(df['Piece'], df['Count'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Chess Pieces Count', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Piece Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 막대 위에 개수 표시
    for bar, count in zip(bars, df['Count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 파이 차트
    wedges, texts, autotexts = ax2.pie(df['Count'], labels=df['Piece'], autopct='%1.1f%%', 
                                      colors=colors, startangle=90, wedgeprops=dict(edgecolor='black', linewidth=1.5))
    ax2.set_title('Piece Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # 파이 차트 텍스트 스타일링
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    plt.tight_layout()
    
    # 통계 테이블 생성
    total_pieces = df['Count'].sum()
    stats_data = {
        'Total Pieces': [total_pieces],
        'Unique Types': [len(df)],
        'Most Common': [df.loc[df['Count'].idxmax(), 'Piece']],
        'Max Count': [df['Count'].max()]
    }
    stats_df = pd.DataFrame(stats_data)
    
    return fig, stats_df

def create_detailed_analysis(detections):
    """상세 분석 결과를 생성합니다."""
    if detections.empty:
        return "No pieces detected in the image."
    
    analysis = []
    analysis.append("🏁 **Chess Board Analysis Results**\n")
    analysis.append("="*50 + "\n")
    
    # 기물별 개수
    piece_counts = Counter(detections['name'])
    analysis.append("📊 **Piece Count Summary:**")
    for piece, count in sorted(piece_counts.items()):
        piece_symbol = CHESS_PIECES.get(piece, '●')
        analysis.append(f"   {piece_symbol} {piece.replace('_', ' ').title()}: **{count}**")
    
    analysis.append(f"\n🎯 **Detection Statistics:**")
    analysis.append(f"   • Total pieces detected: **{len(detections)}**")
    analysis.append(f"   • Unique piece types: **{len(piece_counts)}**")
    analysis.append(f"   • Average confidence: **{detections['confidence'].mean():.3f}**")
    analysis.append(f"   • Highest confidence: **{detections['confidence'].max():.3f}**")
    analysis.append(f"   • Lowest confidence: **{detections['confidence'].min():.3f}**")
    
    # 신뢰도별 분석
    high_conf = len(detections[detections['confidence'] >= 0.8])
    medium_conf = len(detections[(detections['confidence'] >= 0.5) & (detections['confidence'] < 0.8)])
    low_conf = len(detections[detections['confidence'] < 0.5])
    
    analysis.append(f"\n🎖️ **Confidence Distribution:**")
    analysis.append(f"   • High confidence (≥0.8): **{high_conf}** pieces")
    analysis.append(f"   • Medium confidence (0.5-0.8): **{medium_conf}** pieces")
    analysis.append(f"   • Low confidence (<0.5): **{low_conf}** pieces")
    
    return "\n".join(analysis)

def process_chess_image(image, model_path, confidence_threshold):
    """체스보드 이미지를 처리하고 결과를 반환합니다."""
    try:
        # 모델 로드
        model = load_yolo_model(model_path if model_path else None)
        
        # 이미지가 numpy array인 경우 PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 객체 탐지 수행
        detections, results = detect_chess_pieces(image, model, confidence_threshold)
        
        # 탐지 결과가 있는 경우에만 처리
        if not detections.empty:
            # 바운딩 박스가 있는 이미지 생성
            annotated_image = draw_detections(image, detections)
            
            # 통계 차트 생성
            chart_fig, stats_df = create_piece_count_visualization(detections)
            
            # 상세 분석 생성
            analysis_text = create_detailed_analysis(detections)
            
            return annotated_image, chart_fig, stats_df, analysis_text
        else:
            return image, None, None, "No chess pieces detected in the image. Try adjusting the confidence threshold."
            
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return image, None, None, error_msg

# Gradio 인터페이스 생성
def create_gradio_interface():
    """Gradio 인터페이스를 생성합니다."""
    
    with gr.Blocks(title="♛ Chess Piece Detection & Analysis", 
                   theme=gr.themes.Soft(),
                   css="""
                   .gradio-container {background: linear-gradient(45deg, #f0f2f6, #e8eaf6);}
                   .chess-header {text-align: center; color: #2c3e50; margin: 20px 0;}
                   """) as demo:
        
        gr.HTML("""
        <div class="chess-header">
            <h1>♛ Chess Piece Detection & Analysis ♛</h1>
            <p>Upload a chess board image to detect and count pieces using YOLOv5</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📸 Input Settings</h3>")
                
                # 이미지 업로드
                input_image = gr.Image(
                    label="Upload Chess Board Image",
                    type="pil",
                    height=300
                )
                
                # 모델 가중치 파일 업로드
                model_file = gr.File(
                    label="Upload YOLOv5 Model Weights (Optional)",
                    file_types=[".pt"],
                    type="filepath"
                )
                
                # 신뢰도 임계값 설정
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Confidence Threshold",
                    info="Minimum confidence score for detection"
                )
                
                # 분석 버튼
                analyze_btn = gr.Button(
                    "🔍 Analyze Chess Board",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.HTML("<h3>🎯 Detection Results</h3>")
                
                # 결과 이미지
                output_image = gr.Image(
                    label="Detected Pieces",
                    height=400
                )
                
                # 통계 차트
                stats_plot = gr.Plot(
                    label="Piece Statistics",
                    show_label=True
                )
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>📊 Statistics Table</h3>")
                stats_table = gr.Dataframe(
                    label="Detection Summary",
                    interactive=False
                )
            
            with gr.Column():
                gr.HTML("<h3>📝 Detailed Analysis</h3>")
                analysis_output = gr.Markdown(
                    label="Analysis Report",
                    height=300
                )
        
        # 이벤트 핸들링
        analyze_btn.click(
            fn=process_chess_image,
            inputs=[input_image, model_file, confidence_slider],
            outputs=[output_image, stats_plot, stats_table, analysis_output]
        )
        
        # 예제 추가
        gr.HTML("<h3>💡 Tips</h3>")
        gr.HTML("""
        <ul>
            <li>📷 Use clear, well-lit images of chess boards for best results</li>
            <li>🎯 Adjust confidence threshold if detection is too sensitive or missing pieces</li>
            <li>⚡ Upload custom YOLOv5 weights trained on chess pieces for better accuracy</li>
            <li>🔄 Try different angles and lighting conditions</li>
        </ul>
        """)
    
    return demo

# 앱 실행
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        #server_port=7860,
        show_error=True
    )