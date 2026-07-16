"""
AgroGrow PDF Report Generation Module.
Compiles analysis results, original image, segmentation overlays, features,
storage projections, and AI summaries into a premium PDF report using ReportLab.
"""

from pathlib import Path
import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class PDFReportGenerator:
    """
    Builds clean, professional PDF assessment reports.
    """
    @staticmethod
    def generate(
        image_path: Path,
        overlay_path: Path,
        features: dict,
        grading: dict,
        storage: dict,
        assistant_text: str,
        output_pdf_path: Path
    ) -> Path:
        """
        Generates and saves the PDF quality report.
        
        Args:
            image_path (Path): Path to original image.
            overlay_path (Path): Path to overlay visualization.
            features (dict): Dictionary of extracted features.
            grading (dict): Dictionary of grading classification results.
            storage (dict): Dictionary of storage shelf-life results.
            assistant_text (str): Explanatory summary text from AI Assistant.
            output_pdf_path (Path): Path where PDF should be saved.
            
        Returns:
            Path: Path to generated PDF file.
        """
        logger.info(f"Generating PDF report at: {output_pdf_path}")
        
        # 1. Page Template Setup
        doc = SimpleDocTemplate(
            str(output_pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Define clean, premium style variants
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1A365D"),  # Deep navy
            spaceAfter=15
        )
        
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2B6CB0"),  # Medium blue
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        )
        
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#2D3748")  # Dark grey
        )
        
        body_bold_style = ParagraphStyle(
            "ReportBodyBold",
            parent=body_style,
            fontName="Helvetica-Bold"
        )
        
        callout_style = ParagraphStyle(
            "CalloutText",
            parent=body_style,
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1A202C")
        )

        story = []
        
        # Header banner
        story.append(Paragraph("AgroGrow Post-Harvest Quality Assessment Report", title_style))
        story.append(Paragraph(f"<b>Assessment Timestamp:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Paragraph(f"<b>Analyzed Image:</b> {image_path.name}", body_style))
        story.append(Spacer(1, 10))
        
        # Section 1: Classification summary card
        grade_val = grading["grade"]
        conf_val = grading["confidence"]
        
        # Color match grade
        grade_color = "#38A169"  # Green
        if "Grade B" in grade_val:
            grade_color = "#3182CE"  # Blue
        elif "Grade C" in grade_val:
            grade_color = "#DD6B20"  # Orange
        elif "Grade D" in grade_val:
            grade_color = "#E53E3E"  # Red
            
        summary_table_data = [
            [
                Paragraph(f"<font color='{grade_color}' size=14><b>{grade_val}</b></font><br/>Confidence: {conf_val:.2%}", body_bold_style),
                Paragraph(f"<b>Quality Score:</b> {features['quality_score']:.1f}/100<br/>{grading['summary']}", body_style)
            ]
        ]
        summary_table = Table(summary_table_data, colWidths=[2.0*inch, 5.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # Section 2: Side-by-side Visualizations
        story.append(Paragraph("Visual Segmentation Mapping", section_style))
        
        # Resize images to fit side-by-side (3.6 inches width each)
        img_w = 3.5 * inch
        img_h = 3.5 * inch
        
        # Render ReportLab Image
        orig_img_flowable = Image(str(image_path), width=img_w, height=img_h)
        overlay_img_flowable = Image(str(overlay_path), width=img_w, height=img_h)
        
        viz_table_data = [
            [Paragraph("<b>Original Input Image</b>", body_bold_style), Paragraph("<b>Segmentation Classification Overlay</b>", body_bold_style)],
            [orig_img_flowable, overlay_img_flowable]
        ]
        viz_table = Table(viz_table_data, colWidths=[3.75*inch, 3.75*inch])
        viz_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(viz_table)
        story.append(Spacer(1, 15))
        
        # Section 3: Feature statistics & Storage predictions side-by-side
        stat_header = Paragraph("Agronomic Metrics", section_style)
        storage_header = Paragraph("Storage Stability Projection", section_style)
        
        # Build features table
        feat_data = [
            [Paragraph("<b>Metric Name</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style)],
            [Paragraph("Healthy Kernel Ratio", body_style), Paragraph(f"{features['healthy_percentage']:.2f}%", body_style)],
            [Paragraph("Diseased Kernel Ratio", body_style), Paragraph(f"{features['disease_percentage']:.2f}%", body_style)],
            [Paragraph("Missing Kernel Ratio", body_style), Paragraph(f"{features['missing_percentage']:.2f}%", body_style)],
            [Paragraph("Total Corn Cob Area", body_style), Paragraph(f"{features['total_corn_area_pixels']} px", body_style)],
            [Paragraph("Kernel Density Score", body_style), Paragraph(f"{features['kernel_density_score']:.3f}", body_style)],
        ]
        feat_table = Table(feat_data, colWidths=[2.2*inch, 1.2*inch])
        feat_table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        # Build storage table
        st_data = [
            [Paragraph("<b>Parameter</b>", body_bold_style), Paragraph("<b>Value</b>", body_bold_style)],
            [Paragraph("Predicted Shelf Life", body_bold_style), Paragraph(f"<b>{storage['shelf_life_days']:.1f} Days</b>", body_bold_style)],
            [Paragraph("Storage Risk Level", body_style), Paragraph(f"{storage['risk_level']}", body_style)],
            [Paragraph("Selected Storage Mode", body_style), Paragraph(f"{storage['storage_type']}", body_style)],
            [Paragraph("Simulated Temperature", body_style), Paragraph(f"{storage['temperature_c']} °C", body_style)],
            [Paragraph("Simulated Humidity", body_style), Paragraph(f"{storage['humidity_pct']} % RH", body_style)],
        ]
        st_table = Table(st_data, colWidths=[2.2*inch, 1.2*inch])
        st_table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#EDF2F7")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        bottom_table_data = [
            [stat_header, storage_header],
            [feat_table, st_table]
        ]
        bottom_table = Table(bottom_table_data, colWidths=[3.75*inch, 3.75*inch])
        bottom_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(bottom_table)
        story.append(Spacer(1, 10))
        
        # Section 4: Storage recommendation text box
        story.append(Paragraph("Storage Management Recommendations", section_style))
        rec_box_data = [[Paragraph(f"{storage['recommendation']}", body_style)]]
        rec_box = Table(rec_box_data, colWidths=[7.5*inch])
        rec_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFAF0")), # Warm yellow tint
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FEEBC8")),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(rec_box)
        story.append(Spacer(1, 10))
        
        # Section 5: AI assistant explanation
        story.append(Paragraph("Agronomist AI Assistant Summary", section_style))
        assistant_box_data = [[Paragraph(f"{assistant_text}", callout_style)]]
        assistant_box = Table(assistant_box_data, colWidths=[7.5*inch])
        assistant_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")), # Blue tint
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(KeepTogether([assistant_box]))
        
        # Build Document
        doc.build(story)
        logger.info("PDF report compiled successfully.")
        return output_pdf_path
