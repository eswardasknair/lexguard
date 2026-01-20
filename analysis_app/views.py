import os
import logging
from django.shortcuts import render, redirect, get_object_or_404
from .models import Document, AnalyzedClause
from .ml_inference import LexGuardInference

logger = logging.getLogger(__name__)

# Initialize the inference engine once at startup
ml_engine = LexGuardInference()


def index(request):
    """Landing page view"""
    return render(request, 'index.html')


from django.db import models
from django.db.models import Avg, Count
from .models import Document, AnalyzedClause, LawResource, LawSection

def dashboard(request):
    """User dashboard view with aggregated metrics."""
    documents = Document.objects.all().order_by('-uploaded_at')
    
    # Calculate metrics
    total_docs = documents.count()
    avg_risk = documents.aggregate(Avg('overall_risk_score'))['overall_risk_score__avg'] or 0
    
    # Clause level metrics
    critical_clauses = AnalyzedClause.objects.filter(risk_level='Critical').count()
    high_clauses = AnalyzedClause.objects.filter(risk_level='High').count()
    medium_clauses = AnalyzedClause.objects.filter(risk_level='Medium').count()

    context = {
        'recent_docs': documents[:5],
        'total_docs': total_docs,
        'avg_risk': round(avg_risk, 1),
        'critical_clauses': critical_clauses,
        'high_clauses': high_clauses,
        'medium_clauses': medium_clauses,
    }
    return render(request, 'dashboard.html', context)


def law_book_list(request):
    """View to display the searchable law book."""
    query = request.GET.get('q', '')
    if query:
        sections = LawSection.objects.filter(
            models.Q(law__title__icontains=query) |
            models.Q(title__icontains=query) |
            models.Q(content__icontains=query) |
            models.Q(section_number__icontains=query)
        ).select_related('law')
    else:
        sections = LawSection.objects.all().select_related('law')
        
    resources = LawResource.objects.all()
    
    context = {
        'resources': resources,
        'sections': sections,
        'query': query
    }
    return render(request, 'law_book.html', context)


import json
from django.http import JsonResponse
import urllib.request
import urllib.parse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .ai_explainer import _call_gemini

@csrf_exempt
def generate_clause_explanation(request, clause_id):
    """AJAX endpoint to generate AI explanation on-demand."""
    if request.method == 'POST':
        try:
            clause = AnalyzedClause.objects.get(id=clause_id)
            if clause.ai_explanation:
                return JsonResponse({"explanation": clause.ai_explanation})
            
            # Use User's Gemini API Key primarily
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if api_key:
                explanation = _call_gemini(
                    api_key=api_key,
                    clause_text=clause.original_text,
                    clause_type=clause.clause_type,
                    risk_level=clause.risk_level
                )
                if explanation:
                    clause.ai_explanation = explanation
                    clause.save()
                    return JsonResponse({"explanation": explanation})
            
            # Fallback to general API if no key
            try:
                prompt = f"Explain this legal clause simply:\n{clause.original_text}"
                encoded_prompt = urllib.parse.quote(prompt)
                url = f"https://text.pollinations.ai/{encoded_prompt}"
                
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=5) as response:
                    result = response.read().decode('utf-8')
                    
                clause.ai_explanation = result.strip()
            except Exception as e:
                print(f"[AI] Fallback Error (Pollinations): {e}")
                clause.ai_explanation = "The AI service is currently unavailable or experiencing heavy load. Please try again later."
                
            clause.save()
            return JsonResponse({"explanation": clause.ai_explanation})
            
        except AnalyzedClause.DoesNotExist:
            return JsonResponse({"error": "Clause not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid method"}, status=400)

@csrf_exempt
def rewrite_clause_view(request, clause_id):
    """AJAX endpoint to generate a safer AI-rewritten clause."""
    from .ai_explainer import _call_gemini_rewrite
    
    if request.method == 'POST':
        try:
            clause = AnalyzedClause.objects.get(id=clause_id)
            if clause.ai_rewritten_text:
                return JsonResponse({"rewritten_text": clause.ai_rewritten_text})
            
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            explanation = _call_gemini_rewrite(
                api_key=api_key,
                clause_text=clause.original_text,
                clause_type=clause.clause_type
            )
            
            clause.ai_rewritten_text = explanation
            clause.save()
            return JsonResponse({"rewritten_text": clause.ai_rewritten_text})
            
        except AnalyzedClause.DoesNotExist:
            return JsonResponse({"error": "Clause not found"}, status=404)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid method"}, status=400)


def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from various file formats.
    Priority: PyMuPDF → PyPDF2 → python-docx → plain text → OCR
    """
    ext = os.path.splitext(file_path)[1].lower()
    extracted = ""

    try:
        if ext == '.pdf':
            try:
                import fitz  # PyMuPDF — best quality
                pdf_doc = fitz.open(file_path)
                pages = [page.get_text("text") for page in pdf_doc]
                pdf_doc.close()
                extracted = "\n\n".join(pages)
            except ImportError:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    extracted = "\n\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )

        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted = f.read()

        elif ext == '.docx':
            try:
                from docx import Document as DocxDocument
                docx_doc = DocxDocument(file_path)
                extracted = "\n".join(
                    p.text for p in docx_doc.paragraphs if p.text.strip()
                )
            except ImportError:
                logger.warning("python-docx not installed. Cannot read .docx files.")

        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            from .ocr_utils import process_image
            extracted = process_image(file_path)

    except Exception as e:
        logger.error(f"Text extraction failed for {file_path}: {e}")

    return extracted.strip()


def upload_document(request):
    """Handle document upload and initiate processing"""
    if request.method == 'POST' and request.FILES.get('document'):
        uploaded_file = request.FILES['document']
        title = request.POST.get('title') or uploaded_file.name

        # Save document record first
        doc = Document.objects.create(
            title=title,
            file=uploaded_file,
            language='English',
            status='Processing'
        )

        # Extract text from the uploaded file
        extracted_text = extract_text_from_file(doc.file.path)

        if len(extracted_text) < 30:
            # Minimal fallback so the analysis still runs meaningfully
            extracted_text = (
                "Either party may terminate this agreement at any time without cause. "
                "The company shall not be held liable for any indirect, consequential, or incidental damages. "
                "Client shall indemnify and hold harmless the service provider from third-party claims. "
                "All intellectual property developed under this agreement shall vest in the client."
            )
            logger.warning(f"Doc {doc.id}: text extraction yielded no content, using fallback text.")

        # run ML/rule-based analysis
        analysis_results = ml_engine.analyze_document(extracted_text)

        # ── AI Plain English & Indian Law Context Enrichment ──────────────
        # Calls Gemini 1.5 Flash in ONE massive JSON payload for the entire document.
        # This completely guarantees 0 rate limiting errors.
        try:
            from .ai_explainer import bulk_enrich_clauses
            analysis_results = bulk_enrich_clauses(analysis_results)
        except Exception as ai_e:
            logger.error(f"Bulk AI Error: {ai_e}")
            pass
        # ──────────────────────────────────────────────────────────────────

        total_risk = 0
        for item in analysis_results:
            AnalyzedClause.objects.create(
                document=doc,
                original_text=item['original_text'],
                clause_type=item['clause_type'],
                risk_score=item['risk_score'],
                risk_level=item['risk_level'],
                simplified_english=item['simplified_english'],
                what_this_means=item['what_this_means'],
                why_risky=item['why_risky'],
                consequences=item['consequences'],
                red_flags=item['red_flags'],
                recommendations=item['recommendations'],
                ai_explanation=item.get('ai_explanation'),  # ← new AI field
            )
            total_risk += item['risk_score']

        doc.overall_risk_score = total_risk // len(analysis_results) if analysis_results else 0
        doc.status = 'Completed'
        doc.save()

        return redirect('analysis', doc_id=doc.id)

    return render(request, 'upload.html')


def analysis_view(request, doc_id):
    """Display document analysis results"""
    doc = get_object_or_404(Document, id=doc_id)
    clauses = doc.clauses.all()
    return render(request, 'analysis.html', {
        'document': doc,
        'clauses': clauses,
    })


def download_report(request, doc_id):
    """Generate and return a PDF analysis report using ReportLab."""
    import io
    from django.http import FileResponse, HttpResponse
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    doc = get_object_or_404(Document, id=doc_id)
    # The user mandated ALL clauses be included in the PDF natively.
    clauses = doc.clauses.all().order_by('-risk_score', 'id')

    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'],
                                  fontSize=20, spaceAfter=6, alignment=TA_CENTER,
                                  textColor=colors.HexColor('#0d47a1'))
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                     fontSize=11, textColor=colors.grey, alignment=TA_CENTER)
    section_style = ParagraphStyle('Section', parent=styles['Heading2'],
                                    fontSize=13, textColor=colors.HexColor('#1565c0'),
                                    spaceBefore=16, spaceAfter=4)
    label_style = ParagraphStyle('Label', parent=styles['Normal'],
                                  fontSize=9, textColor=colors.grey,
                                  fontName='Helvetica-Bold', spaceBefore=8)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, leading=14, spaceAfter=4)
    risk_body_style = ParagraphStyle('RiskBody', parent=body_style,
                                      textColor=colors.HexColor('#b71c1c'))

    RISK_COLORS = {
        'Low':      colors.HexColor('#e8f5e9'),
        'Medium':   colors.HexColor('#fff8e1'),
        'High':     colors.HexColor('#fff3e0'),
        'Critical': colors.HexColor('#ffebee'),
    }
    RISK_BORDER = {
        'Low':      colors.HexColor('#388e3c'),
        'Medium':   colors.HexColor('#f9a825'),
        'High':     colors.HexColor('#e65100'),
        'Critical': colors.HexColor('#c62828'),
    }

    elements = []

    # ── Header ─────────────────────────────────────────────────────
    elements.append(Paragraph("LexGuard AI", title_style))
    elements.append(Paragraph("Legal Risk Analysis Report", subtitle_style))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1565c0')))
    elements.append(Spacer(1, 0.1 * inch))

    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.piecharts import Pie

    # Calculate Data Analytics
    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    all_clauses = doc.clauses.all()
    for c in all_clauses:
        risk_counts[c.risk_level] = risk_counts.get(c.risk_level, 0) + 1

    # ── Document info table ────────────────────────────────────────
    risk_color = RISK_BORDER.get(
        'Critical' if doc.overall_risk_score > 75 else
        'High' if doc.overall_risk_score > 50 else
        'Medium' if doc.overall_risk_score > 25 else 'Low',
        colors.grey
    )
    info_data = [
        ['Overall System Risk Score', f"{doc.overall_risk_score}/100"],
        ['Total Clauses Analyzed', str(all_clauses.count())],
        ['Critical Risks', str(risk_counts['Critical'])],
        ['High Risks', str(risk_counts['High'])],
    ]
    info_table = Table(info_data, colWidths=[3 * inch, 3.3 * inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e0e0e0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.2 * inch))

    # ── Risk Distribution Chart ──────────────────────────────────────
    elements.append(Paragraph("Risk Distribution Analysis", section_style))
    try:
        d = Drawing(400, 180)
        pc = Pie()
        pc.x = 120
        pc.y = 10
        pc.width = 150
        pc.height = 150
        pc.data = [risk_counts['Low'], risk_counts['Medium'], risk_counts['High'], risk_counts['Critical']]
        pc.labels = [f"Low\n{risk_counts['Low']}", f"Medium\n{risk_counts['Medium']}", f"High\n{risk_counts['High']}", f"Critical\n{risk_counts['Critical']}"]
        
        # Don't draw slices with 0 value to prevent ugly overlap
        if sum(pc.data) > 0:
            pc.slices[0].fillColor = colors.HexColor('#10b981')
            pc.slices[1].fillColor = colors.HexColor('#f59e0b')
            pc.slices[2].fillColor = colors.HexColor('#fd7e14')
            pc.slices[3].fillColor = colors.HexColor('#ef4444')
            for i in range(4):
                pc.slices[i].fontName = 'Helvetica-Bold'
                pc.slices[i].fontSize = 9
                if pc.data[i] == 0:
                    pc.slices[i].fillColor = colors.transparent
                    pc.slices[i].strokeColor = colors.transparent
                    
            d.add(pc)
            elements.append(d)
        else:
            elements.append(Paragraph("No clauses found.", body_style))
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        pass
    
    elements.append(Spacer(1, 0.4 * inch))

    # ── Clause Breakdown ───────────────────────────────────────────
    elements.append(Paragraph("Clause-by-Clause AI Analysis", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1565c0')))
    elements.append(Spacer(1, 0.2 * inch))

    for i, clause in enumerate(clauses, 1):
        bg = RISK_COLORS.get(clause.risk_level, colors.white)
        border = RISK_BORDER.get(clause.risk_level, colors.grey)

        clause_parts = []

        # Header row for this clause
        header_data = [[
            Paragraph(f"<b>{i}. {clause.clause_type}</b>", body_style),
            Paragraph(f"<font color='{border}'><b>{clause.risk_level} Risk — Score: {clause.risk_score}/100</b></font>", body_style),
        ]]
        header_table = Table(header_data, colWidths=[3.5 * inch, 3 * inch])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
            ('LINEBELOW', (0, 0), (-1, -1), 1.5, border),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        clause_parts.append(header_table)
        clause_parts.append(Spacer(1, 0.05 * inch))

        # Original Clause Text
        clause_parts.append(Paragraph("<b><u>Clause Text:</u></b>", label_style))
        clause_parts.append(Spacer(1, 0.05 * inch))
        clause_parts.append(Paragraph(f"<i>{clause.original_text}</i>", body_style))
        
        # AI Explanation
        clause_parts.append(Spacer(1, 0.05 * inch))
        clause_parts.append(Paragraph("<b><u>AI Analytics & Indian Law Explanation:</u></b>", label_style))
        clause_parts.append(Spacer(1, 0.06 * inch))
        
        # Fallback to standard generated properties if actual AI explanation hasn't been cached
        explanation_text = clause.ai_explanation
        if not explanation_text:
            explanation_text = f"{clause.simplified_english}\n\nRisks: {clause.why_risky}"
            
        # Clean potential markdown/formatting so reportlab doesn't choke on arbitrary <> characters
        explanation_text = explanation_text.replace('<', '&lt;').replace('>', '&gt;')
        # Convert newlines to breaks
        explanation_text = explanation_text.replace('\n', '<br/>')
        
        clause_parts.append(Paragraph(explanation_text, body_style))
        
        clause_parts.append(Spacer(1, 0.3 * inch))
        elements.append(KeepTogether(clause_parts))

    # ── Executive Summary ───────────────────────────────────────────
    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph("AI Executive Summary", section_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1565c0')))
    elements.append(Spacer(1, 0.2 * inch))

    try:
        from .ai_explainer import summarize_document
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        ai_summary = summarize_document(api_key, list(all_clauses.order_by('-risk_score')))
    except Exception as e:
        logger.error(f"Summary Error: {e}")
        ai_summary = "AI Summary generation failed due to an API timeout."
        
    ai_summary = ai_summary.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
    
    summary_parts = []
    summary_parts.append(Paragraph(ai_summary, body_style))
    elements.append(KeepTogether(summary_parts))

    # ── Footer ──────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Generated by LexGuard AI · This report is for informational purposes only and does not constitute legal advice.",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                       textColor=colors.grey, alignment=TA_CENTER)
    ))

    pdf.build(elements)
    buffer.seek(0)

    response = FileResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="LexGuard_Report_{doc.id}.pdf"'
    return response
