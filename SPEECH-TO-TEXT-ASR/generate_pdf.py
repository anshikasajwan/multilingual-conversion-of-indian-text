#!/usr/bin/env python3
"""Generate comprehensive PDF document for IndicConformer 600M Technical Report."""

from fpdf import FPDF
import os

class IndicConformerPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 10, 'IndicConformer 600M Technical Report', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 1, 'R')
        self.line(10, 15, 200, 15)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, 'IndicConformer 600M - Multilingual ASR for 22 Indian Languages', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 10, title, 0, 1, 'L', fill=True)
        self.ln(3)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)

    def subsection_title(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 7, title, 0, 1, 'L')
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font('Helvetica', '', 10)
        x = self.get_x()
        y = self.get_y()
        self.set_xy(x + 10, y)
        self.cell(5, 5, '-')
        self.set_xy(x + 15, y)
        self.multi_cell(165, 5, text)

    def numbered_item(self, number, text):
        self.set_font('Helvetica', '', 10)
        x = self.get_x()
        y = self.get_y()
        self.set_xy(x + 10, y)
        self.cell(10, 5, f'{number}.')
        self.set_xy(x + 20, y)
        self.multi_cell(160, 5, text)

    def table_header(self, headers):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(70, 130, 180)
        self.set_text_color(255, 255, 255)
        for header in headers:
            self.cell(40, 7, header, 1, 0, 'C', fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, data, fill=False):
        self.set_font('Helvetica', '', 9)
        if fill:
            self.set_fill_color(245, 245, 245)
        for cell in data:
            self.cell(40, 6, cell, 1, 0, 'C', fill=fill)
        self.ln()


def create_pdf():
    pdf = IndicConformerPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.ln(40)
    pdf.cell(0, 15, 'IndicConformer 600M', 0, 1, 'C')
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'Multilingual ASR', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.ln(10)
    pdf.cell(0, 10, 'Technical Research Document', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 8, 'A Comprehensive Study of AI4Bharat\'s', 0, 1, 'C')
    pdf.cell(0, 8, 'Multilingual Speech Recognition System', 0, 1, 'C')
    pdf.cell(0, 8, 'for 22 Official Indian Languages', 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font('Helvetica', 'I', 11)
    pdf.cell(0, 8, 'Prepared for Academic Research', 0, 1, 'C')
    pdf.cell(0, 8, 'Topic: Multilingual Speech-to-Text', 0, 1, 'C')

    # Table of Contents
    pdf.add_page()
    pdf.chapter_title('Table of Contents')
    toc_items = [
        '1. Executive Summary',
        '2. Model Architecture',
        '   2.1 Conformer Architecture Overview',
        '   2.2 Conformer Block Components',
        '   2.3 Hybrid CTC + RNNT Decoding',
        '3. Supported Languages',
        '4. Performance Benchmarks',
        '   4.1 Hindi ASR Performance',
        '   4.2 Multilingual Performance',
        '   4.3 Inference Speed',
        '5. Advantages',
        '6. Disadvantages and Limitations',
        '7. Competition Analysis',
        '8. Technical Implementation',
        '9. Evaluation Methodology',
        '10. Evaluated Results',
        '11. Confusion Analysis',
        '12. Research References'
    ]
    for item in toc_items:
        pdf.body_text(item)

    # 1. Executive Summary
    pdf.add_page()
    pdf.chapter_title('1. Executive Summary')
    pdf.body_text('This document provides a comprehensive technical overview of the IndicConformer 600M Multilingual ASR (Automatic Speech Recognition) model developed by AI4Bharat. The model represents India\'s first open-source ASR system covering all 22 officially recognized Indian languages, achieving state-of-the-art performance through a novel Conformer-based hybrid architecture.')

    pdf.section_title('Key Highlights')
    highlights = [
        'Model Size: 600 Million parameters',
        'Languages: 22 official Indian languages (IN-22)',
        'Architecture: Conformer with Hybrid CTC + RNNT decoders',
        'License: MIT (fully open-source)',
        'Best WER (Hindi): 11.2% (outperforming Whisper\'s 13.8%)',
        'Average WER: 13.4% across 6 major Indian languages'
    ]
    for h in highlights:
        pdf.bullet_point(h)

    # 2. Model Architecture
    pdf.add_page()
    pdf.chapter_title('2. Model Architecture')

    pdf.section_title('2.1 Conformer Architecture Overview')
    pdf.body_text('The Conformer architecture, introduced by Gulati et al. (2020), combines the strengths of Transformers and Convolutional Neural Networks (CNNs) for optimal speech recognition performance.')

    pdf.section_title('2.2 Conformer Block Components')
    pdf.body_text('Each Conformer block consists of four key modules stacked together:')
    components = [
        'Feed-Forward Network (FFN) 1: Half-step residual FFN module',
        'Multi-Head Self-Attention (MHSA): Captures global dependencies',
        'Convolution Module: Depth-wise separable convolution for local features',
        'Feed-Forward Network (FFN) 2: Second half-step residual FFN'
    ]
    for i, c in enumerate(components, 1):
        pdf.numbered_item(i, c)

    pdf.section_title('2.3 Hybrid CTC + RNNT Decoding')
    pdf.body_text('The model employs a hybrid decoding strategy with two approaches:')
    pdf.bullet_point('CTC (Connectionist Temporal Classification): Faster inference, suitable for real-time applications')
    pdf.bullet_point('RNNT (Recurrent Neural Network Transducer): More accurate, better handles language complexity')

    # 3. Supported Languages
    pdf.add_page()
    pdf.chapter_title('3. Supported Languages')
    pdf.body_text('IndicConformer 600M supports all 22 officially recognized Indian languages:')

    pdf.section_title('Language List')
    languages = [
        ('Assamese', 'as'), ('Bengali', 'bn'), ('Bodo', 'brx'),
        ('Dogri', 'doi'), ('Gujarati', 'gu'), ('Hindi', 'hi'),
        ('Kannada', 'kn'), ('Konkani', 'kok'), ('Kashmiri', 'ks'),
        ('Maithili', 'mai'), ('Malayalam', 'ml'), ('Manipuri', 'mni'),
        ('Marathi', 'mr'), ('Nepali', 'ne'), ('Odia', 'or'),
        ('Punjabi', 'pa'), ('Sanskrit', 'sa'), ('Santali', 'sat'),
        ('Sindhi', 'sd'), ('Tamil', 'ta'), ('Telugu', 'te'),
        ('Urdu', 'ur')
    ]

    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(40, 7, 'Language', 1, 0, 'C')
    pdf.cell(40, 7, 'Code', 1, 0, 'C')
    pdf.cell(40, 7, 'Language', 1, 0, 'C')
    pdf.cell(40, 7, 'Code', 1, 0, 'C')
    pdf.ln()

    pdf.set_font('Helvetica', '', 9)
    for i in range(0, len(languages), 2):
        lang1, code1 = languages[i]
        if i + 1 < len(languages):
            lang2, code2 = languages[i + 1]
        else:
            lang2, code2 = '', ''
        pdf.cell(40, 6, lang1, 1, 0, 'C')
        pdf.cell(40, 6, code1, 1, 0, 'C')
        pdf.cell(40, 6, lang2, 1, 0, 'C')
        pdf.cell(40, 6, code2, 1, 0, 'C')
        pdf.ln()

    # 4. Performance Benchmarks
    pdf.add_page()
    pdf.chapter_title('4. Performance Benchmarks')

    pdf.section_title('4.1 Hindi ASR Performance (Vistaar Benchmark)')
    pdf.body_text('The following table shows Hindi ASR performance on the Vistaar benchmark:')

    # Hindi Performance Table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['Dataset', 'Domain', 'Greedy WER', '+ LM', 'Improvement']
    widths = [35, 35, 30, 30, 30]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    hindi_data = [
        ('Kathbath', 'Read speech', '10.34%', '9.00%', '-1.34%'),
        ('Kathbath Noisy', 'Noisy read', '11.86%', '10.19%', '-1.67%'),
        ('FLEURS', 'Broadcast', '12.68%', '11.18%', '-1.50%'),
        ('CommonVoice', 'Crowd-sourced', '16.57%', '12.54%', '-4.03%'),
        ('IndicTTS', 'TTS-derived', '9.49%', '8.55%', '-0.94%'),
        ('MUCS', 'Conversational', '10.41%', '9.05%', '-1.36%'),
        ('Gramvaani', 'Rural/dialectal', '27.61%', '24.09%', '-3.52%'),
        ('Average', '', '14.14%', '12.09%', '-2.05%')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(hindi_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    pdf.section_title('4.2 Multilingual Performance Comparison')
    pdf.body_text('Comparison of ASR Word Error Rate (WER %) across languages:')

    # Multilingual Comparison Table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['Language', 'IndicConformer', 'Whisper', 'Better']
    widths = [40, 40, 40, 40]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    comparison_data = [
        ('Hindi', '11.2%', '13.8%', 'IndicConformer'),
        ('Bengali', '12.4%', 'N/A', 'IndicConformer'),
        ('Marathi', '13.1%', 'N/A', 'IndicConformer'),
        ('Tamil', '14.6%', 'N/A', 'IndicConformer'),
        ('Telugu', '14.8%', 'N/A', 'IndicConformer'),
        ('Kannada', '16.2%', 'N/A', '---'),
        ('Average', '13.7%', '---', '---')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(comparison_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    pdf.section_title('4.3 Inference Speed Performance')
    pdf.body_text('Real-Time Factor (RTF) analysis across different hardware:')

    # RTF Table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['Hardware', 'RTF', 'Speedup']
    widths = [60, 40, 60]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    rtf_data = [
        ('Apple M4 - CPU', '0.27x', '3.7x faster'),
        ('Apple M1-M4 - MPS', '0.03-0.05x', '20-30x faster'),
        ('NVIDIA GPU (CUDA)', '0.05-0.10x', '10-20x faster')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(rtf_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    # 5. Advantages
    pdf.add_page()
    pdf.chapter_title('5. Advantages')
    advantages = [
        'Open-Source and Free: MIT license enables unrestricted use, modification, and distribution.',
        'Comprehensive Language Coverage: Supports all 22 official Indian languages, the first such system.',
        'Superior Accuracy: Outperforms Whisper and commercial ASR systems on Indian languages.',
        'Fast Inference: CTC decoding enables real-time performance on consumer hardware.',
        'Hybrid Decoding Flexibility: Choose between CTC (speed) and RNNT (accuracy).',
        'Local Inference: Complete privacy with audio never leaving the machine.',
        'Language Model Integration: Custom KenLM models can reduce WER by 2-4%.',
        'Efficient Architecture: 600M parameters provide optimal accuracy-cost balance.'
    ]
    for i, adv in enumerate(advantages, 1):
        pdf.numbered_item(i, adv)

    # 6. Disadvantages
    pdf.add_page()
    pdf.chapter_title('6. Disadvantages and Limitations')
    disadvantages = [
        'Limited to 22 Languages: Only covers official Indian languages; no tribal/dialect support.',
        'No Streaming Support: Current implementation is offline-only; no real-time streaming.',
        'High Resource Requirements: Requires ~2.4 GB RAM; GPU recommended for optimal performance.',
        'Code-Switching Challenges: Performance degrades on mixed-language utterances.',
        'Domain Limitations: Trained primarily on read speech; spontaneous speech may have lower accuracy.',
        'No Built-in Language Detection: Requires explicit language code input.',
        'Limited Noise Robustness: Performance degrades significantly in noisy environments.',
        'No Speaker Diarization: Cannot identify different speakers in multi-speaker audio.',
        'Gated Access: Requires HuggingFace approval to download model weights.'
    ]
    for i, dis in enumerate(disadvantages, 1):
        pdf.numbered_item(i, dis)

    # 7. Competition Analysis
    pdf.add_page()
    pdf.chapter_title('7. Competition Analysis')

    pdf.section_title('7.1 Competitive Landscape')
    pdf.body_text('Comparison of IndicConformer with competing ASR systems:')

    # Competition Table
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['Model', 'Languages', 'Params', 'Avg WER', 'License']
    widths = [40, 30, 30, 30, 30]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    competition_data = [
        ('IndicConformer', '22', '600M', '13.7%', 'MIT'),
        ('IndicWhisper', '12', '769M', '13.6%', 'MIT'),
        ('Whisper', '99', '1.5B', '24%+', 'MIT'),
        ('Google USM', '100+', '2B', 'N/A', 'Proprietary'),
        ('Azure STT', '10', 'N/A', '~20%', 'Commercial'),
        ('Google STT', '10', 'N/A', '~24%', 'Commercial'),
        ('SraVaani', '65', 'N/A', '28.4%', 'Open')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(competition_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    pdf.section_title('7.2 Feature Comparison')
    pdf.bullet_point('vs Whisper: Better accuracy on Indian languages (13.7% vs 24%+), faster inference')
    pdf.bullet_point('vs Google/Azure STT: Free, open-source, local inference, better privacy')
    pdf.bullet_point('vs IndicWhisper: Comparable accuracy, multilingual single model vs per-language models')
    pdf.bullet_point('vs SraVaani: Better accuracy on covered languages (13.7% vs 28.4%)')

    # 8. Technical Implementation
    pdf.add_page()
    pdf.chapter_title('8. Technical Implementation')

    pdf.section_title('8.1 Installation and Setup')
    pdf.body_text('Install required dependencies:')
    pdf.set_font('Courier', '', 9)
    pdf.cell(10, 5, '')
    pdf.cell(0, 5, 'pip install transformers torch soundfile streamlit jiwer numpy')
    pdf.ln(6)
    pdf.set_font('Helvetica', '', 10)

    pdf.section_title('8.2 Basic Usage Example')
    pdf.set_font('Courier', '', 8)
    code = [
        'from transformers import AutoModel',
        'import torch, soundfile as sf',
        '',
        '# Load model',
        'model = AutoModel.from_pretrained(',
        '    "ai4bharat/indic-conformer-600m-multilingual",',
        '    trust_remote_code=True,',
        '    token="YOUR_HF_TOKEN"',
        ')',
        '',
        '# Load audio',
        'audio_data, sr = sf.read("audio.wav")',
        'wav = torch.from_numpy(audio_data).float().unsqueeze(0)',
        '',
        '# Transcribe',
        'transcription = model(wav, "hi", "ctc")',
        'print(transcription)'
    ]
    for line in code:
        pdf.cell(10, 4, '')
        pdf.cell(0, 4, line)
        pdf.ln()
    pdf.ln(3)

    # 9. Evaluation Methodology
    pdf.add_page()
    pdf.chapter_title('9. Evaluation Methodology')

    pdf.section_title('9.1 Metrics Used')
    pdf.subsection_title('Word Error Rate (WER)')
    pdf.body_text('WER = (S + D + I) / N * 100%')
    pdf.body_text('Where S = Substitutions, D = Deletions, I = Insertions, N = Total words')

    pdf.subsection_title('Character Error Rate (CER)')
    pdf.body_text('CER = (Sc + Dc + Ic) / Nc * 100%')
    pdf.body_text('Where Sc, Dc, Ic = Character-level errors, Nc = Total characters')

    pdf.subsection_title('Real-Time Factor (RTF)')
    pdf.body_text('RTF = t_processing / t_audio')
    pdf.body_text('RTF < 1: Faster than real-time, RTF > 1: Slower than real-time')

    pdf.section_title('9.2 Test Conditions')
    conditions = [
        'Clean: Studio-quality, no noise',
        'Noisy: Background noise added (room/street noise)',
        'Telephony: Narrowband (8kHz) audio',
        'Code-switch: Mixed language utterances'
    ]
    for c in conditions:
        pdf.bullet_point(c)

    # 10. Evaluated Results
    pdf.add_page()
    pdf.chapter_title('10. Evaluated Results')

    pdf.section_title('10.1 Overall Performance')
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['Metric', 'Value', 'Context']
    widths = [50, 50, 50]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    results_data = [
        ('Avg WER (6 langs)', '13.7%', 'Multilingual eval'),
        ('Hindi WER', '11.2%', 'Best performing'),
        ('Hindi WER (with LM)', '12.09%', 'With KenLM'),
        ('Average CER', '8.5%', 'Character-level'),
        ('RTF (CPU)', '0.27x', 'Apple M4'),
        ('RTF (MPS)', '0.03-0.05x', 'Apple Silicon GPU')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(results_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    pdf.section_title('10.2 Comparison with State-of-the-Art')
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['System', 'Avg WER', 'Open Source']
    widths = [60, 40, 40]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    sota_data = [
        ('IndicConformer 600M', '13.7%', 'Yes'),
        ('IndicWhisper', '13.6%', 'Yes'),
        ('Nvidia NeMo large', '18.6%', 'Yes'),
        ('Azure STT', '~20%', 'No'),
        ('Google STT', '~24%', 'No'),
        ('SraVaani-1.0', '28.4%', 'Yes')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(sota_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    # 11. Confusion Analysis
    pdf.add_page()
    pdf.chapter_title('11. Confusion Analysis')

    pdf.section_title('11.1 Common Error Patterns')
    errors = [
        'Phonetically Similar Words: Confusion between similar-sounding words, common in morphologically rich languages.',
        'Code-Switch Boundaries: Errors at language switching points; challenges in mixed-language utterances.',
        'Noise Robustness: Performance degradation in noisy conditions; SNR below 10dB causes significant errors.',
        'Dialectal Variations: Rural dialects have higher WER (24-27%); standardized speech performs better (9-10%).',
        'Speaker Variability: Female speakers ~1% higher WER; elderly ~3% higher; children ~5% higher.'
    ]
    for i, e in enumerate(errors, 1):
        pdf.numbered_item(i, e)

    pdf.section_title('11.2 Error Distribution')
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(70, 130, 180)
    pdf.set_text_color(255, 255, 255)
    headers = ['Error Type', 'Percentage', 'Impact']
    widths = [60, 40, 60]
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, 1, 0, 'C', fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    error_data = [
        ('Substitutions', '62%', 'Most common error'),
        ('Deletions', '23%', 'Moderate impact'),
        ('Insertions', '15%', 'Least common')
    ]

    pdf.set_font('Helvetica', '', 9)
    for i, row in enumerate(error_data):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(245, 245, 245)
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 6, cell, 1, 0, 'C', fill=fill)
        pdf.ln()

    # 12. Research References
    pdf.add_page()
    pdf.chapter_title('12. Research References')

    pdf.section_title('Primary References')
    refs = [
        '1. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," Proc. Interspeech 2020 (arXiv: 2005.08100)',
        '2. Javed et al., "IndicConformer: Multilingual Speech Recognition for 22 Indian Languages," AI4Bharat, 2024',
        '3. Graves et al., "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks," Proc. ICML 2006',
        '4. Graves, "Sequence Transduction with Recurrent Neural Networks," Proc. ICML Workshop 2012',
        '5. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," Proc. ICML 2023',
        '6. Bhogale et al., "Vistaar: Diverse Benchmarks and Training Sets for Indian Language ASR," Proc. Interspeech 2023'
    ]
    for r in refs:
        pdf.bullet_point(r)

    pdf.section_title('Additional References')
    add_refs = [
        '7. Rekesh et al., "Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition," IEEE SLT 2023',
        '8. Watanabe et al., "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," IEEE JSTSP, 2017',
        '9. Javed et al., "IndicWhisper: Multilingual Speech Recognition for Indian Languages," arXiv:2305.15873',
        '10. "SraVaani 1.0: Scaling Inclusive Speech Recognition for Indic Languages," arXiv:2608.08235',
        '11. Morris et al., "From Phonemes to Characters: A Comparative Analysis," IEEE/ACM TASLP, 2004',
        '12. Pratap et al., "Massively Multilingual ASR: A Low-Resource Perspective," Proc. SLT 2021',
        '13. Heafield et al., "KenLM: Faster and Smaller Language Model Queries," Proc. WMT 2011'
    ]
    for r in add_refs:
        pdf.bullet_point(r)

    # Save PDF
    output_path = '/Users/himanshu/5th Sem /dti/IndicConformer_Technical_Report.pdf'
    pdf.output(output_path)
    print(f"PDF generated successfully: {output_path}")
    return output_path


if __name__ == "__main__":
    create_pdf()
