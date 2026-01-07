from django.db import models
from django.contrib.auth.models import User

class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='Pending') # Pending, Processing, Completed, Failed
    language = models.CharField(max_length=50, default='Unknown') # English, Malayalam, Unknown
    overall_risk_score = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return self.title

class AnalyzedClause(models.Model):
    document = models.ForeignKey(Document, related_name='clauses', on_delete=models.CASCADE)
    original_text = models.TextField()
    simplified_english = models.TextField(blank=True, null=True)
    
    risk_score = models.IntegerField(default=0) # 0 to 100
    risk_level = models.CharField(max_length=50, default='Low') # Low, Medium, High, Critical
    clause_type = models.CharField(max_length=100, default='General') # Termination, Liability, etc.

    # Advanced explanation fields
    what_this_means = models.TextField(blank=True, null=True)
    why_risky = models.TextField(blank=True, null=True)
    consequences = models.TextField(blank=True, null=True)
    red_flags = models.TextField(blank=True, null=True)
    recommendations = models.TextField(blank=True, null=True)

    # AI-generated plain English explanation (HuggingFace Mistral-7B)
    ai_explanation = models.TextField(blank=True, null=True)

    # NEW FEATURE: AI Negotiator rewritten safer clause
    ai_rewritten_text = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.clause_type} ({self.risk_level}) - {self.document.title}"

class LawResource(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    year = models.IntegerField(blank=True, null=True)
    
    def __str__(self):
        return self.title

class LawSection(models.Model):
    law = models.ForeignKey(LawResource, related_name='sections', on_delete=models.CASCADE)
    section_number = models.CharField(max_length=50) # e.g. "Section 4"
    title = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField()
    
    def __str__(self):
        return f"{self.law.title} - {self.section_number}"
