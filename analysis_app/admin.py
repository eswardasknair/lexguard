from django.contrib import admin
from .models import Document, AnalyzedClause

class AnalyzedClauseInline(admin.StackedInline):
    model = AnalyzedClause
    extra = 0

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'status', 'language', 'overall_risk_score', 'uploaded_at')
    inlines = [AnalyzedClauseInline]

@admin.register(AnalyzedClause)
class AnalyzedClauseAdmin(admin.ModelAdmin):
    list_display = ('document', 'clause_type', 'risk_level', 'risk_score')
    list_filter = ('risk_level', 'clause_type')
