from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('upload/', views.upload_document, name='upload'),
    path('analysis/<int:doc_id>/', views.analysis_view, name='analysis'),
    path('analysis/<int:doc_id>/download/', views.download_report, name='download_report'),
    path('lawbook/', views.law_book_list, name='lawbook'),
    path('api/explain_clause/<int:clause_id>/', views.generate_clause_explanation, name='api_explain_clause'),
    path('api/rewrite_clause/<int:clause_id>/', views.rewrite_clause_view, name='api_rewrite_clause'),
]
