from django import forms
# Define las opciones para las herramientas y el formato de salida
DELIVERY_OPTIONS = [
    ('google_docs', 'Google Docs (Editable)'),
    ('notion', 'Notion (Editable)'),
    ('powerpoint', 'PowerPoint (Editable)'),
    ('pdf', 'PDF (Visual Final)'),
]

class FileInputForm(forms.Form):
    """ 
    ARCHIVO_CHOICES = [
        ('google_docs', 'Google Docs (Editable)'),
        ('notion', 'Notion (Editable)'),
        ('powerpoint', 'PowerPoint (Editable)'),
        ('pdf', 'PDF (Visual Final)'),
    ] """
    #tipo = forms.ChoiceField(choices=ARCHIVO_CHOICES)
    archivo = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))