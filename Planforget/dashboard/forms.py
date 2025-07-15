from django import forms
# Define las opciones para las herramientas y el formato de salida
DELIVERY_OPTIONS = [
    ('google_docs', 'Google Docs (Editable)'),
    ('notion', 'Notion (Editable)'),
    ('powerpoint', 'PowerPoint (Editable)'),
    ('pdf', 'PDF (Visual Final)'),
]

class AnswerForm(forms.Form):
    objetive = forms.CharField(label='Campaign Type/Specific Objectives/Success Metrics', max_length=200)
    product = forms.CharField(label='Product/Service Description, Features & Differentiators', max_length=200)
    public = forms.CharField(label='Audience Demographics, Interests & Preferred Channels', max_length=200)
    content = forms.CharField(label='Desired Content Type, Format & Estimated Length', max_length=200)
    tone = forms.CharField(label='Preferred Tone & Style', max_length=200)
    history = forms.CharField(label='Previous Campaign Insights & Creative References', max_length=200)
    tendencies = forms.CharField(label='Market Trends & Business Insights', max_length=200)
    direction = forms.CharField(label='Stakeholder Feedback & Business Alignment', max_length=200)
    topic = forms.CharField(label='Content Taxonomy & Key Themes', max_length=200)
    result = forms.ChoiceField(label='Expected Delivery Tools & Output Format',choices=DELIVERY_OPTIONS,help_text='Select where the brief/content should be generated or integrated and its format.')

class FileInputForm(forms.Form):
    ARCHIVO_CHOICES = [
        ('google_docs', 'Google Docs (Editable)'),
        ('notion', 'Notion (Editable)'),
        ('powerpoint', 'PowerPoint (Editable)'),
        ('pdf', 'PDF (Visual Final)'),
    ]
    tipo = forms.ChoiceField(choices=ARCHIVO_CHOICES)
    archivo = forms.FileField()