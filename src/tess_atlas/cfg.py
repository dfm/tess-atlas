c = get_config()
c.NbConvertApp.export_format = "html"
c.Exporter.preprocessors = [
    "nbconvert.preprocessors.ExtractOutputPreprocessor"
]
