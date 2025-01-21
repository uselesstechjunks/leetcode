# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SelfStudyGuide'
copyright = '2022-2025, UselessTechJunks'
author = 'Useless Tech Junks'

release = '0.1'
version = '0.1.0'

# -- General configuration
from custom_directives import DefinitionsDirective

'''
Markdown helps:
https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-admonition
'''

def setup(app):
	app.add_directive('definitions', DefinitionsDirective)

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_toolbox.collapse',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

mathjax3_config = {
    'chtml' : {
        'mtextInheritFont' : 'true',
    }
}
