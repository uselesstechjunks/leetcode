from custom_directives import DefinitionsDirective

extensions = ['sphinx_rtd_theme','sphinx.ext.autosectionlabel']

'''
Markdown helps:
https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-admonition
'''

def setup(app):
	app.add_directive('definitions', DefinitionsDirective)
