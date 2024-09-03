from custom_directives import DefinitionsDirective

extensions = ['sphinx_rtd_theme','sphinx.ext.autosectionlabel']

def setup(app):
	app.add_directive('definitions', DefinitionsDirective)
