from sphinx.addnodes import desc_addname
from sphinx.application import Sphinx
from sphinx.domains.python import PyVariable


class PTConstDirective(PyVariable):
    def run(self):
        self.name = 'py:data'
        res = super().run()
        for node in res[1].findall(desc_addname):
            node.parent.remove(node)
        return res



def setup(app: Sphinx):
    setup.app = app

    app.add_directive('ptconst', PTConstDirective)
