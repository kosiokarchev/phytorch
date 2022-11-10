from docutils.nodes import Text
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.transforms import SphinxTransform


class AnyTildeTransform(SphinxTransform):
    default_priority = 999

    def apply(self, **kwargs):
        for node in self.document.traverse(addnodes.pending_xref):
            if node['reftype'] == 'any' and node['reftarget'].startswith('~'):
                node['reftarget'] = node['reftarget'][1:]
                node.children[0].children[0] = Text(node.children[0].children[0].rsplit('.', 1)[-1])


def setup(app: Sphinx):
    app.add_transform(AnyTildeTransform)
