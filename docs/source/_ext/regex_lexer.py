# ------------------------------------------------------------------------------
#  Clipppy.
#  Copyright (C) 2021 Kosio Karchev
#
#  This program is free software: you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by the Free
#  Software Foundation, either version 3 of the License, or (at your option)
#  any later version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
#  more details.
#
#  You should have received a copy of the GNU General Public License along with
#  this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

import re

import pygments.lexers
import sphinx.highlighting
from pygments.lexer import bygroups, RegexLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String
from sphinx.application import Sphinx


class regexLexer(RegexLexer):
    name = 'regex'

    aliases = 'regex',

    tokens = {
        'root': [
            ('|'.join(map(re.escape, (*r'.*+?|-', '*?', '+?', '??'))), Operator),
            (r'({)(\d+)(?P<second>(,)(\d+))?(})(?(second)\??|)', bygroups(Punctuation, Number, None, Punctuation, Number, Punctuation, Operator)),
            (r'[\$\^]', Name.Builtin),
            ('|'.join(map(re.escape, ('\\'+_ for _ in 'AbBdDsSwWZ'))), Name.Builtin),
            ('\\\\.', String.Escape),
            ('\(\?\#.*?\)', Comment),
            (r'\(\?(?:\:|\=|\!|\<\=|\<\!)', Keyword.Namespace),
            (r'(\(\?P)(\<)([_\w\d][\w\d]*)(\>)', bygroups(Keyword.Namespace, Punctuation, Name.Variable, Punctuation)),
            (r'(\(\?)(\()([_\w\d][\w\d]*)(\))', bygroups(Keyword.Namespace, Punctuation, Name.Variable, Punctuation)),
            (r'(\(\?P)(=)([_\w\d][\w\d]*)', bygroups(Keyword.Namespace, Punctuation, Name.Variable)),
            (r'\(|\)|\[\^?|\]', Keyword.Namespace),
            (r'.', String.Single),
        ],
    }


def setup(app: Sphinx):
    sphinx.highlighting.lexers['regex'] = regexLexer()
    pygments.lexers.LEXERS['regexLexer'] = __name__, regexLexer.name, regexLexer.aliases, None, None
    pygments.lexers._lexer_cache['regex'] = regexLexer
