"""Environment contract checks for the actual IR pathname override."""
from pathlib import Path
import unittest
import re
import vm

ROOT = Path(__file__).resolve().parents[3]


def validate_probe(source):
    graph = source[source.index('static void graph('):source.index('static void kernels(')]
    clear = 'unsetenv("EMBER_IR");'
    constructor = 'tinyllama::TinyLlamaSession s('
    if clear not in graph or graph.index(clear) > graph.index(constructor):
        raise ValueError('probe must clear pathname override before model construction')
    if re.search(r'\bsetenv\("EMBER_IR"', graph):
        raise ValueError('probe must not write Boolean-like pathname overrides')


class EnvironmentTests(unittest.TestCase):
    def test_controller_clears_inherited_pathnames(self):
        for value in ('0', '', '/tmp/not-a-model-ir.json'):
            with self.subTest(value=value):
                env = vm.runtime_environment({'EMBER_IR': value, 'KEEP_TEST': 'unchanged'})
                self.assertNotIn('EMBER_IR', env)
                self.assertEqual(env['KEEP_TEST'], 'unchanged')
                self.assertEqual(env['EMBER_DETERMINISTIC'], '1')
                self.assertEqual(env['EMBER_DECODE_PICK_PARTS'], '4')

    def test_probe_clears_before_constructor(self):
        validate_probe(Path(__file__).with_name('integrated_probe.cpp').read_text())

    def test_v1_boolean_override_is_rejected(self):
        current = Path(__file__).with_name('integrated_probe.cpp').read_text()
        broken = current.replace('unsetenv("EMBER_IR");', 'setenv("EMBER_IR", "0", 1);')
        with self.assertRaises(ValueError): validate_probe(broken)

    def test_production_interprets_even_zero_as_path(self):
        source = (ROOT / 'graph_ir.cpp').read_text()
        start = source.index('std::string find_ir(const GGUFData& gguf)')
        end = source.index('auto it = gguf.metadata.find', start)
        override = source[start:end]
        self.assertIn('if (const char* path = std::getenv("EMBER_IR"))', override)
        self.assertIn('std::ifstream f(path);', override)
        self.assertIn('EMBER_IR: cannot read ', override)
        self.assertNotIn('std::string(path) == "0"', override)

if __name__ == '__main__': unittest.main()
