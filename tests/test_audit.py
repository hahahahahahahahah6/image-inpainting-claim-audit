import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from image_inpainting_claim_audit import phrases
from image_inpainting_claim_audit.audit import audit, claims, evidence

INPAINTING = [
    'image inpainting', 'image-inpainting', 'imageinpainting',
    'image inpainting model', 'image inpainting models',
    'image inpainting system', 'image inpainting systems',
    'image inpainting feature', 'image inpainting capability',
    'image inpainting support', 'supports image inpainting',
    'image inpainting tool', 'image inpainting api', 'image inpainting pipeline',
    'image inpainting algorithm', 'image inpainting library',
    'image inpainting framework', 'image inpainting service',
    'image inpainting function', 'image inpainting engine',
    'image inpainting engines', 'image inpainter', 'image inpainters',
    'inpaints images', 'inpaint images', 'inpaints photos', 'inpaint photos',
    'inpaints pictures', 'inpaint pictures', 'photo inpainting',
    'photo-inpainting', 'photo inpainter', 'photo inpainters',
    'inpaints photo', 'inpaint photo', 'image hole filling', 'image hole filler',
    'image inpainting task', 'image inpainting benchmark',
    'automatic image inpainting', 'automated image inpainting',
    'inpaints image files', 'inpaint image files', 'image inpainting software',
    'image inpainting app', 'image inpainting program', 'inpaints your photos',
    'inpaint my photos', 'picture inpainting', 'fills missing image regions',
]
NOINPAINTING = [
    'does not support image inpainting', 'do not support image inpainting',
    "doesn't support image inpainting", "don't support image inpainting",
    'no image inpainting', 'without image inpainting',
    'never performs image inpainting', 'does not perform image inpainting',
    'image inpainting not supported', 'image inpainting unsupported',
    'no image inpainting support', 'lacks image inpainting',
    'missing image inpainting', 'image inpainting disabled',
    'no image inpainting capability', 'without image inpainting support',
    'does not provide image inpainting', 'no image inpainting model',
    'image inpainting not available', 'does not use image inpainting',
    'no image inpainter', 'without image inpainter',
    'does not support image inpainter', 'no photo inpainter',
]
GENERIC_NAMES = [
    'inpaint_image', 'image_inpainting', 'inpaint_photo', 'image_inpainter',
    'predict_inpainted_image', 'inpaint_picture', 'fill_missing_region',
    'fill_missing_regions', 'image_hole_filling', 'inpaint_image_file',
    'predict_filled_image', 'inpaint_grayscale', 'inpaint_color_image',
    'image_inpaint', 'photo_inpainting', 'photo_inpainter', 'inpaint_your_photo',
    'remove_object_from_image', 'image_object_removal', 'inpaint_masked_image',
    'masked_image_inpainting', 'predict_complete_image', 'inpaint_frame',
    'image_inpainting_model',
]
DIRECT_NAMES = ['inpaint', 'inpaint_biharmonic']
PIPELINE_CLASSES = ['StableDiffusionInpaintPipeline',
                    'StableDiffusionXLInpaintPipeline', 'AutoPipelineForInpainting']


class OracleTests(unittest.TestCase):
    def test_phrases(self):
        self.assertEqual(phrases.INPAINTING, INPAINTING)
        self.assertEqual(phrases.NOINPAINTING, NOINPAINTING)

    def test_generic_names(self):
        self.assertEqual(phrases.GENERIC_NAMES, GENERIC_NAMES)

    def test_direct_names(self):
        self.assertEqual(phrases.DIRECT_NAMES, DIRECT_NAMES)

    def test_pipeline_classes(self):
        self.assertEqual(phrases.PIPELINE_CLASSES, PIPELINE_CLASSES)


class MatchingTests(unittest.TestCase):
    def test_all_positive_phrases(self):
        for phrase in INPAINTING:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), ([phrase], []))

    def test_all_denials_remove_spans(self):
        for phrase in NOINPAINTING:
            with self.subTest(phrase=phrase):
                self.assertEqual(claims(phrase), ([], [phrase]))

    def test_longest_model(self):
        self.assertEqual(claims('image inpainting model')[0], ['image inpainting model'])

    def test_longest_models(self):
        self.assertEqual(claims('image inpainting models')[0], ['image inpainting models'])

    def test_case(self):
        self.assertEqual(claims('IMAGE INPAINTING')[0], ['image inpainting'])

    def test_collision(self):
        self.assertEqual(claims('inpaint image files by date')[0], ['inpaint image files'])

    def test_fences(self):
        self.assertEqual(claims('```text\nimage inpainting\n```'), ([], []))

    def test_inline(self):
        self.assertEqual(claims('`image inpainting`')[0], ['image inpainting'])

    def test_no_hyphen_normalization(self):
        self.assertEqual(claims('fills-missing-image-regions'), ([], []))

    def test_occurrences(self):
        self.assertEqual(claims('image inpainting; image inpainting')[0], ['image inpainting'] * 2)


class EvidenceTests(unittest.TestCase):
    def test_rejections(self):
        cases = [
            'import cv2\nimport skimage\nimport diffusers',
            'cv2.fastNlMeansDenoising(img)',
            'skimage.restoration.denoise_tv_chambolle(img)',
            'skimage.restoration.denoise_wavelet(img)',
            'cv2.GaussianBlur(img, (3,3), 0)',
            'PIL.ImageFilter.GaussianBlur(2)', 'cv2.medianBlur(img, 3)',
            'cv2.blur(img, (3,3))', 'cv2.boxFilter(img, -1, (3,3))',
            'skimage.filters.unsharp_mask(img)', 'cv2.filter2D(img, -1, kernel)',
            'model.upscale(img)', 'detector.detect(img)', 'model.predict(img)',
            'img.resize((100,100))', 'img.crop((0,0,1,1))', 'img.convert("RGB")',
            '"image inpainting"', '# image inpainter',
            'def f():\n    """image inpainting"""\n    pass',
            'from cv2 import inpaint\ninpaint',
            'import subprocess\nsubprocess.run(["image-inpainting", "a.png"])',
            'inpaint()',
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_shadowing(self):
        for name in DIRECT_NAMES + GENERIC_NAMES:
            for kind in ['def', 'class']:
                definition = ('def ' + name + '(*args):\n    pass\n' if kind == 'def'
                              else 'class ' + name + ':\n    pass\n')
                for call in [name + '()', 'obj.' + name + '()']:
                    with self.subTest(name=name, kind=kind, call=call):
                        self.assertFalse(evidence('from lib import ' + name + '\n' + definition + call))

    def test_pipeline_shadowing(self):
        for name in PIPELINE_CLASSES:
            self.assertFalse(evidence('from diffusers import ' + name + '\nclass ' + name + ':\n    pass\np = ' + name + '.from_pretrained("x")\np()'))

    def test_alias_shadowing(self):
        self.assertFalse(evidence('from cv2 import inpaint as ip\ndef ip():\n    pass\nip()'))

    def test_split_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo/'README.md').write_text('image inpainting')
            (repo/'a.py').write_text('from diffusers import StableDiffusionInpaintPipeline\np=StableDiffusionInpaintPipeline()')
            (repo/'b.py').write_text('p()')
            self.assertEqual(audit(repo)[0][0]['severity'], 'HIGH')

    def test_single_module_call_suffices(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            (repo/'README.md').write_text('image inpainting')
            (repo/'a.py').write_text('cv2.inpaint(x)')
            (repo/'b.py').write_text('x=1')
            self.assertEqual(audit(repo)[0], [])


def route_test(source):
    def test(self):
        self.assertTrue(evidence(source), source)
    return test


# Separate discoverable tests for every name and call form.
for route, names in [('direct', DIRECT_NAMES), ('generic', GENERIC_NAMES)]:
    for name in names:
        forms = {
            'attribute': 'obj.' + name + '(img)',
            'chained_attribute': 'obj.child.' + name + '(img)',
            'from_import': 'from library import ' + name + '\n' + name + '(img)',
            'aliased': 'from library import ' + name + ' as operation\noperation(img)',
            'module_alias': 'import library as lib\nlib.' + name + '(img)',
        }
        for form, source in forms.items():
            setattr(EvidenceTests, 'test_' + route + '_' + name + '_' + form, route_test(source))

for name in PIPELINE_CLASSES:
    for factory in ['', '.from_pretrained']:
        for chained in [False, True]:
            expr = name + factory + '("model")'
            source = 'from diffusers import ' + name + '\n'
            source += expr + '(img)' if chained else 'pipe = ' + expr + '\npipe(img)'
            setattr(EvidenceTests, 'test_pipeline_' + name + '_' + str(bool(factory)) + '_' + str(chained), route_test(source))
    def rejection(self, name=name):
        self.assertFalse(evidence('from diffusers import ' + name))
        self.assertFalse(evidence('from diffusers import ' + name + '\np=' + name + '()'))
        self.assertFalse(evidence('from diffusers import ' + name + '\np=' + name + '.from_pretrained("x")'))
    setattr(EvidenceTests, 'test_pipeline_incomplete_' + name, rejection)


for name in DIRECT_NAMES + GENERIC_NAMES:
    def incomplete(self, name=name):
        self.assertFalse(evidence('from library import ' + name))
        self.assertFalse(evidence('from library import ' + name + '\n' + name))
        self.assertFalse(evidence('obj.' + name))
    setattr(EvidenceTests, 'test_uncalled_' + name, incomplete)

for name in PIPELINE_CLASSES:
    for factory in ['', '.from_pretrained']:
        source = ('from diffusers import ' + name + ' as Pipeline\n'
                  + 'pipe = Pipeline' + factory + '("model")\npipe(img)')
        setattr(EvidenceTests, 'test_pipeline_alias_' + name + '_' + str(bool(factory)), route_test(source))


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)

    def write(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def cli(self, *args, before=False):
        command = ['python3', '-m', 'image_inpainting_claim_audit', 'audit']
        command += list(args) + [str(self.repo)] if before else [str(self.repo)] + list(args)
        return subprocess.run(command, capture_output=True, text=True)

    def test_good(self):
        self.write('README.md', 'image inpainting')
        self.write('src/app.py', 'cv2.inpaint(img, mask, 3, flag)')
        result = self.cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn('PASS', result.stdout)

    def test_bad(self):
        self.write('README.md', 'IMAGE INPAINTING')
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn('HIGH documented-image-inpainting-claim-false:', result.stdout)

    def test_test_only(self):
        self.write('tests/fixture.md', 'image inpainting')
        result = self.cli()
        self.assertEqual(result.returncode, 0)
        self.assertIn('MEDIUM', result.stdout)
        self.assertEqual(self.cli('--fail-on', 'medium').returncode, 2)

    def test_denial(self):
        self.write('docs/features.md', 'does not support image inpainting')
        self.write('main.py', 'obj.inpaint_image(img)')
        result = self.cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn('HIGH documented-no-image-inpainting-claim-false:', result.stdout)

    def test_denial_no_source(self):
        self.write('README.md', 'does not support image inpainting')
        self.assertEqual(self.cli().returncode, 0)

    def test_no_claim(self):
        self.write('README.md', 'A useful tool.')
        self.assertIn('PASS', self.cli().stdout)
        self.assertEqual(self.cli().returncode, 0)

    def test_json(self):
        self.write('README.md', 'image inpainting')
        result = self.cli('--json')
        self.assertEqual(result.returncode, 2)
        data = json.loads(result.stdout)
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertEqual(len(data['findings']), 1)
        self.assertEqual(data['findings'][0]['verdict'], 'FAIL')
        self.assertTrue(data['findings'][0]['message'])
        self.assertEqual(data['findings'][0]['rule'], 'documented-image-inpainting-claim-false')

    def test_version(self):
        result = subprocess.run(['python3', '-m', 'image_inpainting_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), '0.1.0')

    def test_ignore(self):
        self.write('docs/bad.md', 'image inpainting')
        self.assertEqual(self.cli().returncode, 2)
        self.assertEqual(self.cli('--ignore', 'docs/').returncode, 0)

    def test_ignore_before_positional(self):
        self.write('docs/bad.md', 'image inpainting')
        self.assertEqual(self.cli('--ignore', 'docs/', before=True).returncode, 0)

    def test_repeated_ignore(self):
        self.write('docs/a.md', 'image inpainting')
        self.write('tests/a.md', 'image inpainting')
        self.assertEqual(self.cli('--ignore', 'docs/', '--ignore', 'tests/', '--fail-on', 'medium').returncode, 0)

    def test_self_audit(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(claims((root/'README.md').read_text()), ([], []))
        result = subprocess.run(['python3', '-m', 'image_inpainting_claim_audit', 'audit', str(root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_verbose(self):
        self.write('README.md', 'nothing claimed')
        result = self.cli('--verbose', '--json')
        self.assertIn('Scope:', result.stderr)
        self.assertIn('README.md: prose scanned', result.stderr)
        self.assertEqual(json.loads(result.stdout)['verdict'], 'PASS')

    def test_errors(self):
        self.assertEqual(self.cli('--fail-on', 'invalid').returncode, 1)
        result = subprocess.run(['python3', '-m', 'image_inpainting_claim_audit', 'audit', str(self.repo/'missing')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)

    def test_scope_and_exclusions(self):
        self.write('notes.md', 'image inpainting')
        self.write('docs/nested/features.md', 'image inpainting')
        for name in ['.git/a.py', '__pycache__/a.py', '.venv/a.py', 'venv/a.py', 'custom/a.py']:
            self.write(name, 'obj.inpaint(img)')
        self.write('custom/pyvenv.cfg', '')
        self.assertEqual(self.cli().returncode, 2)
        self.assertEqual(self.cli('--ignore', 'docs/').returncode, 0)

    def test_families_collapse(self):
        self.write('README.md', 'image inpainting\nimage inpainting models')
        self.write('docs/a.md', 'inpaint images')
        self.assertEqual(len(audit(self.repo)[0]), 1)


if __name__ == '__main__':
    unittest.main()
